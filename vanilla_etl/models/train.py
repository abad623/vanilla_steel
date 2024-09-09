import torch
import torch.nn as nn
from transformers import BertModel, AdamW, AutoConfig
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os
from transformers import DistilBertForTokenClassification
from prefect import flow, task, get_run_logger
from vanilla_etl.models.validation import validate_model
from vanilla_etl.reports.plot_generator import plot_metrics_per_fold, plot_loss_accuracy
from vanilla_etl.utils.save_outputs import (
    save_predictions_with_conf_to_csv,
    save_predictions_only_to_csv,
)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Set MPS high watermark for memory usage on M1 machines
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class CharacterCNN(nn.Module):
    def __init__(
        self,
        char_vocab_size,
        char_embedding_dim=30,
        kernel_size=3,
        num_filters=50,
        max_word_len=20,
    ):
        super(CharacterCNN, self).__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(char_embedding_dim, num_filters, kernel_size, padding="same")
        self.max_pool = nn.MaxPool1d(max_word_len)
        self.relu = nn.ReLU()

    def forward(self, char_ids):
        batch_size, seq_len, max_word_len = char_ids.size()
        char_ids = char_ids.view(
            -1, max_word_len
        )  # Flatten to (batch_size * seq_len, max_word_len)
        char_embeds = self.char_embedding(char_ids)
        char_embeds = char_embeds.transpose(
            1, 2
        )  # Convert to (batch_size * seq_len, char_embedding_dim, max_word_len)
        conv_out = self.relu(self.conv1d(char_embeds))
        pooled_out = self.max_pool(conv_out).squeeze(2)  # (batch_size * seq_len, num_filters)
        return pooled_out.view(
            batch_size, seq_len, -1
        )  # Reshape to (batch_size, seq_len, num_filters)


class BERTWithCharAndNumericEmbeddingForNER(nn.Module):
    def __init__(
        self,
        bert_model_name,
        char_vocab_size,
        num_labels,
        numeric_feature_dim,
        categorical_vocab_sizes,
        cat_embedding_dim=50,
        max_word_len=20,
    ):
        super(BERTWithCharAndNumericEmbeddingForNER, self).__init__()
        configuration = AutoConfig.from_pretrained(bert_model_name)
        configuration.hidden_dropout_prob = 0.3
        configuration.attention_probs_dropout_prob = 0.3
        self.bert_model = BertModel.from_pretrained(bert_model_name, config=configuration)
        self.char_cnn = CharacterCNN(char_vocab_size, max_word_len=max_word_len)
        # self.numeric_embedding = nn.Linear(
        #     numeric_feature_dim, 50)
        self.numeric_embedding = nn.Linear(5, 50)
        # Example for numeric feature embedding
        self.dropout = nn.Dropout(0.1)
        # Assign num_labels as an attribute of the class
        self.num_labels = num_labels

        # Create embeddings for each categorical feature

        self.categorical_embeddings = nn.ModuleDict(
            {
                str(key): nn.Embedding(vocab_size, cat_embedding_dim)
                for key, vocab_size in categorical_vocab_sizes.items()
            }
        )

        # The final layer size will be BERT hidden size + char embeddings + numeric + categorical embeddings
        combined_embedding_size = (
            self.bert_model.config.hidden_size
            + 50
            + 50
            + (len(categorical_vocab_sizes) * cat_embedding_dim)
        )
        self.classifier = nn.Linear(combined_embedding_size, num_labels)

    #  self.classifier = nn.Linear(
    #      self.bert.config.hidden_size + 50 + 50, num_labels
    #  )  # BERT + char + numeric

    def forward(
        self,
        input_ids,
        attention_mask,
        char_ids,
        numeric_features,
        categorical_features,
        labels=None,
    ):
        bert_outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs[0]  # (batch_size, seq_len, hidden_size)

        # Character embeddings
        char_embeds = self.char_cnn(char_ids)  # (batch_size, seq_len, num_filters)

        # Ensure numeric features are in the shape (batch_size, seq_len, numeric_feature_dim)
        if numeric_features.dim() == 2:
            # Add a sequence dimension if numeric_features are just (batch_size, numeric_feature_dim)
            numeric_features = numeric_features.unsqueeze(1).expand(
                -1, sequence_output.size(1), -1
            )

        # Numeric embeddings
        # Reshape numeric features: (batch_size, seq_len, 5) - Remove the extra dimension
        numeric_features = numeric_features.squeeze(-1)
        numeric_embeds = self.numeric_embedding(numeric_features)  # (batch_size, seq_len, 50)

        categorical_embeds = []

        for key in self.categorical_embeddings:
            feature_values = categorical_features[key].to(
                input_ids.device
            )  # Ensure correct device
            embedded_values = self.categorical_embeddings[key](feature_values)

            # Ensure embedded values are expanded to match the sequence length (seq_len)
            # embedded_values has shape (batch_size, cat_embedding_dim), so we expand it
            embedded_values = embedded_values.unsqueeze(1).expand(-1, sequence_output.size(1), -1)
            categorical_embeds.append(embedded_values)

        categorical_embeds = torch.cat(categorical_embeds, dim=-1)

        # Concatenate BERT embeddings with character, numeric, and categorical embeddings
        combined_output = torch.cat(
            [sequence_output, char_embeds, numeric_embeds, categorical_embeds], dim=2
        )

        combined_output = self.dropout(combined_output)
        logits = self.classifier(combined_output)  # (batch_size, seq_len, num_labels)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits


def get_model(config, dataset):
    """
    Initialize and return the BERT-based model.
    Args:
        config: Model configuration dictionary
    """
    model = BERTWithCharAndNumericEmbeddingForNER(
        bert_model_name="bert-base-uncased",
        char_vocab_size=dataset.char_vocab_size,
        num_labels=len(config["label_map"]),
        numeric_feature_dim=dataset.num_vocab_size,
        categorical_vocab_sizes=dataset.categorical_vocab_sizes,
        max_word_len=20,
    )

    model.to(device)
    return model


@task
def cross_validate_model(config, dataset, k=5, epochs=3):
    """
    Performs k-fold cross-validation with BERT model, logs training metrics, and saves the model.
    Args:
        config: Dictionary containing training configurations and paths.
        dataset: Dataset to use for training and validation.
        k: Number of folds for k-fold cross-validation.
        epochs: Number of epochs to train the model in each fold.
    """

    model = get_model(config, dataset)

    label_dict = config["label_dict"]
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    print(model)
    print(f"Model starts learning!. Please wait...")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold + 1}/{k}")

        # Create train and validation subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create DataLoaders for train and validation sets
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=config["train_configurations"]["train_params"]["batch_size"],
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=config["train_configurations"]["train_params"]["batch_size"]
        )

        # Define optimizer
        optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

        # Training loop for epochs
        for epoch in range(epochs):
            avg_train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer)
            avg_val_loss, val_accuracy, precision, recall, f1, true_labels, pred_labels = (
                validate_model(
                    model,
                    val_loader,
                    device,
                    config["label_map"],
                    fold,
                    epoch,
                    config["data_path"]["processed"],
                )
            )

            # Print training and validation metrics
            print(
                f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        # Log evaluation metrics and plot fold metrics
    # show_evaluation_metrics(true_labels, pred_labels, label_dict)
    # plot_metrics_per_fold(fold, true_labels, pred_labels, label_dict)


def train_one_epoch(model, train_loader, optimizer):
    """
    Trains the model for one epoch, logs training metrics, and returns the average loss and accuracy.
    """
    model.train()
    logger = get_run_logger()

    total_train_loss, correct_train_preds, total_train_preds = 0, 0, 0

    for _, batch in enumerate(train_loader):

        optimizer.zero_grad()

        # Forward pass
        loss, logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            char_ids=batch["char_ids"].to(device),
            numeric_features=batch["numeric_features"].to(device),
            labels=batch["labels"].to(device),
            categorical_features={
                key: batch["categorical_features"][key].to(device)
                for key in batch["categorical_features"]
            },
        )

        # Calculate loss outside the model
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(logits.view(-1, model.num_labels), batch["labels"].view(-1).to(device))

        total_train_loss += loss.item()

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()

        # Ignore padding labels (-100)
        mask = labels != -100
        correct_train_preds += np.sum(preds[mask] == labels[mask])
        total_train_preds += np.sum(mask)

    train_accuracy = correct_train_preds / total_train_preds
    avg_train_loss = total_train_loss / len(train_loader)

    return avg_train_loss, train_accuracy


@task
def train_model_only(config, dataset, epochs=3):
    """
    Trains the model on the entire dataset for a specified number of epochs, logs metrics, and saves the model.
    Args:
        config: Dictionary with training configurations and paths.
        dataset: Dataset to use for training.
        epochs: Number of epochs to train the model.
    """
    logger = get_run_logger()

    model = get_model(config, dataset)

    save_dir = config["model_path"]["artifacts"]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create DataLoader for the dataset
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train_configurations"]["train_params"]["batch_size"],
        shuffle=True,
    )

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    train_losses, train_accuracies = [], []

    print(f"Model starts learning!. Please wait...")
    for epoch in range(epochs):
        avg_train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Save the model after each epoch
        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pt")

        torch.save(model.state_dict(), model_save_path)

        logger.info(
            f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
        )

    torch.mps.empty_cache()

    # Plot loss and accuracy curves
    plot_loss_accuracy(train_losses, train_accuracies)
