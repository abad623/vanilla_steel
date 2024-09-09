import matplotlib
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_score, recall_score, f1_score

matplotlib.use("agg")


def plot_metrics_per_fold(
    fold: int, true_labels: list, pred_labels: list, label_dict: dict
) -> None:
    """
    Plots precision, recall, and F1-score for each combined category per fold.

    Args:
        fold (int): The current fold number.
        true_labels (list): List of true labels as indices.
        pred_labels (list): List of predicted labels as indices.
        label_dict (dict): Dictionary mapping label indices to label names.
    """
    combined_labels, precision, recall, f1 = calculate_combined_metrics(
        true_labels, pred_labels, label_dict
    )

    x = range(len(combined_labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, precision, width=0.2, label="Precision", align="center")
    plt.bar([i + 0.2 for i in x], recall, width=0.2, label="Recall", align="center")
    plt.bar([i + 0.4 for i in x], f1, width=0.2, label="F1-Score", align="center")

    plt.xticks([i + 0.2 for i in x], combined_labels, rotation=45, ha="right")
    plt.title(f"Fold {fold} - Precision, Recall, F1-Score for Each Combined Label")
    plt.xlabel("Entity Label")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"vanilla_etl/reports/figures/loss_accuracy_per_folder_{fold}.png")


def plot_loss_accuracy(train_losses, train_accuracies):
    """
    Plots the training loss and accuracy over epochs.
    """
    epochs = len(train_losses)
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label="Training Loss", color="blue")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label="Training Accuracy", color="blue")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(f"vanilla_etl/reports/figures/loss_accuracy_{random.randint(1,10)}.png")


def plot_embedding_tsne(emb_dic: dict) -> None:
    """Plot the embedding point with tsne"""
    labels = list(emb_dic.keys())

    pca = PCA(n_components=(len(labels) - 1), svd_solver="full")
    embeddings_pca = pca.fit_transform(np.concatenate(list(emb_dic.values()), axis=0))

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_pca)

    # Create a scatter plot
    plt.figure(figsize=(10, 7))
    for i, label in enumerate(labels):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9)

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid()
    plt.savefig(f"reports/figues/tsne_header_detection_bert_{random.randint(1,10)}.png")


def get_bert_embeddings_temp(reference_embeddings, word_embeddings, header_reference, words):

    # Combine embeddings for visualization
    all_embeddings = np.concatenate((reference_embeddings, word_embeddings), axis=0)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    # Split reduced embeddings into reference and word embeddings
    reference_reduced = reduced_embeddings[: len(header_reference)]
    word_reduced = reduced_embeddings[len(header_reference) :]

    # Visualize clusters
    plt.figure(figsize=(12, 8))

    # Plot reference points (cluster centers)
    for i, ref in enumerate(header_reference):
        plt.scatter(
            reference_reduced[i, 0],
            reference_reduced[i, 1],
            color="red",
            marker="x",
            s=200,
            label=ref if i == 0 else "",
        )
        plt.text(reference_reduced[i, 0], reference_reduced[i, 1], ref, fontsize=12, color="black")

    # Plot word points
    for i, word in enumerate(words):
        plt.scatter(word_reduced[i, 0], word_reduced[i, 1], color="blue", marker="o", s=100)
        plt.text(word_reduced[i, 0], word_reduced[i, 1], word, fontsize=9, color="gray")

    plt.title("t-SNE Visualization of Word Clusters Based on Reference Columns")
    plt.legend(loc="best")
    plt.savefig(f"/reports/figgures/tsne_header_detection_bert_{random.randint(1,10)}.png")
