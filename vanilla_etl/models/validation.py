import torch
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from vanilla_etl.reports.metrics_report import show_classification_report
from vanilla_etl.utils.save_outputs import (
    save_predictions_with_conf_to_csv,
    save_predictions_only_to_csv,
)

torch.mps.empty_cache()


def validate_model(model, val_loader, device, label_map, fold, epoch, output_file_pred_csv):
    """
    Validates the model and saves predictions to CSV files.

    Args:
        model: The trained model.
        val_loader: DataLoader for the validation set.
        device: Device where the model is being run (CPU/GPU).
        label_dict: Dictionary mapping label indices to label names.
        fold: Fold number in cross-validation.
        epoch: Current epoch number.
        output_file_pred_true_conf: File path for saving predictions with confidence scores.
        output_file_pred_only: File path for saving prediction-only results.

    Returns:
        tuple: Average validation loss, validation accuracy, precision, recall, F1 scores.
    """
    model.eval()
    total_val_loss, correct_val_preds, total_val_preds = 0, 0, 0
    true_labels, pred_labels = [], []
    predictions_list = []
    pred_only_list = []

    for batch in val_loader:
        with torch.no_grad():
            loss, logits = model(
                input_ids=batch["input_ids"].to(device),
                numeric_features=batch["numeric_features"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                char_ids=batch["char_ids"].to(device),
                categorical_features={
                    key: batch["categorical_features"][key].to(device)
                    for key in batch["categorical_features"]
                },
                labels=batch["labels"].to(device),
            )

            total_val_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()

            confidence_scores, _ = torch.max(logits, axis=-1)  # Extract confidence scores

            mask = labels != -100  # Ignore padding
            correct_val_preds += np.sum(preds[mask] == labels[mask])
            total_val_preds += np.sum(mask)

            true_labels_masked = labels[mask].tolist()
            pred_labels_masked = preds[mask].tolist()
            confidence_scores_masked = confidence_scores[mask].tolist()

            predictions_list.extend(
                zip(true_labels_masked, pred_labels_masked, confidence_scores_masked)
            )
            pred_only_list.append([label_map.get(pred, "O") for pred in pred_labels_masked])

            true_labels.extend(true_labels_masked)
            pred_labels.extend(pred_labels_masked)

    val_accuracy = correct_val_preds / total_val_preds
    avg_val_loss = total_val_loss / len(val_loader)

    precision = precision_score(true_labels, pred_labels, average=None, zero_division=0)
    recall = recall_score(true_labels, pred_labels, average=None, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)

    # print(f"precison: {precision}, recall: {recall}, f1-measure:{f1}")
    print(f"val acc:{val_accuracy}  avg val loss {avg_val_loss}")

    show_classification_report(labels, preds, label_map)

    # if output_file_pred_csv:
    #    save_predictions_with_conf_to_csv(
    #        predictions_list, f"{os.path.join([output_file_pred_csv,fold])}_fold_{epoch}_conf.csv"
    #    )
    #    save_predictions_only_to_csv(
    #        pred_only_list,
    #       f"{os.path.join([output_file_pred_csv,fold])}_fold_{epoch}pred.csv",
    #       label_dict,
    #   )

    return avg_val_loss, val_accuracy, precision, recall, f1, true_labels, pred_labels
