import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
from sklearn.metrics import classification_report
from tabulate import tabulate
from collections import defaultdict
import warnings

from sklearn.metrics import classification_report
from tabulate import tabulate

warnings.simplefilter("ignore")


def flatten_labels_and_predictions(true_labels, pred_labels, label_map):
    """
    Flatten lists of true and predicted labels, and map them back to original category names.

    Args:
        true_labels (list): Nested list of true labels (integers).
        pred_labels (list): Nested list of predicted labels (integers).
        label_map (dict): Dictionary mapping label indices to label names.

    Returns:
        Tuple of (flattened true labels, flattened predicted labels) as strings.
    """
    # Reverse label map (index to label name)
    idx_to_label = {v: k for k, v in label_map.items()}

    flattened_true_labels = []
    flattened_pred_labels = []

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for true, pred in zip(true_seq, pred_seq):
            if true != -100:  # Ignore padding label
                # Map the integer labels to their string equivalents
                flattened_true_labels.append(idx_to_label.get(true, "O"))
                flattened_pred_labels.append(idx_to_label.get(pred, "O"))

    return flattened_true_labels, flattened_pred_labels


def combine_bio_labels(flat_labels):
    """
    Combine 'B-' and 'I-' labels into a single entity for the classification report.
    """
    combined_labels = []
    for label in flat_labels:
        if label.startswith("B-") or label.startswith("I-"):
            combined_labels.append(label[2:])  # Remove "B-" or "I-" prefix
        else:
            combined_labels.append(label)
    return combined_labels


def show_classification_report(true_labels, pred_labels, label_map):
    """
    Show classification report for NER task by combining B- and I- prefixes for each entity.
    This version formats the report as a pretty table using `tabulate`.
    """
    flat_true_labels, flat_pred_labels = flatten_labels_and_predictions(
        true_labels, pred_labels, label_map
    )

    # Combine B- and I- labels to treat them as a single entity
    combined_true_labels = combine_bio_labels(flat_true_labels)
    combined_pred_labels = combine_bio_labels(flat_pred_labels)

    # Ensure the classification report includes all categories from the label map
    # all_categories = list(set(combine_bio_labels(list(label_map.values()))))

    report = classification_report(combined_true_labels, combined_pred_labels, output_dict=True)

    # Prepare table data for pretty output
    headers = ["Label", "Precision", "Recall", "F1-Score", "Support"]
    table_data = []
    for label, metrics in report.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:  # Skip these fields
            table_data.append(
                [
                    label,
                    round(metrics["precision"], 2),
                    round(metrics["recall"], 2),
                    round(metrics["f1-score"], 2),
                    int(metrics["support"]),
                ]
            )

    # Add macro average and weighted average if desired
    table_data.append(
        [
            "Macro Avg",
            round(report["macro avg"]["precision"], 2),
            round(report["macro avg"]["recall"], 2),
            round(report["macro avg"]["f1-score"], 2),
            int(report["macro avg"]["support"]),
        ]
    )

    table_data.append(
        [
            "Weighted Avg",
            round(report["weighted avg"]["precision"], 2),
            round(report["weighted avg"]["recall"], 2),
            round(report["weighted avg"]["f1-score"], 2),
            int(report["weighted avg"]["support"]),
        ]
    )

    # Print the table using tabulate
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
