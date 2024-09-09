import csv


def save_predictions_with_conf_to_csv(predictions_list: list, output_file: str) -> None:
    """
    Saves predictions, true labels, and confidence scores to a CSV file.

    Args:
        predictions_list (list): List of tuples containing true labels, predicted labels, and confidence scores.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["True Label", "Predicted Label", "Confidence Score"])
        writer.writerows(predictions_list)


def save_predictions_only_to_csv(pred_only_list: list, output_file: str, label_dict: dict) -> None:
    """
    Saves only the predicted labels with the true labels as headers.

    Args:
        pred_only_list (list): List of predicted labels for each data point.
        output_file (str): Path to the output CSV file.
        label_dict (dict): Dictionary mapping label indices to label names.
    """
    headers = sorted(set(label_dict.values()))

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(pred_only_list)
