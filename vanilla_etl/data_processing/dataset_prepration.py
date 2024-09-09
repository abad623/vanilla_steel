import os
import random
import numpy as np
import pandas as pd
import torch
import string
from collections import Counter
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from typing import List, Tuple, Dict, Any
from prefect import flow, task, get_run_logger
from sklearn.preprocessing import LabelEncoder


# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


class NERDataset(Dataset):
    def __init__(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        labels: List[List[int]],
        char_ids: List[List[List[int]]],
        cat_features: Dict[str, List[Any]],
        num_features: Dict[str, List[float]],
        num_vocab_size: int,
        char_vocab_size: int,
    ):
        """
        Dataset class for NER that incorporates tokenized inputs, labels, character-level IDs,
        as well as categorical and numeric features.
        """
        self.inputs = inputs
        self.labels = labels
        self.char_ids = char_ids
        self.cat_features = cat_features
        self.num_features = num_features
        self.num_vocab_size = num_vocab_size
        self.char_vocab_size = char_vocab_size

        # Encode categorical features as tensors
        self.categorical_vocab_sizes = {key: len(set(cat_features[key])) for key in cat_features}
        for key in self.cat_features:
            self.cat_features[key] = torch.tensor(self.cat_features[key], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        input_ids = self.inputs[idx]["input_ids"].squeeze()
        attention_mask = self.inputs[idx]["attention_mask"].squeeze()
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        char_ids = torch.tensor(self.char_ids[idx], dtype=torch.long)

        seq_len = input_ids.size(0)

        # categorical_data = {key: self.cat_features[key][idx] for key in self.cat_features}
        categorical_data = {key: self.cat_features[key][idx] for key in self.cat_features}

        # Handle categorical features
        # categorical_data = []
        # for key in self.cat_features:
        #    value = self.cat_features[key][idx]

        # Ensure value is not a list or multi-dimensional data
        #  if isinstance(value, list) or isinstance(value, np.ndarray):
        #      raise ValueError(f"Expected single value for key '{key}', got list/array: {value}")

        # Handle missing values (None or NaN)
        #   if pd.isna(value):
        #       value = -1  # Default value for missing categorical data

        #   try:
        #       # Append each value as a tensor
        #       categorical_data.append(torch.tensor(value, dtype=torch.long))
        #   except Exception as e:
        #       print(f"Error converting categorical value for key '{key}': {value}, Error: {e}")
        #       raise ValueError(f"Failed to convert categorical value for key '{key}'")

        # if categorical_data:
        #    categorical_data = torch.stack(categorical_data, dim=0)

        # Handle numeric features
        numeric_data = []
        for key in self.num_features:
            value = self.num_features[key][idx]

            # Handle missing numeric values (None or NaN)
            if pd.isna(value):
                value = 0.0  # Default value for missing numeric data

            # Ensure value is a tensor
            value = torch.tensor(value, dtype=torch.float)

            # Repeat the numeric feature for the entire sequence length
            value = value.unsqueeze(0).repeat(seq_len, 1)  # Shape: (seq_len, 5)

            numeric_data.append(value)

        # Concatenate all numeric tensors into a single tensor for each token in the sequence
        if numeric_data:
            numeric_data = torch.stack(numeric_data, dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "char_ids": char_ids,
            "categorical_features": categorical_data,
            "numeric_features": numeric_data,
        }


def tokenize_and_align(
    texts: List[List[str]], labels: List[List[str]], max_length: int = 20
) -> Tuple[List[Dict[str, torch.Tensor]], List[List[int]]]:
    """
    Tokenize text and align corresponding labels with subword tokens.
    """
    tokenized_inputs = []
    aligned_labels = []

    for text, label in zip(texts, labels):
        tokenized_input = tokenizer(
            text,
            is_split_into_words=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        word_ids = tokenized_input.word_ids()
        label_alignment = []

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_alignment.append(-100)  # Padding tokens get -100
            elif word_idx != previous_word_idx:
                label_alignment.append(label[word_idx])
            else:
                label_alignment.append(-100)
            previous_word_idx = word_idx

        tokenized_inputs.append(tokenized_input)
        aligned_labels.append(label_alignment)

    return tokenized_inputs, aligned_labels


def char_ids_extraction(
    texts: List[List[str]], max_word_len: int = 20, max_seq_len: int = 20
) -> List[List[List[int]]]:
    """
    Extract character-level IDs for tokens, limiting to max_word_len.
    Pads or truncates character IDs to max_seq_len for consistent tensor shapes.

    Args:
        texts (List[List[str]]): List of tokenized texts.
        max_word_len (int, optional): Maximum number of characters per word. Defaults to 20.
        max_seq_len (int, optional): Maximum sequence length for padding. Defaults to 9.

    Returns:
        List[List[List[int]]]: Character-level IDs, padded to max_seq_len and max_word_len.
    """
    char_ids = []
    for text in texts:
        word_chars = [[ord(char) for char in word[:max_word_len]] for word in text]

        # Pad or truncate the word_chars to max_seq_len
        word_chars = word_chars[:max_seq_len]  # Truncate to max_seq_len
        word_chars += [[0] * max_word_len] * (max_seq_len - len(word_chars))  # Pad with zeros

        # Ensure each word's characters are padded to max_word_len
        word_chars = [
            chars[:max_word_len] + [0] * (max_word_len - len(chars)) for chars in word_chars
        ]

        char_ids.append(word_chars)
    return char_ids


def generate_text_and_labels(
    df: pd.DataFrame, config: Dict[str, Any], swap_prob: float = 0.3
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Generate text and labels from a DataFrame, with an optional swap for data augmentation.
    """
    text_data, label_data = [], []
    label_mapping = config["label_dict"]

    # Iterate through each row to generate text and label sequences
    for _, row in df.iterrows():
        text, labels = [], []
        row_values = row.copy()  # Copy row to manipulate column swapping

        # Determine column order to shuffle (based on swap probability)
        columns = list(df.columns)
        if random.random() < swap_prob:
            random.shuffle(columns)  # Shuffle the column order

        # Iterate through the shuffled columns and generate text/labels
        for col in columns:
            value = str(row_values[col])  # Get the value of the column
            words = value.split()  # Tokenize the value if it has multiple words
            text.extend(words)  # Add tokens to the text list

            # Assign labels based on the column's mapped entity
            entity_label = label_mapping[col]
            labels.append(f"B-{entity_label}")  # First token gets a "B-" label
            labels.extend(
                [f"I-{entity_label}"] * (len(words) - 1)
            )  # Remaining tokens get "I-" labels

        text_data.append(text)
        label_data.append(labels)

    return text_data, label_data


def extract_features(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[Dict[str, List[Any]], Dict[str, List[float]]]:
    """
    Extract categorical and numeric features from DataFrame based on configuration.
    """
    categorical, numeric = {}, {}

    for col in df.columns:
        col_type = config["column_type_dict"].get(col)
        if col_type == "categorical":
            categorical[col] = df[col].tolist()
        elif col_type == "numeric":
            # numeric[col] = df[col].astype(float).tolist()
            numeric[col] = [[float(v) for v in str(val).split()] for val in df[col]]

    return categorical, numeric


def fill_missing_values(df: pd.DataFrame, missing_rate: float) -> pd.DataFrame:
    """
    Randomly fill missing values in categorical and numeric columns.
    """
    cat_cols = df.select_dtypes(include=["object"]).columns
    num_cols = df.select_dtypes(include=[np.number]).columns

    total_values = df.size
    missing_count = int(missing_rate * total_values)
    random.seed(42)
    random_indices = random.sample(
        [(r, c) for r in range(df.shape[0]) for c in range(df.shape[1])], missing_count
    )

    for row, col in random_indices:
        col_name = df.columns[col]
        df.iat[row, col] = "[UNK]" if col_name in cat_cols else -1

    df[cat_cols] = df[cat_cols].fillna("[UNK]")
    df[num_cols] = df[num_cols].fillna(-1)

    return df


def build_char_vocab(texts: list) -> dict:
    """
    Build a character-level vocabulary from a list of texts.

    Args:
        texts (list): List of texts.

    Returns:
        dict: Character vocabulary and its size.
    """
    # Initialize an empty Counter for characters
    char_counter = Counter()

    # Iterate through each text and update the character count
    for text in texts:
        for word in text:
            char_counter.update(list(word))  # Split each word into characters

    # Create a list of unique characters sorted in order
    char_vocab = sorted(char_counter.keys())

    # Add special tokens (optional)
    char_vocab.extend(["<PAD>", "<UNK>"])  # Padding and unknown characters

    # Create a character-to-index mapping
    char_to_idx = {char: idx for idx, char in enumerate(char_vocab)}

    # Return the character vocabulary and its size
    return char_to_idx, len(char_to_idx)


def encode_categorical_features(cat_features: Dict[str, List[Any]]) -> Dict[str, List[int]]:

    label_encoders = {}
    encoded_cat_features = {}

    for col, values in cat_features.items():
        # Create a LabelEncoder for each categorical column
        label_encoders[col] = LabelEncoder()
        encoded_cat_features[col] = label_encoders[col].fit_transform(values).tolist()

    return encoded_cat_features, label_encoders


def prepare_training_data(config: Dict[str, Any], data: pd.DataFrame) -> NERDataset:
    """
    Prepare data for training: filling missing values, tokenizing, extracting features, etc.
    """
    logger = get_run_logger()
    sparsity = config["train_configurations"]["dataset_sparsity_rate"]

    logger.info(f"Processing {data.shape[0]} samples for training")

    data = fill_missing_values(data.copy(), missing_rate=sparsity)
    logger.info("Missing values filled")

    texts, labels = generate_text_and_labels(data, config)

    char_to_idx, char_vocab_size = build_char_vocab(texts)

    tokenized_inputs, aligned_labels = tokenize_and_align(texts, labels)
    char_ids = char_ids_extraction(texts)

    cat_features, num_features = extract_features(data, config)
    encoded_cat_features, label_encoders = encode_categorical_features(cat_features)

    logger.info("Tokenization, feature extraction, and alignment complete")

    label_map = config["label_map"]
    numeric_labels = [
        [label_map.get(lbl, -100) for lbl in label_seq] for label_seq in aligned_labels
    ]

    num_vocab_size = len(set([lbl for label_seq in numeric_labels for lbl in label_seq]))

    return NERDataset(
        tokenized_inputs,
        numeric_labels,
        char_ids,
        encoded_cat_features,
        num_features,
        num_vocab_size,
        char_vocab_size,
    )
