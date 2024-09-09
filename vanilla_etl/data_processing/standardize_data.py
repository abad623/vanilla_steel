import pandas as pd
from difflib import SequenceMatcher
import numpy as np
from prefect import task, get_run_logger
from vanilla_etl.models.bert_similarity import get_bert_embedding, cluster_words
from sklearn.metrics.pairwise import cosine_similarity


def is_fuzzy_match(word1: str, word2: str, threshold: float) -> bool:
    """
    Check if two words are similar based on fuzzy matching with a threshold.

    Args:
        word1 (str): First word.
        word2 (str): Second word.
        threshold (float): Similarity threshold.

    Returns:
        bool: True if the words are similar above the threshold, False otherwise.
    """
    similarity = SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
    return similarity > threshold


def collect_all_columns(dataframes: list) -> list:
    """
    Collect all unique columns from a list of DataFrames.

    Args:
        dataframes (list): List of processed DataFrames.

    Returns:
        list: List of all column names from the DataFrames.
    """
    all_columns = []
    for dfs in dataframes:
        if isinstance(dfs, list):
            for df in dfs:
                all_columns.extend(df.columns)
        else:
            all_columns.extend(dfs.columns)

    return list(set(all_columns))  # To keep unique columns only


def merge_dataframe(
    df: pd.DataFrame, column_mapping: dict, reference_columns: list
) -> pd.DataFrame:
    """
    Merge DataFrames based on cluster output and fuzzy matching.

    Args:
        df (pd.DataFrame): DataFrame to merge.
        column_mapping (dict): Clustered columns mapping.
        reference_columns (list): List of reference columns to standardize against.

    Returns:
        pd.DataFrame: Merged DataFrame with standardized column names.
    """
    for col in df.columns:
        for key, val in column_mapping.items():
            if key != "Unclassified":
                if col in [word[0] for word in val] and is_fuzzy_match(col, key, 0.6):
                    df.rename(columns={col: key}, inplace=True)
                    break
            else:
                for ref in reference_columns:
                    if is_fuzzy_match(col, ref, 0.7):
                        df.rename(columns={col: ref}, inplace=True)
                        break

    # Ensure all reference columns exist in the DataFrame
    for ref_col in reference_columns:
        if ref_col not in df.columns:
            df[ref_col] = np.nan

    # Drop extra columns that are not in reference columns
    df.drop(columns=[col for col in df.columns if col not in reference_columns], inplace=True)

    return df


def flatten_dataframes(dataframes: list) -> list:
    """
    Flatten a list of lists of DataFrames into a single list.

    Args:
        dataframes (list): List of DataFrames or lists of DataFrames.

    Returns:
        list: A flattened list of DataFrames.
    """
    flattened_list = []
    for df in dataframes:
        if isinstance(df, list):
            flattened_list.extend(df)
        else:
            flattened_list.append(df)
    return flattened_list


@task
def standardize_columns(dataframes: list, config: dict) -> tuple:
    """
    Standardize column names across multiple DataFrames based on BERT embeddings and fuzzy matching.

    Args:
        dataframes (list): List of DataFrames.
        config (dict): Configuration dictionary with thresholds and references.

    Returns:
        tuple:
            - pd.DataFrame: Combined DataFrame with standardized column names.
            - dict: Mapping of clustered column names.
    """
    threshold = config["cluster_threshold"]
    logger = get_run_logger()

    # Collect all unique columns from dataframes
    all_columns = collect_all_columns(dataframes)

    # Get BERT embeddings for reference and all columns
    reference_embeddings = get_bert_embedding(
        [col.lower() for col in config["table_header_reference"]]
    )
    column_embeddings = get_bert_embedding([col.lower() for col in all_columns])

    # Compute cosine similarity and cluster columns
    similarity_matrix = cosine_similarity(column_embeddings, reference_embeddings)
    column_clusters = cluster_words(
        similarity_matrix, all_columns, config["table_header_reference"], threshold
    )

    # Flatten and process the DataFrames
    flattened_dfs = flatten_dataframes(dataframes)
    standardized_dfs = []

    for df in flattened_dfs:
        standardized_df = merge_dataframe(df, column_clusters, config["table_header_reference"])
        standardized_dfs.append(standardized_df)

    # Concatenate all standardized DataFrames into one
    combined_df = pd.concat(standardized_dfs, ignore_index=True)

    logger.info("Standardization and merging of DataFrames completed.")
    return combined_df, column_clusters
