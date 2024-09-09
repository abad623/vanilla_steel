from prefect import task, get_run_logger
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


def identify_headers_with_bert(df: pd.DataFrame, config) -> int:
    """
    Identify the most likely header row in a DataFrame using BERT embeddings.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
        config (dict): Configuration containing reference header information.

    Returns:
        int: Index of the identified header row.
    """
    header_reference = config["table_header_reference"]
    reference_text = " ".join(header_reference)

    reference_embedding = get_bert_embedding(reference_text)
    max_similarity = -1
    header_idx = 0

    for i, row in df.iterrows():
        row_text = " ".join([str(cell) for cell in row.dropna()])
        if row_text:
            row_embedding = get_bert_embedding(row_text)
            similarity = cosine_similarity(row_embedding, reference_embedding).mean()

            if similarity > max_similarity:
                max_similarity = similarity
                header_idx = i

    return header_idx


def get_bert_embedding(text: str) -> np.ndarray:
    """
    Generate a BERT embedding for a given text.

    Args:
        text (str): The text to be encoded by BERT.

    Returns:
        np.ndarray: BERT embedding vector for the text.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embedding
    except Exception as e:
        raise ValueError(f"Error generating BERT embedding for '{text}': {e}")


def cluster_words(
    similarity_matrix: np.ndarray, words: list, references: list, threshold: float
) -> dict:
    """
    Cluster words based on their cosine similarity to reference words.

    Args:
        similarity_matrix (np.ndarray): Matrix of similarity scores.
        words (list): List of words to be clustered.
        references (list): Reference words for clustering.
        threshold (float): Similarity threshold for clustering.

    Returns:
        dict: Dictionary of clustered words.
    """
    clusters = {ref: [] for ref in references}
    clusters["Unclassified"] = []

    for i, word in enumerate(words):
        max_similarity = np.max(similarity_matrix[i])
        max_index = np.argmax(similarity_matrix[i])

        if max_similarity >= threshold:
            clusters[references[max_index]].append((word, max_similarity))
        else:
            clusters["Unclassified"].append((word, max_similarity))

    return clusters
