import pandas as pd
from prefect import task, get_run_logger
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTENC, RandomOverSampler
import random
import numpy as np


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values for key columns in the DataFrame with defaults or averages.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    fill_defaults = {
        "description": "Unknown",
        "article_id": "0000/00000",
        "grade": "Unknown",
        "coating": "Unknown",
        "finish": "Unknown",
    }
    df.fillna(fill_defaults, inplace=True)

    fill_mean_cols = ["weight", "quantity", "height (mm)", "width (mm)", "length (mm)"]
    for col in fill_mean_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    return df


def generate_synthetic_data(num_samples: int, combinations: dict) -> pd.DataFrame:
    """
    Generate synthetic data based on valid combinations of material attributes.

    Args:
        num_samples (int): Number of synthetic samples to generate.
        combinations (dict): Dictionary of valid material attribute combinations.

    Returns:
        pd.DataFrame: DataFrame containing synthetic data.
    """
    synthetic_records = []

    for _ in range(num_samples):
        grade_tuple = random.choice(list(combinations.keys()))
        grade = grade_tuple[0]
        coating = random.choice(combinations[grade_tuple]["coating"])
        finish = random.choice(combinations[grade_tuple]["finish"])
        height = round(random.uniform(*combinations[grade_tuple]["height_range"]), 2)
        width = round(random.uniform(*combinations[grade_tuple]["width_range"]), 2)
        length = round(random.uniform(*combinations[grade_tuple]["length_range"]), 2)

        if random.choice([True, False]):
            dimensions = f"{height}x{width}x{length}"
        else:
            dimensions = f"{width}x{length}"

        weight = round(random.uniform(1200, 3000), 2)
        quantity = round(random.uniform(10, 60), 2)

        material = f"{grade} +{coating} {finish} {dimensions}"
        description = random.choice(combinations[grade_tuple]["description"])
        article_id = int(random.uniform(1000000, 9999999))

        synthetic_records.append(
            [
                article_id,
                material,
                description,
                weight,
                quantity,
                grade,
                coating,
                finish,
                height,
                width,
                length,
            ]
        )

    # Convert to DataFrame
    return pd.DataFrame(
        synthetic_records,
        columns=[
            "article_id",
            "material",
            "description",
            "weight",
            "quantity",
            "grade",
            "coating",
            "finish",
            "height (mm)",
            "width (mm)",
            "length (mm)",
        ],
    )


@task
def run_data_synthesizer(config: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize additional data based on valid combinations from the provided DataFrame.

    Args:
        config (dict): Configuration object containing parameters like `augmented_data`.
        df (pd.DataFrame): Original DataFrame to base the synthetic data generation on.

    Returns:
        pd.DataFrame: DataFrame with original and synthetic data combined.
    """
    logger = get_run_logger()
    logger.info("Starting data synthesis process...")

    data_combinations = {}

    # Group data by grade, coating, and finish to create valid combinations
    for grade, group in df.groupby(["grade", "coating", "finish"]):
        data_combinations[grade] = {
            "article_id": df["article_id"].unique(),
            "description": df["description"].unique(),
            "coating": group["coating"].unique(),
            "finish": group["finish"].unique(),
            "height_range": (df["height (mm)"].min(), df["height (mm)"].max()),
            "width_range": (df["width (mm)"].min(), df["width (mm)"].max()),
            "length_range": (df["length (mm)"].min(), df["length (mm)"].max()),
        }

    # Generate synthetic data
    synthetic_df = generate_synthetic_data(
        num_samples=config["augmented_data"], combinations=data_combinations
    )

    # Combine original and synthetic data
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)

    logger.info(f"Data synthesis completed. Added {len(synthetic_df)} synthetic samples.")

    return combined_df
