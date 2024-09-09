import pandas as pd
from prefect import flow, task, get_run_logger
from deep_translator import GoogleTranslator
import re


def contains_letters(text: str) -> bool:
    """
    Check if the input string contains at least one letter.
    Args:
        text (str): Input string.

    Returns:
        bool: True if the string contains letters, False otherwise.
    """
    return bool(re.search(r"[a-zA-Z]", text)) if isinstance(text, str) else False


@task
def translate_dataframes(dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Translates the content of the dataframes and headers into English using Google Translator.

    Args:
        dataframes (pd.DataFrame): List of DataFrames to translate.

    Returns:
        pd.DataFrame: List of translated DataFrames.
    """
    logger = get_run_logger()
    translated_frames = []

    try:
        for idx, df in enumerate(dataframes):
            logger.info(f"Translating table {idx + 1} of {len(dataframes)}")

            # Translate column headers
            df.columns = [
                GoogleTranslator(source="auto", target="en").translate(str(col))
                for col in df.columns
            ]

            # Translate cell values (if column is object type and contains letters)
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].apply(
                        lambda cell: (
                            GoogleTranslator(source="auto", target="en").translate(cell)
                            if pd.notnull(cell) and contains_letters(cell)
                            else cell
                        )
                    )

            translated_frames.append(df)

        logger.info("Translation completed for all tables.")
        return translated_frames

    except Exception as error:
        logger.error(f"Translation process failed: {error}")
        raise
