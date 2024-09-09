import pandas as pd
from prefect import flow, task, get_run_logger
from difflib import SequenceMatcher
import re


def fuzzy_match_column(headers, target="Material", threshold=0.5):
    """
    Use fuzzy matching to select a column name based on similarity to the target string.

    Args:
        headers (list): List of column names.
        target (str): Target string to match against.
        threshold (float): Similarity threshold for matching.

    Returns:
        str: Matched column name or None if no match found.
    """
    match_ratios = [SequenceMatcher(None, target.lower(), col.lower()).ratio() for col in headers]

    try:
        return headers[next(i for i, ratio in enumerate(match_ratios) if ratio > threshold)]
    except StopIteration:
        return None


def insert_spaces_in_description(text: str, config) -> str:
    """
    Insert spaces around keywords in the description based on gazetteers.

    Args:
        text (str): Description text to process.
        config (dict): Config dictionary with gazetteers.

    Returns:
        str: Processed text with spaces inserted.
    """
    for grade in config["grade_gazetteer"]:
        text = text.replace(grade, f" {grade} ")
    for coating in config["coating_gazetteer"]:
        text = text.replace(coating, f" {coating} ")

    return re.sub(r" +", " ", text).strip()


def validate_finish_type(finish_text: str, config) -> str:
    """
    Validate the finish components in a string against a predefined list.

    Args:
        finish_text (str): Finish string to validate.
        config (dict): Configuration with finish gazetteer.

    Returns:
        str: Validated finish string or None if invalid.
    """
    parts = re.findall(r"[A-Z0-9/.]+", finish_text.upper())
    valid_parts = [part for part in parts if part in config["finish_gazetteer"]]

    if valid_parts:
        return " ".join(valid_parts)
    return None


def parse_dimensions(dim_text: str) -> dict:
    """
    Parse dimension string and return a dictionary with dimensions.

    Args:
        dim_text (str): Dimension string to process.

    Returns:
        dict: Parsed dimensions (Thickness, Width, Length).
    """
    # Clean up dimension text by removing units and spaces
    dim_text = re.sub(r"(mm|cm|m)", "", dim_text.lower())
    dim_text = dim_text.replace(",", ".").strip()

    # Split the dimensions based on common delimiters like x or *
    dimensions = re.split(r"[xX*]", dim_text)

    # Convert string dimensions to float values and handle two or three dimensions
    try:
        normalized_dims = [float(dim.strip()) for dim in dimensions]
    except ValueError:
        normalized_dims = []

    if len(normalized_dims) == 3:
        # Case with three dimensions: Thickness x Width x Length
        return {
            "Thickness (mm)": normalized_dims[0],
            "Width (mm)": normalized_dims[1],
            "Length (mm)": normalized_dims[2],
        }
    elif len(normalized_dims) == 2:
        # Case with two dimensions: Width x Length
        return {
            "Thickness (mm)": None,
            "Width (mm)": normalized_dims[0],
            "Length (mm)": normalized_dims[1],
        }
    elif len(normalized_dims) == 1:
        # Case with only one dimension (Width or Length)
        return {
            "Thickness (mm)": None,
            "Width (mm)": normalized_dims[0],
            "Length (mm)": None,
        }
    else:
        return {"Thickness (mm)": None, "Width (mm)": None, "Length (mm)": None}


def parse_material_description(description: str, config) -> dict:
    """
    Parse material description and extract relevant information (grade, coating, dimensions, finish).

    Args:
        description (str): Material description text.
        config (dict): Configuration with gazetteers and patterns.

    Returns:
        dict: Parsed material information.
    """
    description = insert_spaces_in_description(description, config)

    grade = None
    coating = None
    dimensions = None
    finish = ""

    # Extract grade
    for grade_item in config["grade_gazetteer"]:
        if grade_item in description.upper():
            grade = grade_item
            description = description.replace(grade_item, "")
            break

    # Remove units like mm, cm, m
    description = re.sub(r"(mm|cm|m)", "", description.lower())

    # Clean comma issues before dimension parsing
    description = description.replace(",", ".").strip()

    # Extract dimensions using regex
    dim_match = re.search(
        r"(\d+(\.\d+)?\s*[*x]\s*\d+(\.\d+)?(?:\s*[*x]\s*\d+(\.\d+)?)?)", description
    )
    if dim_match:
        dimensions = parse_dimensions(dim_match.group(0))
        description = description.replace(dim_match.group(0), "")

    # Remove any + symbols for coatings
    description = description.replace("+", "")

    # Extract coating and remaining description
    remaining_parts = description.split()

    for part in remaining_parts:
        part = part.strip()
        # Check if the part is a known coating from the gazetteer
        if part.upper() in config["coating_gazetteer"]:
            coating = part if not coating else coating  # Ensure we only capture the coating once
        else:
            finish += f"{part} "  # Add other parts to finish

    # Clean finish string further
    finish = finish.strip()

    return {
        "Grade": grade,
        "Coating": coating,
        "Thickness (mm)": dimensions.get("Thickness (mm)", None) if dimensions else None,
        "Width (mm)": dimensions.get("Width (mm)", None) if dimensions else None,
        "Length (mm)": dimensions.get("Length (mm)", None) if dimensions else None,
        "Finish": finish if finish else None,  # Only return finish if it's non-empty
    }


@task
def process_material_dataframes(dataframes: list, config: dict) -> list:
    """
    Process multiple DataFrames to extract and expand material descriptions.

    Args:
        dataframes (list): List of DataFrames to process.
        config (dict): Configuration for parsing material information.

    Returns:
        list: List of expanded DataFrames with material information.
    """
    logger = get_run_logger()
    expanded_dataframes = []

    for idx, df in enumerate(dataframes):
        results = []
        col_name = fuzzy_match_column(df.columns)

        if col_name:
            for description in df[col_name]:
                parsed_info = parse_material_description(description, config)
                results.append(parsed_info)

            expanded_df = pd.concat([df, pd.DataFrame(results)], axis=1)
            expanded_dataframes.append(expanded_df)
            logger.info(f"DataFrame {idx} expanded successfully.")
        else:
            expanded_dataframes.append(df)

    return expanded_dataframes
