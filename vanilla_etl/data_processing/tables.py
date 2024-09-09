import pandas as pd
from prefect import task, get_run_logger
from vanilla_etl.models.bert_similarity import identify_headers_with_bert


def find_potential_headers(df: pd.DataFrame, config) -> int:
    """
    Find potential header rows and select the best candidate based on string and non-null value distribution.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (dict): Configuration dictionary for BERT processing.

    Returns:
        int: Index of the identified header row.
    """
    logger = get_run_logger()
    potential_header_indices = []

    for idx, row in df.iterrows():
        non_null_count = row.notnull().sum()
        string_count = row.apply(lambda x: isinstance(x, str)).sum()

        # Identify rows where 70% are non-null and 50% are strings
        if non_null_count > len(row) * 0.7 and string_count > len(row) * 0.5:
            potential_header_indices.append(idx)

    # Use BERT to identify headers
    bert_header_index = identify_headers_with_bert(df, config)

    if potential_header_indices:
        heuristic_header_idx = max(
            potential_header_indices, key=lambda idx: df.iloc[idx].notnull().sum()
        )

        if bert_header_index == heuristic_header_idx:
            logger.info("BERT and heuristic methods agree on the header index.")
        else:
            logger.warning("BERT and heuristic methods do not agree on the header index.")
        return heuristic_header_idx
    elif bert_header_index is not None:
        logger.warning("Using BERT to identify header as no heuristic candidates were found.")
        return bert_header_index
    else:
        logger.error("No valid header found. Defaulting to index 0.")
        return 0


def create_dataframe_from_table(table: pd.DataFrame, header_row_idx: int) -> pd.DataFrame:
    """
    Create a DataFrame by extracting headers and table boundaries.

    Args:
        table (pd.DataFrame): Input DataFrame representing a table.
        header_row_idx (int): Index of the header row.

    Returns:
        pd.DataFrame: DataFrame with headers applied, cleaned of empty rows and columns.
    """
    logger = get_run_logger()

    try:
        df_with_headers = pd.DataFrame(
            table.values[header_row_idx:], columns=table.iloc[header_row_idx]
        )
        df_with_headers.dropna(how="all", inplace=True)
        df_with_headers = df_with_headers.reset_index(drop=True)

        # Drop the header row itself
        df_with_headers.drop(index=0, inplace=True)

        # Sanity check: Ensure the DataFrame has at least 2 columns and data
        if df_with_headers.empty or len(df_with_headers.columns) < 2:
            return None

        return df_with_headers
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        return None


@task
def identify_tables(df: pd.DataFrame, config) -> list:
    """
    Identify and extract tables from an Excel sheet, splitting by empty rows.

    Args:
        df (pd.DataFrame): DataFrame representing an Excel sheet.
        config (dict): Configuration dictionary for BERT header identification.

    Returns:
        list: List of DataFrames representing extracted tables.
    """
    logger = get_run_logger()
    extracted_tables = []

    if df.empty:
        logger.warning("Input DataFrame is empty. No tables to extract.")
        return extracted_tables

    start_idx = None
    current_table = None

    for idx, row in df.iterrows():
        if not row.isnull().all():
            if start_idx is None:
                start_idx = idx
                current_table = row.to_frame().T
            else:
                current_table = pd.concat([current_table, row.to_frame().T], ignore_index=True)
        else:
            if start_idx is not None:
                # Clean the current table and attempt to extract headers
                cleaned_table = current_table.dropna(how="all", axis=0).dropna(how="all", axis=1)
                header_idx = find_potential_headers(cleaned_table, config)
                table_df = create_dataframe_from_table(cleaned_table, header_idx)

                if table_df is not None:
                    extracted_tables.append(table_df)

                # Reset for the next table
                start_idx = None
                current_table = None

    # Handle the last table in the DataFrame if it doesn't end with an empty row
    if start_idx is not None and current_table is not None:
        header_idx = find_potential_headers(current_table, config)
        table_df = create_dataframe_from_table(current_table, header_idx)
        if table_df is not None:
            extracted_tables.append(table_df)

    logger.info(f"Number of detected tables: {len(extracted_tables)}")
    for tbl in extracted_tables:
        logger.info(f"Extracted headers: {tbl.columns}")

    return extracted_tables
