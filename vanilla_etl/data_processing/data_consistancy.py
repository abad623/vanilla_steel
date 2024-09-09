import numpy as np
import pandas as pd
from prefect import task, get_run_logger


@task
def evaluate_data_consistency(data_frames: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates the consistency of the data by checking for high variance, null values,
    and duplicates. Cleans the data by dropping unnecessary rows or columns based on thresholds.
    """
    logger = get_run_logger()

    try:
        # Evaluate the standard deviation vs. mean for numeric columns
        dflist = []
        for data_frame in data_frames:
            for column in data_frame.columns:
                if data_frame[column].dtype in ["int64", "float64"]:
                    column_std = data_frame[column].std()
                    column_mean = data_frame[column].mean()

                    # Check for high variance
                    if column_std > column_mean * 0.5:
                        logger.warning(f"Column {column} has a high standard deviation.")

            threshold = len(data_frame) * 0.1
            # Drop columns where more than 90% of the values are missing
            data_frame.dropna(thresh=threshold, axis=1, inplace=True)

            threshold = len(data_frame.columns) * 0.1
            # Drop rows where more than 90% of the values are missing
            data_frame.dropna(thresh=threshold, axis=0, inplace=True)

            # Optional: Drop duplicate rows and keep the first
            data_frame.drop_duplicates(keep="first", inplace=True)
            dflist.append(data_frame)

            (data_frame.info())

        logger.info("Data consistency check completed.")

        return dflist

    except Exception as error:
        logger.error(f"Data consistency check failed: {error}")
        raise
