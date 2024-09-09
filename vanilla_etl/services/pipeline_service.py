from vanilla_etl.utils.helper import generate_path_list
from vanilla_etl.data_processing.process_excel import load_data_concurrently
from vanilla_etl.data_processing.deep_translation import translate_dataframes
from vanilla_etl.data_processing.standardize_data import standardize_columns
from vanilla_etl.data_processing.process_materials import process_material_dataframes
from vanilla_etl.data_processing.data_consistancy import evaluate_data_consistency
from vanilla_etl.data_processing.synthesize_data import run_data_synthesizer
from vanilla_etl.data_processing.dataset_prepration import prepare_training_data
from vanilla_etl.models.train import train_model_only, cross_validate_model
import pandas as pd
from prefect import flow, task, get_run_logger


@flow(name="Vanilla Steel ETL Pipeline")
def run_pipeline(config: dict) -> None:
    """
    End-to-end pipeline for processing documents, extracting tables,
    standardizing outputs, and training models.

    Args:
        config (dict): Configuration object from a YAML file.
    """
    logger = get_run_logger()
    try:
        # Step 1: Generate a list of file paths (Excel files).
        file_paths = generate_path_list(config)

        # EXTRACT PHASE
        # Step 2: Load and extract tables from Excel sheets.
        extracted_dataframes = load_data_concurrently(file_paths, config)

        # TRANSORM PHASES

        # Step 3: Translate the table contents to English.
        translated_dataframes = [translate_dataframes.submit(df) for df in extracted_dataframes]

        # Step 3.5: Evaluate consistency of the standardized dataset (optional).
        validated_dataframe = [evaluate_data_consistency(df) for df in translated_dataframes]

        # Step 4: Process material columns to extract and expand relevant components.
        expanded_materials = [
            process_material_dataframes(df, config) for df in validated_dataframe
        ]

        # LOAD PHASES

        # Step 5: Standardize column names using BERT embeddings and cosine similarity.
        standardized_dataframe, column_mapping = standardize_columns(expanded_materials, config)

        # Step 6: Save the standardized DataFrame as a JSON file if it contains data.
        if not standardized_dataframe.empty:
            output_path = f"{config['data_path']['processed']}standard_output.json"
            standardized_dataframe.to_json(output_path, orient="columns")
            logger.info(f"Standardized DataFrame saved successfully at {output_path}.")

        # TRAINIGN PHASE

        # Step 7: Check if training mode is enabled in the configuration.
        if config["train_configurations"]["train_mode"]:
            # Step 9: Optionally synthesize additional data for training.
            if config["train_configurations"]["data_synthesizing"]:
                synthesized_dataset = run_data_synthesizer(config, standardized_dataframe)
            else:
                synthesized_dataset = standardized_dataframe

            # Step 10: Generate training data from the synthesized dataset.
            training_dataset = prepare_training_data(config, synthesized_dataset)

            # Step 11: Either cross-validate or train the model based on configuration.
            epochs = config["train_configurations"]["train_params"]["num_epochs"]
            if config["train_configurations"]["validation"]:
                cross_validate_model(
                    config,
                    dataset=training_dataset,
                    k=5,
                    epochs=epochs,
                )
            else:
                train_model_only(
                    config,
                    training_dataset,
                    epochs=epochs,
                )

        logger.info("Vanilla Steel ETL Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")
        raise


if __name__ == "__main__":
    run_pipeline()
