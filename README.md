
# Vanilla Steel ETL Pipeline

This repository contains an ETL (Extract, Transform, Load) pipeline developed for Vanilla Steel Company. The pipeline is designed to process Excel files and transform the data based on specific configurations. The project follows the Cookiecutter framework, making it modular and easily extensible.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

### 1. Clone the repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/abad623/vanilla_steel.git
cd vanilla_steel_etl
```

### 2. Create and activate a virtual environment

It's recommended to use `virtualenv` to manage dependencies. Install `virtualenv` if you don't have it:

```bash
pip install virtualenv
```

Then, create a virtual environment and activate it:

```bash
# Create a virtual environment
virtualenv venv

# Activate the virtual environment
# For Windows
venv\Scripts\activate

# For Linux/Mac
source venv/bin/activate
```

### 3. Install the dependencies

Install all required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Install the package

This project uses a `pyproject.toml` for configuration. Install the package itself using the following command:

```bash
pip install .
```

Alternatively, if you are developing and want changes to take effect immediately:

```bash
pip install -e .
```

## Project Structure

Here is an overview of the main directories and files in this repository:

```plaintext
vanilla_etl/
│
├── data/                       # Data directory (Excel files should be placed here)
│   ├── external/               # External data
│   ├── interim/                # Interim data after transformation
│   └── processed/              # Final processed data
│
├── data_processing/            # Data processing scripts
│
├── docs/                       # Documentation
│
├── models/                     # Trained models and related artifacts
│
├── reports/                    # Generated analysis and reports
│
├── services/                   # Services for data extraction, transformation, etc.
│
├── tests/                      # Unit tests for the project
│
├── utils/                      # Utility functions
│
├── config.yaml                 # Configuration file for paths, file types, and processing options
│
├── main.py                     # Main entry point for running the ETL process
│
├── requirements.txt            # Python dependencies
│
├── pyproject.toml              # Project metadata and build configuration
│
├── README.md                   # This file
```

## Usage

### 1. Upload Excel Files

Before running the ETL pipeline, place your Excel files in the appropriate directory:

- Place the files you want to process in `vanilla_etl/data/external/`.

### 2. Configuration

Ensure that the `config.yaml` file is correctly configured with paths and settings. Here is a brief overview of key settings from the `config.yaml`:

- **Data paths**:
  - External: `vanilla_etl/data/external/`
  - Interim: `vanilla_etl/data/interim/`
  - Processed: `vanilla_etl/data/processed/`

- **File types**: Specifies the valid file extensions for the input data.

- **Column Headers**: The expected headers for your Excel files.

Refer to the `config.yaml` file for detailed settings:

```yaml
data_path:
    external: "vanilla_etl/data/external/"
    interim: "vanilla_etl/data/interim/"
    processed: "vanilla_etl/data/processed/"
```

### 3. Run the ETL pipeline

After configuring everything, run the ETL pipeline by executing the following command:

```bash
python main.py
```

This will start the pipeline, process the Excel files, and output the results to the `vanilla_etl/data/processed/` directory.


### 4.Configuration Flags

he ETL pipeline uses a centralized config.yaml file to manage parameters and flags. The train_configurations section of the YAML file allows you to control various pipeline behaviors:

```yaml
Train_configurations:
    train_mode: True          # If set to True, training mode is activated
    validation: True          # If True, validation will be performed
    dataset_sparsity_rate: 0.2  # Control the sparsity of your dataset
    data_synthesizing: True   # Enable or disable data augmentation
```
By keeping all necessary parameters centralized in the YAML file, the pipeline can be easily configured without modifying the code itself.


### 5.Monitoring

The ETL pipeline supports monitoring through a Prefect dashboard. Prefect is a workflow orchestration tool that allows you to monitor and control your data pipeline.

#### 1. Start Prefect server

To start the Prefect server and monitor the pipeline, run the following command:

```bash
prefect server start
```
#### 2. Access the dashboard

This will launch the Prefect dashboard where you can track the status of your ETL jobs.

```plaintext
http://localhost:8080
```
From here, you can monitor your pipeline's performance, view logs, and manage tasks.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you encounter any problems or have suggestions for improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
