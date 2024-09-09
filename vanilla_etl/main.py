from services.pipeline_service import run_pipeline
from utils.helper import load_config_file


# Main funtion for programm inpyt
def main():
    config = load_config_file(file_name="config.yaml")
    run_pipeline(config)


if __name__ == "__main__":
    main()
