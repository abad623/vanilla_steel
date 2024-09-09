import yaml
import filetype
import os
from prefect import get_run_logger


def load_config_file(file_name: str) -> yaml:
    """loading the config file of project
    Args:
        file_name (str): [config file name]

    Returns:
        yaml: [parsed config]
    """
    with open(file_name, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_path_list(config: yaml) -> list:
    """create a list of full path to external files (e.g. Excel or PDF)

    Args:
        folder_path (str): [relative path tp folder]

    Returns:
        list: [list of available files with validation]
    """
    logger = get_run_logger()

    valid_file_list = []

    accpeted_file_type = config["file_types"]["valid_file_extensions"]
    dir_path = config["data_path"]["external"]

    for file_name in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file_name)):
            kind = filetype.guess(os.path.join(dir_path, file_name))
            if kind is None:  # Check if the file has a valid type and extension
                logger.error(
                    f"File{file_name} IS NOT A VALID FILE TYPE IN {dir_path}. (check config.yaml for more info.)"
                )
            elif (
                kind.extension in accpeted_file_type.keys()
                and kind.mime in accpeted_file_type.values()
            ):
                # add filename to list
                valid_file_list.append(os.path.join(dir_path, file_name))
            else:
                raise ValueError(f"{file_name} is not a valid format")

    return valid_file_list if len(valid_file_list) > 0 else None
