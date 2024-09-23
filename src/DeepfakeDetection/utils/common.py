import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from DeepfakeDetection import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create a list of directories if they don't already exist or are not empty.
    Args:
        path_to_directories (list): List of path of directories
    """
    for path in path_to_directories:
        # Check if directory exists and has files
        if os.path.exists(path):
            if verbose:
                logger.info(
                    f"Directory at {path} already exists and contains files. Skipping creation."
                )
            continue  # Skip creating the directory if it's not empty
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def get_size_in_kbs(path: Path) -> int:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        int: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return size_in_kb
