import json
import os
from pathlib import Path
from typing import Any

import h5py
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
    """create list of directories
    Args:
        path_to_directories (list): list of path of directories
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


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


def save_h5py(data: np.ndarray, path: Path, dataset_name="data", compression="gzip"):
    """save data to an HDF5 file using h5py
    Args:
        data (np.ndarray): data to be saved
        path (Path): path to the HDF5 file
        dataset_name (str): name of the dataset in the HDF5 file
        compression (str): compression method (default: "gzip")
    """
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset(dataset_name, data=data, compression=compression)

    logger.info(f"HDF5 file saved at: {path}")


def load_h5py(path: Path, dataset_name="data") -> Any:
    """load data from an HDF5 file using h5py
    Args:
        path (Path): path to the HDF5 file
        dataset_name (str): name of the dataset in the HDF5 file
    Returns:
        Any: data loaded from the file
    """
    with h5py.File(path, "r") as h5f:
        data = np.array(h5f[dataset_name])

    logger.info(f"HDF5 file loaded from: {path}")
    return data


@ensure_annotations
def get_size_in_kbs(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
