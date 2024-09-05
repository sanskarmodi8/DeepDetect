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


def save_h5py(
    data: np.ndarray, file_path: Path, dataset_name="data", compression="gzip"
):
    """Append data to an HDF5 file if the dataset exists, or create a new dataset if it doesn't.
    Args:
        data (np.ndarray): Data to be saved or appended.
        path (Path): Path to the HDF5 file.
        dataset_name (str): Name of the dataset in the HDF5 file.
        compression (str): Compression method (default: "gzip").
    """
    with h5py.File(file_path, "a") as f:
        if dataset_name in f:
            # Append to existing dataset
            dataset = f[dataset_name]
            dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
            dataset[-data.shape[0] :] = data
        else:
            # Create new dataset
            f.create_dataset(dataset_name, data=data, maxshape=(None,) + data.shape[1:])
    logger.info(f"HDF5 file updated at: {file_path}")


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
