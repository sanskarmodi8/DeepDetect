import os
import zipfile

import gdown

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the DataIngestion class with the provided configuration.

        Args:
            config (DataIngestionConfig): Configuration object that contains the necessary parameters
                                          for data ingestion such as source URL, local file paths, etc.
        """
        self.config = config

    def download_file(self) -> str:
        """
        Download the dataset file from the specified Google Drive URL.

        Returns:
            str: Path to the downloaded zip file.

        Raises:
            Exception: If any error occurs during the download process, it is logged and raised.
        """
        try:
            # URL of the dataset to be downloaded
            dataset_url = self.config.source_url

            # Local directory path where the downloaded file will be saved
            zip_download_dir = self.config.local_data_file

            # Ensure that the directory for storing downloaded files exists
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            logger.info(
                f"Downloading data from {dataset_url} into file {zip_download_dir}"
            )

            # Extract the file ID from the Google Drive URL
            file_id = dataset_url.split("/")[-2]

            # Google Drive URL prefix for direct download
            prefix = "https://drive.google.com/uc?/export=download&id="

            # Download the file using gdown
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(
                f"Downloaded data from {dataset_url} into file {zip_download_dir}"
            )

        except Exception as e:
            # Log any errors that occur during download
            logger.info(e)
            # Re-raise the exception to halt execution and notify the calling function
            raise e

    def extract_zip_file(self):
        """
        Extract the contents of the downloaded zip file into the specified directory.

        This function extracts all files and directories from the zip file to the path specified in the configuration.

        Args:
            None

        Returns:
            None
        """
        # Path where the extracted contents will be stored
        unzip_path = self.config.unzip_dir

        # Ensure that the directory for extracted files exists
        os.makedirs(unzip_path, exist_ok=True)

        # Open the zip file in read mode and extract all contents
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
