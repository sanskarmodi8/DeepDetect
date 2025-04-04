import os
import random
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import DataIngestionConfig
from DeepfakeDetection.utils.common import create_directories


class DataFactory(ABC):
    """
    Abstract base class for creating data handlers.
    """

    @abstractmethod
    def handle_data(self, config: DataIngestionConfig) -> str:
        """
        Abstract method to handle data.

        Args:
            config (DataIngestionConfig): Configuration object with paths and URLs.

        Returns:
            str: Path to the processed data.
        """
        pass


class FaceForensicsplusplusFactory(DataFactory):
    """
    Factory for handling FaceForensics++ data.
    """

    def handle_data(self, config: DataIngestionConfig) -> str:
        """
        Handles FaceForensics++ data by directly using it and sampling videos.

        Args:
            config (DataIngestionConfig): Configuration object with local folder path.

        Returns:
            str: Path to the data folder.
        """
        self.sample_videos_from_folder(
            config.source_data, config.final_data_path, config.num_videos
        )
        return config.source_data

    def sample_videos_from_folder(
        self, source_folder, destination_folder, num_videos=1200
    ):
        """
        Samples videos from the given source folder and moves them to the destination folder.

        Args:
            source_folder (str): Path to the source folder containing the data.
            destination_folder (str): Path to the destination folder where sampled videos will be saved.
        """

        # Check if the destination folder exists
        if not os.path.exists(destination_folder):
            raise ValueError(f"Destination folder {destination_folder} does not exist.")

        # Check if the video files are already present in the destination folder
        if len(os.listdir(destination_folder)) > 0:
            logger.info(
                f"Destination folder {destination_folder} is not empty. Skipping sampling."
            )
            return

        # Check if the source_data folder exists
        if not os.path.exists(source_folder):
            raise ValueError(f"Source folder {source_folder} does not exist.")

        original_folder = Path(source_folder) / "original_sequences"
        manipulated_folder = Path(source_folder) / "manipulated_sequences"
        sampled_originals_folder = Path(destination_folder) / "original"

        create_directories([sampled_originals_folder])

        # Sample original videos
        original_videos = []
        for cat in original_folder.iterdir():
            if cat.is_dir():
                original_videos.extend(list(Path(cat / "c23/videos").glob("*.mp4")))

        sampled_originals = random.sample(
            original_videos, min(num_videos, len(original_videos))
        )
        for video in sampled_originals:
            shutil.copy(video, sampled_originals_folder / video.name)

        # Sample manipulated videos (keeping each category separate)
        manipulated_categories = [
            cat for cat in manipulated_folder.iterdir() if cat.is_dir()
        ]

        for category in manipulated_categories:
            category_name = category.name  # Example: "faceswap", "deepfakes", etc.
            category_videos_path = category / "c23/videos"

            if not category_videos_path.exists():
                continue  # Skip if videos don't exist

            category_videos = list(category_videos_path.glob("*.mp4"))
            sampled_category_videos = random.sample(
                category_videos, min(num_videos, len(category_videos))
            )

            sampled_category_folder = Path(destination_folder) / category_name
            create_directories([sampled_category_folder])

            for video in sampled_category_videos:
                shutil.copy(video, sampled_category_folder / video.name)

        logger.info(f"Sampled {len(sampled_originals)} original videos.")
        for category in manipulated_categories:
            category_name = category.name
            category_folder = Path(destination_folder) / category_name
            num_sampled = len(list(category_folder.glob("*.mp4")))
            logger.info(f"Sampled {num_sampled} videos for category: {category_name}")


class DataIngestion:
    """
    Manages the data ingestion process.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes DataIngestion with a configuration object.

        Args:
            config (DataIngestionConfig): Configuration object with paths and URLs.
        """
        self.config = config
        self.factory = FaceForensicsplusplusFactory()

    def run(self):
        """
        Executes the data ingestion process by handling the data source and sampling videos.
        """
        if os.path.exists(self.config.final_data_path):
            logger.info(
                f"Data path {self.config.final_data_path} already exists. Skipping data ingestion."
            )
            return

        create_directories([self.config.final_data_path])
        data_path = self.factory.handle_data(self.config)
        logger.info(f"Data has been sampled. Sampled data path: {data_path}")
