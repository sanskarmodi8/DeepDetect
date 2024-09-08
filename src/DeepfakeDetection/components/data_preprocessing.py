import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import DataPreprocessingConfig
from DeepfakeDetection.utils.common import create_directories, save_h5py


# Strategy interfaces
class FrameExtractionStrategy(ABC):
    @abstractmethod
    def extract_frames(self, video_path: str, max_frames: int):
        """Method to be implemented for extracting frames from a video."""
        pass


class FaceDetectionStrategy(ABC):
    @abstractmethod
    def detect_faces(self, frame: np.ndarray):
        """Method to be implemented for detecting faces from a frame."""
        pass


# Strategy for frame extraction
class FFmpegFrameExtraction(FrameExtractionStrategy):
    def __init__(self, scene_change_threshold):
        """
        Initialize the FFmpegFrameExtraction with a scene change threshold.

        Args:
        scene_change_threshold (float): Threshold to determine if a frame is a scene change.
        """
        self.scene_change_threshold = scene_change_threshold

    def extract_frames(self, video_path: str, max_frames: int):
        """
        Extract frames from a video using FFmpeg where scene changes are significant.

        Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to extract.

        Returns:
        np.ndarray: A NumPy array of frames, shape (n_frames, height, width, 3).

        Raises:
        ffmpeg.Error: If an error occurs while extracting frames.
        """
        try:
            # Probe video to get metadata
            probe = ffmpeg.probe(video_path)
            video_info = next(
                stream for stream in probe["streams"] if stream["codec_type"] == "video"
            )
            duration = float(video_info["duration"])
            width = int(video_info["width"])
            height = int(video_info["height"])

            # Calculate frame rate for extracting desired number of frames
            fps = max_frames / duration

            # Use ffmpeg to extract frames where scene changes are significant
            out, _ = (
                ffmpeg.input(video_path)
                .filter("select", f"gt(scene,{self.scene_change_threshold})")
                .filter("fps", fps=fps)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24")
                .run(
                    capture_stdout=True,
                    capture_stderr=True,
                    cmd=["ffmpeg", "-hwaccel", "cuda"],
                )
            )

            # Convert the raw output into a NumPy array of frames
            frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            return frames

        except ffmpeg.Error as e:
            logger.error(f"Error selecting key frames from video {video_path}: {e}")
            return np.array([])


# Strategy for face detection using OpenCV haar cascades
class OpenCVFaceDetection(FaceDetectionStrategy):
    def __init__(self, scale_factor, min_neighbors, min_size):
        """
        Parameters
        ----------
        scale_factor : float
            Parameter specifying how much the image size is reduced at each scale.
        min_neighbors : int
            Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        min_size : list
            Parameter specifying the minimum possible object size in the output image.
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect_faces(self, frame: np.ndarray):
        """
        Detect faces in a frame using OpenCV's Haar cascade classifier.

        Parameters
        ----------
        frame : np.ndarray
            Input frame to detect faces in.

        Returns
        -------
        face_crops : list
            List of detected face crops.
        """
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Check for CUDA support in OpenCV
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gpu_gray.download(),
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )

        face_crops = []
        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            face_crops.append(face)
        return face_crops


class DataPreprocessing:
    """
    A class to handle preprocessing of video data for deepfake detection.

    This class handles the extraction of frames from videos, detection of faces in frames,
    and the saving of preprocessed data into HDF5 files. It also handles stratified splitting
    of the dataset and checkpointing of the preprocessing progress.

    Attributes:
        config (DataPreprocessingConfig): Configuration for preprocessing.
        frame_extraction_strategy (FrameExtractionStrategy): Strategy for extracting frames from videos.
        face_detection_strategy (FaceDetectionStrategy): Strategy for detecting faces in video frames.
        checkpoint_file (str): Path to the checkpoint file used to track preprocessing progress.
        checkpoint (dict): Dictionary storing the last processed index for each data split.
    """

    def __init__(self, config: DataPreprocessingConfig):
        """
        Initializes the DataPreprocessing class with given configuration, frame extraction, and face detection strategies.

        Args:
            config (DataPreprocessingConfig): The configuration object for preprocessing.
            frame_extraction_strategy (FrameExtractionStrategy): The strategy for extracting frames from videos.
            face_detection_strategy (FaceDetectionStrategy): The strategy for detecting faces in frames.
        """
        self.config = config
        self.frame_extraction_strategy = (
            FFmpegFrameExtraction(self.config.scene_change_threshold),
        )
        self.face_detection_strategy = (
            OpenCVFaceDetection(
                self.config.scale_factor,
                self.config.min_neighbors,
                self.config.min_size,
            ),
        )
        self.checkpoint_file = os.path.join(
            self.config.root_dir, "preprocessing_checkpoint.json"
        )
        self.load_checkpoint()

    def load_checkpoint(self):
        """
        Loads the preprocessing checkpoint if it exists, or initializes a new one if not.

        The checkpoint tracks the progress of preprocessing for each data split (train, val, test)
        to allow for resuming from where the process left off.
        """
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                self.checkpoint = json.load(f)
            logger.info("Checkpoint loaded successfully.")
        else:
            self.checkpoint = {"train": 0, "val": 0, "test": 0}
            logger.info("No checkpoint found. Initialized new checkpoint.")

    def save_checkpoint(self, split_name: str, index: int):
        """
        Saves the current preprocessing progress to the checkpoint file for a given split.

        Args:
            split_name (str): The name of the data split (train, val, or test).
            index (int): The index of the last processed video file in the split.
        """
        self.checkpoint[split_name] = index
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.checkpoint, f)
        logger.info(f"Checkpoint updated for {split_name} split at index {index}.")

    def preprocess_frame(self, frame: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Preprocesses a given frame by resizing and normalizing it.

        Args:
            frame (np.ndarray): The input frame to be preprocessed.
            target_size (tuple): The target size to resize the frame.

        Returns:
            np.ndarray: The preprocessed frame.
        """
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        frame = tf.image.resize(frame, target_size)
        frame = frame / 255.0
        return frame.numpy()

    def process_video(self, video_path: str, label: str, max_frames: int):
        """
        Extracts and preprocesses frames from a video file, applying face detection if applicable.

        Args:
            video_path (str): Path to the video file.
            label (str): The label of the video (e.g., "original" or "manipulated").
            max_frames (int): Maximum number of frames to extract from the video.

        Returns:
            list: A list of preprocessed frames.
        """
        key_frames = self.frame_extraction_strategy.extract_frames(
            video_path, max_frames
        )
        processed_frames = []
        for frame in key_frames:
            faces = self.face_detection_strategy.detect_faces(frame)
            if faces:
                processed_frame = self.preprocess_frame(
                    faces[0], tuple(self.config.target_size)
                )
            else:
                processed_frame = self.preprocess_frame(
                    frame, tuple(self.config.target_size)
                )
            processed_frames.append(processed_frame)
        return processed_frames

    def save_dataset(self, data: list, labels: list, split_name: str):
        """
        Saves preprocessed data and labels to HDF5 files for a given data split.

        Args:
            data (list): A list of preprocessed frames.
            labels (list): A list of labels corresponding to the frames.
            split_name (str): The name of the data split (train, val, or test).
        """
        split_dir = os.path.join(
            self.config.output_data, "preprocessed_data", split_name
        )
        data_dir = os.path.join(split_dir, "data")
        labels_dir = os.path.join(split_dir, "labels")
        labels = [0 if label == "original" else 1 for label in labels]
        create_directories([split_dir, data_dir, labels_dir])
        data_file = os.path.join(data_dir, "data.h5")
        labels_file = os.path.join(labels_dir, "labels.h5")
        save_h5py(np.array(data), Path(data_file), dataset_name="data")
        save_h5py(np.array(labels), Path(labels_file), dataset_name="labels")
        logger.info(f"Appended {split_name} dataset with {len(data)} frames.")

    def process_and_save_split(self, split_name: str, split_files: list):
        """
        Processes and saves a dataset split by extracting and preprocessing frames from videos.

        Args:
            split_name (str): The name of the data split (train, val, or test).
            split_files (list): A list of video files to be processed.
        """
        data, labels = [], []
        start_index = self.checkpoint.get(split_name, 0)
        files_to_process = split_files[start_index:]

        for idx, video_file in enumerate(
            tqdm(files_to_process, desc=f"Processing {split_name} data")
        ):
            video_path = os.path.join(
                self.config.data_path, video_file[0], video_file[1]
            )
            label = video_file[0]
            frames = self.process_video(video_path, label, self.config.max_frames)
            if not frames:
                continue
            data.extend(frames)
            labels.extend([label] * len(frames))
            if idx % 10 == 0:
                self.save_dataset(data, labels, split_name)
                self.save_checkpoint(split_name, start_index + idx + 1)
                data, labels = [], []

        if data:
            self.save_dataset(data, labels, split_name)

    def stratified_split(
        self, video_files: list, test_size: float = 0.15, val_size: float = 0.15
    ):
        """
        Splits the dataset into train, validation, and test sets in a stratified manner.

        Args:
            video_files (list): A list of video file tuples where each tuple contains the label and filename.
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.

        Returns:
            tuple: Three lists of files corresponding to the train, validation, and test splits.
        """
        labels = np.array([file[0] for file in video_files])
        stratified_split = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=42
        )
        train_indices, test_indices = next(stratified_split.split(video_files, labels))
        train_files = [video_files[i] for i in train_indices]
        test_files = [video_files[i] for i in test_indices]
        val_size_adjusted = val_size / (1 - test_size)
        stratified_split_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size_adjusted, random_state=42
        )
        train_indices_final, val_indices = next(
            stratified_split_val.split(train_files, labels[train_indices])
        )
        val_files = [train_files[i] for i in val_indices]
        train_files_final = [train_files[i] for i in train_indices_final]
        return train_files_final, val_files, test_files

    def run(self):
        """
        Executes the entire preprocessing pipeline, processing and saving the train, validation, and test splits.
        """
        files = [
            (folder, file)
            for folder in ["manipulated", "original"]
            for file in sorted(os.listdir(os.path.join(self.config.data_path, folder)))
        ]

        train_files, val_files, test_files = self.stratified_split(files)

        logger.info("Processing train split...")
        self.process_and_save_split("train", train_files)

        logger.info("Processing validation split...")
        self.process_and_save_split("val", val_files)

        logger.info("Processing test split...")
        self.process_and_save_split("test", test_files)
