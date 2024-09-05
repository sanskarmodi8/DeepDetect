import json
import os
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


class DataPreprocessing:
    """
    A class to handle the preprocessing of video data for deepfake detection.

    This class includes methods for:
    - Loading and saving processing checkpoints.
    - Extracting and preprocessing frames from videos using ffmpeg.
    - Detecting and extracting faces from frames using OpenCV with CUDA acceleration.
    - Performing stratified splitting of data into training, validation, and test sets.
    - Saving preprocessed data and labels into HDF5 format.
    """

    def __init__(self, config: DataPreprocessingConfig):
        """
        Initializes the DataPreprocessing instance with the given configuration.

        Args:
            config (DataPreprocessingConfig): Configuration object containing all necessary parameters.
        """
        self.config = config
        # Create output directory if it doesn't exist
        create_directories([self.config.output_dir])
        # Define the path for the checkpoint file
        self.checkpoint_file = os.path.join(
            self.config.output_dir, "preprocessing_checkpoint.json"
        )
        # Load existing checkpoint or initialize a new one
        self.load_checkpoint()

    def load_checkpoint(self):
        """
        Load the last processed file index from the checkpoint file.

        If the checkpoint file does not exist, initialize the checkpoint with zeros for each data split.
        """
        if os.path.exists(self.checkpoint_file):
            # Load existing checkpoint
            with open(self.checkpoint_file, "r") as f:
                self.checkpoint = json.load(f)
            logger.info("Checkpoint loaded successfully.")
        else:
            # Initialize a new checkpoint
            self.checkpoint = {"train": 0, "val": 0, "test": 0}
            logger.info("No checkpoint found. Initialized new checkpoint.")

    def save_checkpoint(self, split_name: str, index: int):
        """
        Save the current processing state to the checkpoint file.

        Args:
            split_name (str): Name of the data split ('train', 'val', or 'test').
            index (int): Index of the last processed file in the split.
        """
        # Update checkpoint dictionary
        self.checkpoint[split_name] = index
        # Write updated checkpoint to file
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.checkpoint, f)
        logger.info(f"Checkpoint updated for {split_name} split at index {index}.")

    def preprocess_frame(self, frame: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Preprocess a single frame by resizing and normalizing it.

        Args:
            frame (np.ndarray): Input frame image as a NumPy array.
            target_size (tuple): Desired output size (height, width).

        Returns:
            np.ndarray: Preprocessed frame as a NumPy array.
        """
        # Convert frame to TensorFlow tensor
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        # Resize frame to target size
        frame = tf.image.resize(frame, target_size)
        # Normalize pixel values to [0, 1] range
        frame = frame / 255.0
        # Convert tensor back to NumPy array
        frame = frame.numpy()
        return frame

    def extract_faces_from_frame(self, frame: np.ndarray) -> list:
        """
        Detect and extract faces from a single frame using OpenCV with optional CUDA support.

        Args:
            frame (np.ndarray): Input frame image as a NumPy array.

        Returns:
            list: List of extracted face images as NumPy arrays.
        """
        # Load pre-trained Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Check for CUDA support in OpenCV
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Upload frame to GPU memory
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # Convert frame to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

            # Perform face detection on GPU
            faces = face_cascade.detectMultiScale(
                gpu_gray.download(),
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=tuple(self.config.min_size),
            )
        else:
            # Convert frame to grayscale on CPU
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform face detection on CPU
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=tuple(self.config.min_size),
            )

        face_crops = []
        # Extract and store each detected face region
        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            face_crops.append(face)

        return face_crops

    def select_key_frames(self, video_path: str, max_frames: int) -> np.ndarray:
        """
        Select key frames from a video using scene change detection with ffmpeg.

        Args:
            video_path (str): Path to the input video file.
            max_frames (int): Maximum number of frames to extract.

        Returns:
            np.ndarray: Array of selected key frames as NumPy arrays.
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
                .filter("select", f"gt(scene,{self.config.scene_change_threshold})")
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

    def process_video(self, video_path: str, label: str, max_frames: int) -> list:
        """
        Process a single video by extracting key frames, detecting faces, and preprocessing them.

        Args:
            video_path (str): Path to the input video file.
            label (str): Label of the video ('REAL' or 'FAKE').
            max_frames (int): Maximum number of frames to process from the video.

        Returns:
            list: List of preprocessed frames as NumPy arrays.
        """
        # Select key frames from the video
        key_frames = self.select_key_frames(video_path, max_frames)
        processed_frames = []

        for frame in key_frames:
            # Attempt to extract faces from the frame
            faces = self.extract_faces_from_frame(frame)

            if faces:
                # Preprocess the first detected face
                processed_frame = self.preprocess_frame(
                    faces[0], tuple(self.config.target_size)
                )
            else:
                # If no face detected, preprocess the entire frame
                processed_frame = self.preprocess_frame(
                    frame, tuple(self.config.target_size)
                )

            processed_frames.append(processed_frame)

        return processed_frames

    def save_dataset(self, data: list, labels: list, split_name: str):
        """
        Save the processed data and labels to HDF5 files for the specified data split.

        Args:
            data (list): List of preprocessed frames.
            labels (list): Corresponding labels for the frames.
            split_name (str): Name of the data split ('train', 'val', or 'test').
        """
        # Define directories for saving data and labels
        split_dir = os.path.join(self.config.output_dir, split_name)
        data_dir = os.path.join(split_dir, "data")
        labels_dir = os.path.join(split_dir, "labels")

        # Create directories if they do not exist
        create_directories([split_dir, data_dir, labels_dir])

        # Define file paths
        data_file = os.path.join(data_dir, "data.h5")
        labels_file = os.path.join(labels_dir, "labels.h5")

        # encode the labels
        labels = [0 if label == "REAL" else 1 for label in labels]

        # Save data and labels using utility functions
        save_h5py(np.array(data), Path(data_file), dataset_name="data")
        save_h5py(np.array(labels), Path(labels_file), dataset_name="labels")

        logger.info(f"Appended {split_name} dataset with {len(data)} frames for this video.")
        

    def process_and_save_split(
        self, split_name: str, split_files: list, metadata: dict
    ):
        """
        Process all videos in a data split and save the processed frames and labels.

        Args:
            split_name (str): Name of the data split ('train', 'val', or 'test').
            split_files (list): List of video file names in the split.
            metadata (dict): Dictionary containing metadata for all videos.
        """
        data = []
        labels = []
        real_count = 0
        fake_count = 0

        # Retrieve the starting index from the checkpoint
        start_index = self.checkpoint.get(split_name, 0)
        files_to_process = split_files[start_index:]

        # Iterate over each video file in the split
        for idx, video_file in enumerate(
            tqdm(files_to_process, desc=f"Processing {split_name} data")
        ):
            
            num_frames = 0 # number of frames for the video
            
            video_path = os.path.join(self.config.data_path, video_file)
            label = metadata[video_file]["label"]

            # Process the video to extract preprocessed frames
            frames = self.process_video(video_path, label, self.config.max_frames)
            num_frames = len(frames)
            if num_frames == 0:
                logger.info(f"Appended {split_name} dataset with 0 frames for this video.")
                continue
            # Append frames and corresponding labels
            for frame in frames:
                data.append(frame)
                labels.append(0 if label == "REAL" else 1)

            # Update counts based on label
            if label == "REAL":
                real_count += 1
            else:
                fake_count += 1

            # Save data incrementally after processing a video
            if idx > 0:
                self.save_dataset(data, labels, split_name)
                self.save_checkpoint(split_name, start_index + idx + 1)
                # reset lists
                data = []
                labels = []

        # Save any remaining data after processing all videos
        if data:
            self.save_dataset(data, labels, split_name)

        # Calculate and log the ratio of REAL to FAKE samples
        real_fake_ratio = real_count / (fake_count + 1e-6)
        logger.info(
            f"{split_name.capitalize()} Split - REAL: {real_count}, FAKE: {fake_count}, "
            f"RATIO (REAL/FAKE): {real_fake_ratio:.2f}"
        )
        print(
            f"{split_name.capitalize()} Split - REAL: {real_count}, FAKE: {fake_count}, "
            f"RATIO (REAL/FAKE): {real_fake_ratio:.2f}"
        )

    def stratified_split(
        self,
        video_files: list,
        metadata: dict,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ):
        """
        Perform stratified splitting of video files into training, validation, and test sets with an equal distribution
        of real and fake samples in each set.

        Args:
            video_files (list): List of all video file names.
            metadata (dict): Dictionary containing metadata for all videos.
            test_size (float): Proportion of data to include in the test split.
            val_size (float): Proportion of data to include in the validation split from the training data.

        Returns:
            tuple: Lists of video files for training, validation, and test splits.
        """
        # Create labels array based on metadata
        labels = np.array(
            [0 if metadata[video]["label"] == "REAL" else 1 for video in video_files]
        )

        # Separate real and fake video files
        real_files = [video_files[i] for i in range(len(video_files)) if labels[i] == 0]
        fake_files = [video_files[i] for i in range(len(video_files)) if labels[i] == 1]

        # Function to perform stratified split for each category
        def stratified_split_category(files: list, test_size: float, val_size: float):
            # Perform initial stratified split for training and testing
            stratified_split = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=42
            )
            train_indices, test_indices = next(
                stratified_split.split(files, np.zeros(len(files)))
            )

            # Extract training and testing files
            train_files = [files[i] for i in train_indices]
            test_files = [files[i] for i in test_indices]

            # Calculate validation size based on the training data
            val_size_adjusted = val_size / (1 - test_size)

            # Perform stratified split on training data to get validation set
            stratified_split_val = StratifiedShuffleSplit(
                n_splits=1, test_size=val_size_adjusted, random_state=42
            )
            train_indices_final, val_indices = next(
                stratified_split_val.split(train_files, np.zeros(len(train_files)))
            )

            # Extract final training and validation files
            val_files = [train_files[i] for i in val_indices]
            train_files_final = [train_files[i] for i in train_indices_final]

            return train_files_final, val_files, test_files

        # Split real and fake files separately
        real_train_files, real_val_files, real_test_files = stratified_split_category(
            real_files, test_size, val_size
        )
        fake_train_files, fake_val_files, fake_test_files = stratified_split_category(
            fake_files, test_size, val_size
        )

        # Combine real and fake files
        train_files = real_train_files + fake_train_files
        val_files = real_val_files + fake_val_files
        test_files = real_test_files + fake_test_files

        # Count real and fake files in each split
        def count_labels(files: list) -> dict:
            counts = {"REAL": 0, "FAKE": 0}
            for file in files:
                label = metadata[file]["label"]
                counts[label] += 1
            return counts

        train_counts = count_labels(train_files)
        val_counts = count_labels(val_files)
        test_counts = count_labels(test_files)

        # Log and print the counts
        logger.info(
            f"Training Split - REAL: {train_counts['REAL']}, FAKE: {train_counts['FAKE']}"
        )
        logger.info(
            f"Validation Split - REAL: {val_counts['REAL']}, FAKE: {val_counts['FAKE']}"
        )
        logger.info(
            f"Test Split - REAL: {test_counts['REAL']}, FAKE: {test_counts['FAKE']}"
        )

        print(
            f"Training Split - REAL: {train_counts['REAL']}, FAKE: {train_counts['FAKE']}"
        )
        print(
            f"Validation Split - REAL: {val_counts['REAL']}, FAKE: {val_counts['FAKE']}"
        )
        print(f"Test Split - REAL: {test_counts['REAL']}, FAKE: {test_counts['FAKE']}")

        return train_files, val_files, test_files

    def execute(self):
        """
        Execute the complete preprocessing pipeline:
        - Load metadata.
        - Perform stratified data splitting.
        - Process and save training, validation, and test datasets.
        """
        logger.info("Starting data preprocessing...")

        # Load metadata from JSON file
        metadata_path = os.path.join(self.config.data_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info("Metadata loaded successfully.")

        # Retrieve list of all video files
        video_files = list(metadata.keys())

        # Perform stratified split to maintain label distribution
        train_files, val_files, test_files = self.stratified_split(
            video_files, metadata
        )

        # Process and save each data split
        self.process_and_save_split("train", train_files, metadata)
        self.process_and_save_split("val", val_files, metadata)
        self.process_and_save_split("test", test_files, metadata)

        logger.info("Data preprocessing completed successfully.")
