import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import DataPreprocessingConfig
from DeepfakeDetection.utils.common import create_directories, save_h5py
from concurrent.futures import ThreadPoolExecutor, as_completed


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        create_directories([self.config.output_dir])
        self.checkpoint_file = os.path.join(
            self.config.output_dir, "preprocessing_checkpoint.json"
        )
        self.load_checkpoint()

    def load_checkpoint(self):
        """Load the last processed file index from the checkpoint."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                self.checkpoint = json.load(f)
        else:
            self.checkpoint = {"train": 0, "val": 0, "test": 0}

    def save_checkpoint(self, split_name, index):
        """Save the current processing state to the checkpoint."""
        self.checkpoint[split_name] = index
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.checkpoint, f)

    def preprocess_frame(self, frame, target_size):
        """Preprocess a single frame using TensorFlow on GPU."""
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        frame = tf.image.resize(frame, target_size)
        frame = frame / 255.0  # Normalize to [0, 1]
        frame = frame.numpy()  # Convert back to NumPy array
        return frame

    def extract_faces_from_frame(self, frame):
        """Extract faces using OpenCV with CUDA."""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Check if CUDA is available and use it if possible
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # Convert to grayscale
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

            # Perform face detection on GPU
            faces = face_cascade.detectMultiScale(
                gpu_gray.download(),
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=tuple(self.config.min_size),
            )
        else:
            # Fall back to CPU if GPU is not available
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=tuple(self.config.min_size),
            )

        face_crops = []
        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            face_crops.append(face)

        return face_crops

    def process_video(self, video_path, label, max_frames):
        """Process a video: extract frames and faces, preprocess them."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(frame_count // max_frames, 1)

        processed_frames = []

        for i in range(0, frame_count, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.extract_faces_from_frame(frame)

            if faces:
                processed_frame = self.preprocess_frame(
                    faces[0], tuple(self.config.target_size)
                )
            else:
                processed_frame = self.preprocess_frame(
                    frame, tuple(self.config.target_size)
                )

            processed_frames.append(processed_frame)

            if len(processed_frames) >= max_frames:
                break

        cap.release()
        return processed_frames

    def save_dataset(self, data, labels, split_name):
        """Save the processed data and labels to HDF5 format."""
        split_dir = os.path.join(self.config.output_dir, split_name)
        data_dir = os.path.join(split_dir, "data")
        labels_dir = os.path.join(split_dir, "labels")

        create_directories([split_dir, data_dir, labels_dir])

        data_file = os.path.join(data_dir, "data.h5")
        labels_file = os.path.join(labels_dir, "labels.h5")

        # Save data and labels using h5py utility functions
        save_h5py(np.array(data), Path(data_file), dataset_name="data")
        save_h5py(np.array(labels), Path(labels_file), dataset_name="labels")

    def process_and_save_split(self, split_name, split_files, metadata):
        """Process a single split and return processed data and labels."""
        data, labels = [], []
        real_count, fake_count = 0, 0

        start_index = self.checkpoint.get(split_name, 0)
        files_to_process = split_files[start_index:]

        for idx, video_file in enumerate(
            tqdm(files_to_process, desc=f"Processing {split_name} data")
        ):
            video_path = os.path.join(self.config.data_path, video_file)
            label = metadata[video_file]["label"]

            frames = self.process_video(video_path, label, self.config.max_frames)

            for frame in frames:
                data.append(frame)
                labels.append(0 if label == "REAL" else 1)

            if label == "REAL":
                real_count += 1
            else:
                fake_count += 1

            # Save data incrementally after processing each video
            if (idx + 1) % self.config.incremental_save_frequency == 0:
                self.save_dataset(data, labels, split_name)
                data, labels = [], []  # Clear lists to free memory

            # Save checkpoint after processing each video
            self.save_checkpoint(split_name, start_index + idx + 1)

        # Save any remaining data
        if data:
            self.save_dataset(data, labels, split_name)

        # Print the counts of REAL and FAKE videos for the split
        print(
            f"{split_name.capitalize()} Split - REAL: {real_count}, FAKE: {fake_count}"
        )

        return data, labels

    def stratified_split(self, video_files, metadata, test_size=0.2, val_size=0.1):
        """Perform stratified splitting to maintain the balance between REAL and FAKE."""
        labels = np.array(
            [0 if metadata[video]["label"] == "REAL" else 1 for video in video_files]
        )
        stratified_split = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=42
        )
        train_indices, test_indices = next(stratified_split.split(video_files, labels))

        train_files = [video_files[i] for i in train_indices]
        test_files = [video_files[i] for i in test_indices]

        labels_train = np.array([labels[i] for i in train_indices])
        stratified_split_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=42
        )
        train_indices, val_indices = next(
            stratified_split_val.split(train_files, labels_train)
        )

        val_files = [train_files[i] for i in val_indices]
        train_files = [train_files[i] for i in train_indices]

        return train_files, val_files, test_files

    def execute(self):
        """Execute the preprocessing pipeline: process videos and save datasets."""
        logger.info("Starting data preprocessing...")

        # Load metadata from the JSON file
        with open(os.path.join(self.config.data_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # Stratified splitting to ensure balanced distribution of REAL and FAKE videos
        video_files = list(metadata.keys())
        train_files, val_files, test_files = self.stratified_split(
            video_files, metadata
        )

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        # Process videos sequentially and save using multithreading
        data_splits = {}
        for split_name, split_files in splits.items():
            data_splits[split_name] = self.process_and_save_split(
                split_name, split_files, metadata
            )

        logger.info("Data preprocessing completed successfully.")
