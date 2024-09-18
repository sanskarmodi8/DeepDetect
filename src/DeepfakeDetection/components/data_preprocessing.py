import os
from abc import ABC, abstractmethod

import cv2
import face_recognition
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import DataPreprocessingConfig
from DeepfakeDetection.utils.common import create_directories

class FrameExtractionStrategy(ABC):
    @abstractmethod
    def extract_frames(self, video_path):
        """
        Abstract method to extract frames from a given video path.

        Args:
            video_path (str): Path to the video file from which frames will be extracted.

        Returns:
            np.ndarray: Array of frames with shape (num_frames, height, width, channels).
        """
        pass


class FaceDetectionStrategy(ABC):
    @abstractmethod
    def detect_faces(self, frames):
        """
        Abstract method to detect faces from a given array of frames.

        Args:
            frames (np.ndarray): Array of frames with shape (num_frames, height, width, channels).

        Returns:
            list: List of detected face locations with shape (num_faces, 4) and dtype int32.
        """
        pass


class OpenCVFrameExtraction(FrameExtractionStrategy):
    def extract_frames(self, video_path):
        """
        Extract frames from a video file at the given path.

        Args:
            video_path (str): Path to the video file.

        Yields:
            A numpy array, where each array is a frame from the video.
        """
        vidobj = cv2.VideoCapture(video_path)
        success, image = vidobj.read()
        while success:
            yield image
            success, image = vidobj.read()


class FaceRecognitionStrategy(FaceDetectionStrategy):
    def detect_faces(self, frames):
        """
        Detect faces in a list of frames.

        Args:
            frames (list): A list of numpy arrays, where each array is a frame from a video.

        Returns:
            A list of lists of face bounding boxes, where each inner list is a list of bounding boxes for a frame.
        """
        return face_recognition.batch_face_locations(frames)


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """
        Initializes DataPreprocessing with a configuration object.

        Args:
            config (DataPreprocessingConfig): Configuration object with frame extraction and face detection parameters.
        """
        self.config = config
        self.frame_extraction_strategy = OpenCVFrameExtraction()
        self.face_detection_strategy = FaceRecognitionStrategy()

    def write_video(self, output_path, frames):
        """
        Writes a video from a list of frames to a file.

        Args:
            output_path (str): Path to the output video file.
            frames (list): A list of numpy arrays, where each array is a frame from a video.
        """
        # Use 'mp4v' for MP4 files, which is widely supported
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.config.fps, tuple(self.config.resolution)
        )
        for frame in frames:
            out.write(frame)
        out.release()

    def process_video(self, video_file, output_dir):
        """
        Process a video file, detect faces, and save the resulting video.

        Args:
            video_file (tuple): Tuple containing the folder name and the video file name.
            output_dir (str): Directory where the output video will be saved.
        """
        folder, video_file = video_file
        out_path = os.path.join(output_dir, video_file)
        video_path = os.path.join(self.config.data_path, folder, video_file)

        # Check if the file already exists to avoid duplication
        if os.path.exists(out_path):
            print(f"File already exists: {out_path}")
            return

        # Extract frames
        frames = []
        processed_frames = []
        for idx, frame in enumerate(
            self.frame_extraction_strategy.extract_frames(video_path)
        ):
            if idx <= self.config.max_frames:  # Limiting the number of frames
                frames.append(frame)
                # Process batches of 4 frames
                if len(frames) == 4:
                    faces = self.face_detection_strategy.detect_faces(frames)
                    for i, face in enumerate(faces):
                        if len(face) > 0:
                            top, right, bottom, left = face[0]
                            cropped_face = cv2.resize(
                                frames[i][top:bottom, left:right, :],
                                tuple(self.config.resolution),
                            )
                            processed_frames.append(cropped_face)
                    frames = []  # Reset frames list after processing

        # Save the processed video
        self.write_video(out_path, processed_frames)

    def process_and_save_split(self, split_name, video_files):
        """
        Process and save videos in a specific split (train, val, or test),
        and organize them into 'original' and 'fake' subdirectories.

        Args:
            split_name (str): Name of the data split ('train', 'val', 'test').
            video_files (list): List of video file paths to process.
        """
        # Create directories for 'original' and 'fake' videos within each split directory
        output_original_dir = os.path.join(self.config.root_dir, split_name, "original")
        output_fake_dir = os.path.join(self.config.root_dir, split_name, "fake")
        create_directories([output_original_dir, output_fake_dir])

        for video_file in tqdm(video_files, desc=f"Processing {split_name} split"):
            folder, file_name = video_file

            # Check whether the video is 'original' or 'fake' based on the folder name
            if folder == "original":
                output_dir = output_original_dir
            elif folder == "manipulated":
                output_dir = output_fake_dir
            else:
                raise ValueError(f"Unexpected folder name: {folder}")

            self.process_video(video_file, output_dir)

    def stratified_split(
        self, video_files: list, test_size: float = 0.15, val_size: float = 0.15
    ):
        labels = np.array([file[0] for file in video_files])
        stratified_split = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=42
        )

        train_indices, test_indices = next(stratified_split.split(video_files, labels))
        train_files = [video_files[i] for i in train_indices]
        test_files = [video_files[i] for i in test_indices]

        val_size_adjusted = val_size / (1 - test_size)
        labels_train = labels[train_indices]
        stratified_split_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size_adjusted, random_state=42
        )

        train_indices_final, val_indices = next(
            stratified_split_val.split(train_files, labels_train)
        )

        val_files = [train_files[i] for i in val_indices]
        train_files_final = [train_files[i] for i in train_indices_final]

        return train_files_final, val_files, test_files

    def run(self):
        # Gather files from manipulated and original video directories
        files = [
            (folder, file)
            for folder in ["manipulated", "original"]
            for file in sorted(os.listdir(os.path.join(self.config.data_path, folder)))
        ]

        # Perform stratified split into train, validation, and test sets
        train_files, val_files, test_files = self.stratified_split(files)

        logger.info("Processing train split...")
        self.process_and_save_split("train", train_files)

        logger.info("Processing validation split...")
        self.process_and_save_split("val", val_files)

        logger.info("Processing test split...")
        self.process_and_save_split("test", test_files)
