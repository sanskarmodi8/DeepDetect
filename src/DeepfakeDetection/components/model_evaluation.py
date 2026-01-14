"""Model Evaluation component."""

import os
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import mlflow
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import torch
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import ModelEvaluationConfig
from DeepfakeDetection.utils.common import save_json

# load environment variables
load_dotenv()


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, sequence_length=60, transform=None):
        """
        Initialize the VideoDataset class.

        Args:
            video_paths (list): List of paths to the video files.
            labels (list): List of labels for the video files.
            sequence_length (int, optional): The length of the sequence of frames to extract from the video. Defaults to 60.
            transform (callable, optional): A function to apply to the frames before they are returned. Defaults to None.
        """
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the frames of the video and the label.
        """
        rng = np.random.default_rng(seed=42)
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = []
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            logger.warning(f"Could not open video file: {video_path}")
            # Create a default frame with zeros - use numpy array to match cv2 output format
            dummy_frame = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )  # Using numpy array format
            if self.transform:
                dummy_frame = self.transform(dummy_frame)
            frames = [dummy_frame] * self.sequence_length
            cap.release()
            return torch.stack(frames), torch.tensor(label, dtype=torch.long)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > self.sequence_length:
            start = rng.integers(0, frame_count - self.sequence_length)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        # If we don't have enough frames, pad with zeros
        if len(frames) == 0:
            # No frames were read, create a default frame as numpy array
            dummy_frame = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )  # CV2 returns numpy arrays
            if self.transform:
                dummy_frame = self.transform(dummy_frame)
            frames = [dummy_frame] * self.sequence_length
        elif len(frames) < self.sequence_length:
            # Pad with the last frame if we have at least one frame
            last_frame = frames[-1]
            frames.extend([last_frame] * (self.sequence_length - len(frames)))

        return torch.stack(frames), torch.tensor(label, dtype=torch.long)


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, model, dataloader, criterion, device):
        """
        Evaluates the model on the given dataloader and returns a dictionary containing the evaluation metrics.

        Args:
            model (nn.Module): The model to be evaluated.
            dataloader (DataLoader): The dataloader containing the evaluation data.
            criterion (nn.Module): The loss function to be used.
            device (torch.device): The device to be used for evaluation.

        Returns:
            tuple: A tuple containing a dictionary of evaluation metrics and a dictionary of plots.
        """
        pass

    @abstractmethod
    def create_plots(self, all_preds, all_labels, pred_classes):
        """
        Creates plots for the given predictions and labels.

        Args:
            all_preds (list): List of predicted probabilities.
            all_labels (list): List of true labels.
            pred_classes (list): List of predicted classes.

        Returns:
            dict: A dictionary containing the evaluation plots.
        """
        pass


class ResNextLSTMEvaluationStrategy(EvaluationStrategy):
    def evaluate(self, model, dataloader, criterion, device):
        """
        Evaluates the model on the given dataloader and returns a dictionary containing the evaluation metrics.

        Args:
            model (nn.Module): The model to be evaluated.
            dataloader (DataLoader): The dataloader containing the evaluation data.
            criterion (nn.Module): The loss function to be used.
            device (torch.device): The device to be used for evaluation.

        Returns:
            tuple: A tuple containing a dictionary of evaluation metrics and a dictionary of plots.
        """
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        pred_classes = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)

                _, _, outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                # Store full output probabilities
                all_preds.append(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Get predicted class for confusion matrix
                _, predicted = torch.max(outputs, 1)
                pred_classes.extend(predicted.cpu().numpy())

        # Concatenate all predictions if there are multiple batches
        all_preds = np.vstack(all_preds) if len(all_preds) > 0 else np.array([])

        epoch_loss = running_loss / len(dataloader)
        metrics = self.calculate_metrics(all_labels, all_preds, pred_classes)
        metrics["loss"] = epoch_loss

        plots = self.create_plots(all_preds, all_labels, pred_classes)

        return metrics, plots

    def calculate_metrics(self, all_labels, all_preds, pred_classes):
        """
        Calculates evaluation metrics from the given predictions and labels.

        Args:
            all_labels (list): List of true labels.
            all_preds (ndarray): Array of prediction probabilities for each class.
            pred_classes (list): List of predicted class indices.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        # Convert to numpy arrays if not already
        all_labels = np.array(all_labels)
        pred_classes = np.array(pred_classes)

        accuracy = accuracy_score(all_labels, pred_classes)

        try:
            precision = precision_score(
                all_labels, pred_classes, average="weighted", zero_division=0
            )
            recall = recall_score(
                all_labels, pred_classes, average="weighted", zero_division=0
            )
            f1 = f1_score(all_labels, pred_classes, average="weighted", zero_division=0)
        except Exception as e:
            logger.warning(f"Error calculating precision/recall/f1: {str(e)}")
            precision = recall = f1 = 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def create_plots(self, all_preds, all_labels, pred_classes):
        """
        Creates plots for the given predictions and labels.

        Args:
            all_preds (ndarray): Array of prediction probabilities for each class.
            all_labels (list): List of true labels.
            pred_classes (list): List of predicted class indices.

        Returns:
            dict: A dictionary containing the evaluation plots.
        """
        # Convert to numpy arrays for easier handling
        all_labels = np.array(all_labels)
        pred_classes = np.array(pred_classes)

        # Create confusion matrix
        cm = confusion_matrix(all_labels, pred_classes)
        class_names = ["Real", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"]
        cm_plot = go.Figure(
            data=go.Heatmap(z=cm, x=class_names, y=class_names, colorscale="Viridis")
        )
        cm_plot.update_layout(
            title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual"
        )

        # Create plots for each class
        plots = {"confusion_matrix": cm_plot}

        # Create prediction distribution plots
        try:
            # For each class, create a histogram of predicted probabilities
            dist_fig = go.Figure()
            for i, class_name in enumerate(class_names):
                # Get predictions for examples that truly belong to this class
                class_mask = all_labels == i
                if np.any(class_mask):  # Only add trace if we have examples
                    dist_fig.add_trace(
                        go.Histogram(
                            x=all_preds[class_mask, i],
                            name=f"True {class_name}",
                            opacity=0.7,
                            histnorm="probability",
                        )
                    )

            dist_fig.update_layout(
                title="Prediction Score Distribution by Class",
                xaxis_title="Prediction Score",
                yaxis_title="Probability",
                barmode="overlay",
            )
            plots["prediction_distribution"] = dist_fig

        except Exception as e:
            logger.warning(f"Error creating prediction distribution plots: {str(e)}")

        return plots


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes ModelEvaluation with a configuration object.

        Args:
            config (ModelEvaluationConfig): Configuration object with evaluation settings.

        Attributes:
            config (ModelEvaluationConfig): Configuration object with evaluation settings.
            device (torch.device): Device to be used for evaluation.
            evaluation_strategy (EvaluationStrategy): Strategy for evaluating the model.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_strategy = ResNextLSTMEvaluationStrategy()

    def initialize_mlflow(self):
        """
        Initializes the MLflow experiment for evaluation.

        This function sets the MLflow tracking URI to the environment variable
        MLFLOW_TRACKING_URI and starts a new run if there is no active run.
        """
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        if not mlflow.active_run():
            mlflow.start_run()
        logger.info("MLflow experiment initialized for evaluation.")

    def load_model(self):
        """
        Loads the model from the given model path with safeguards for PyTorch 2.6+.

        This method attempts to load the model first with weights_only=False.
        If that fails, it tries using safe_globals to add the model class.

        Returns:
            nn.Module: The loaded model.
        """
        try:
            # Try loading with weights_only=False first (less secure but backward compatible)
            model = torch.load(
                self.config.model_path, map_location=self.device, weights_only=False
            )
            logger.info("Model loaded successfully with weights_only=False.")
        except Exception as e:
            logger.warning(f"Failed to load model with weights_only=False: {str(e)}")

            try:
                # Try using safe_globals for a more secure approach
                # First import the model class
                from DeepfakeDetection.components.model_training import ResNextLSTMModel

                # Add it to safe globals
                torch.serialization.add_safe_globals([ResNextLSTMModel])

                # Now try loading with weights_only=True (more secure)
                model = torch.load(
                    self.config.model_path, map_location=self.device, weights_only=True
                )
                logger.info("Model loaded successfully with safe_globals.")
            except Exception as nested_e:
                logger.error(f"Failed to load model with safe_globals: {str(nested_e)}")
                # If all attempts fail, raise a comprehensive error
                raise RuntimeError(
                    f"Could not load model from {self.config.model_path}. Original error: {str(e)}"
                )

        model.eval()
        return model

    def load_video_paths(self, data_path):
        """
        Loads video paths and labels from the given data path.

        Args:
            data_path (str): Path to the dataset.

        Returns:
            tuple: A tuple of two lists. The first list contains the video paths,
            and the second list contains the corresponding labels.
        """
        video_paths = []
        labels = []

        # Dictionary mapping folder names to class indices
        class_folders = {
            "original": 0,  # Real videos
            "Face2Face": 1,  # Face2Face deepfake
            "FaceSwap": 2,  # FaceSwap deepfake
            "FaceShifter": 3,  # FaceShifter deepfake
            "NeuralTextures": 4,  # NeuralTextures deepfake
        }

        for folder, label in class_folders.items():
            folder_path = os.path.join(data_path, folder)

            # Check if the folder exists
            if not os.path.exists(folder_path):
                logger.warning(f"Folder not found: {folder_path}")
                continue

            try:
                for video in os.listdir(folder_path):
                    if video.endswith(".mp4"):
                        video_path = os.path.join(folder_path, video)
                        # Verify the file exists and is accessible
                        if os.path.isfile(video_path) and os.access(
                            video_path, os.R_OK
                        ):
                            video_paths.append(video_path)
                            labels.append(label)
                        else:
                            logger.warning(f"Video file not accessible: {video_path}")
            except Exception as e:
                logger.error(f"Error loading videos from {folder_path}: {str(e)}")

        logger.info(
            f"Loaded {len(video_paths)} videos across {len(class_folders)} classes"
        )

        # Check if any videos were found
        if len(video_paths) == 0:
            raise ValueError(f"No video files found in {data_path}")

        return video_paths, labels

    def prepare_data(self):
        """
        Prepare the data for evaluation by creating a DataLoader.

        This function loads the video paths and labels for the test split,
        and creates a VideoDataset object. The VideoDataset object is used
        to create a DataLoader for testing.

        Attributes:
            test_loader (DataLoader): DataLoader for testing.
        """
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(tuple(self.config.input_shape[:2])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_videos, test_labels = self.load_video_paths(self.config.data_path)

        # Safety check to avoid empty dataset
        if len(test_videos) == 0 or len(test_labels) == 0:
            raise ValueError("No test videos or labels were loaded")

        test_dataset = VideoDataset(
            test_videos,
            test_labels,
            sequence_length=self.config.sequence_length,
            transform=transform,
        )

        # Use a more conservative number of workers if needed
        workers = min(self.config.num_workers, 4, os.cpu_count() or 1)

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=workers,
        )

    def evaluate_model(self):
        """
        Evaluate the model on the test split.

        The model is loaded from the given model path, and the evaluation strategy
        is used to evaluate the model on the test loader. The evaluation metrics
        are logged to the console, and the model is saved to the given score path.
        The plots are saved to the given plots path.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        try:
            model = self.load_model()
            criterion = nn.CrossEntropyLoss()

            metrics, plots = self.evaluation_strategy.evaluate(
                model, self.test_loader, criterion, self.device
            )

            logger.info(f"Evaluation metrics: {metrics}")

            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config.score), exist_ok=True)

            save_json(Path(self.config.score), metrics)
            logger.info(f"Evaluation metrics saved to {self.config.score}")

            self.save_plots(plots)

            # Log metrics to mlflow
            if mlflow.active_run():
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                mlflow.log_artifact(str(Path(self.config.score)))

            return metrics

        except Exception as e:
            logger.error(f"Error in evaluate_model: {str(e)}")
            raise

    def save_plots(self, plots):
        """
        Saves the evaluation plots to the directory specified in the configuration.

        Args:
            plots (dict): A dictionary with plot names as keys and plotly figures as values.
        """
        plots_dir = os.path.join(self.config.root_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for plot_name, plot_figure in plots.items():
            try:
                plot_path = os.path.join(plots_dir, f"{plot_name}.html")
                pio.write_html(plot_figure, file=plot_path)
                logger.info(f"Plot saved: {plot_path}")

                if mlflow.active_run():
                    mlflow.log_artifact(plot_path)
            except Exception as e:
                logger.warning(f"Failed to save plot {plot_name}: {str(e)}")

    def execute(self):
        """
        Executes the model evaluation pipeline.

        This function initializes MLflow, prepares the data for evaluation,
        evaluates the model using the evaluation strategy, and logs the metrics
        to MLflow. If an exception occurs during evaluation, it is logged and
        re-raised. Finally, if an MLflow run is active, it is ended.
        """
        logger.info("Starting model evaluation...")
        try:
            self.initialize_mlflow()
            self.prepare_data()
            metrics = self.evaluate_model()
            logger.info(f"Model evaluation completed with metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise
        finally:
            if mlflow.active_run():
                mlflow.end_run()
                logger.info("MLflow run ended after evaluation.")
