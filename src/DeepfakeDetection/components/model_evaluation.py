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
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
            tuple: A tuple containing the frames of the video and the label. The frames are a tensor of shape (sequence_length, height, width, channels) and the label is a tensor of shape (1,).
        """
        rng = np.random.default_rng(seed=42)

        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = []
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > self.sequence_length:
            start = rng.randint(0, frame_count - self.sequence_length)
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

        while len(frames) < self.sequence_length:
            frames.append(torch.zeros_like(frames[0]))

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
            dict: A dictionary containing the evaluation metrics.
        """
        pass

    @abstractmethod
    def create_plots(self, all_preds, all_labels):
        """
        Creates plots for the given predictions and labels.

        Args:
            all_preds (list): List of predicted probabilities.
            all_labels (list): List of true labels.

        Returns:
            None
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
            dict: A dictionary containing the evaluation metrics.
        """
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)

                _, outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy()[:, 1])
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        metrics = self.calculate_metrics(all_labels, all_preds)
        metrics["loss"] = epoch_loss

        plots = self.create_plots(all_preds, all_labels)

        return metrics, plots

    def calculate_metrics(self, all_labels, all_preds):
        """
        Calculates evaluation metrics from the given predictions and labels.

        Args:
            all_labels (list): List of true labels.
            all_preds (list): List of predicted probabilities.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        precision = precision_score(
            all_labels, (np.array(all_preds) > 0.5).astype(int), average="weighted"
        )
        recall = recall_score(
            all_labels, (np.array(all_preds) > 0.5).astype(int), average="weighted"
        )
        f1 = f1_score(
            all_labels, (np.array(all_preds) > 0.5).astype(int), average="weighted"
        )
        auc = roc_auc_score(all_labels, all_preds)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

    def create_plots(self, all_preds, all_labels):
        # Confusion Matrix
        """
        Creates plots for the given predictions and labels.

        Args:
            all_preds (list): List of predicted probabilities.
            all_labels (list): List of true labels.

        Returns:
            dict: A dictionary containing the evaluation plots.
        """
        cm = confusion_matrix(all_labels, (np.array(all_preds) > 0.5).astype(int))
        cm_plot = go.Figure(data=go.Heatmap(z=cm, zmin=0, zmax=cm.max()))
        cm_plot.update_layout(
            title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual"
        )

        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_plot = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode="lines"))
        roc_plot.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
        roc_plot.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )

        # Prediction Distribution
        pred_dist_plot = go.Figure(
            data=[
                go.Histogram(
                    x=np.array(all_preds)[np.array(all_labels) == 0], name="Real"
                ),
                go.Histogram(
                    x=np.array(all_preds)[np.array(all_labels) == 1], name="Fake"
                ),
            ]
        )
        pred_dist_plot.update_layout(
            title="Prediction Distribution",
            xaxis_title="Prediction Score",
            yaxis_title="Count",
        )

        return {
            "confusion_matrix": cm_plot,
            "roc_curve": roc_plot,
            "prediction_distribution": pred_dist_plot,
        }


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
        Loads the model from the given model path.

        Args:
            None

        Returns:
            nn.Module: The loaded model.
        """
        model = torch.load(self.config.model_path, map_location=self.device)
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

        original_path = os.path.join(data_path, "original")
        for video in os.listdir(original_path):
            if video.endswith(".mp4"):
                video_paths.append(os.path.join(original_path, video))
                labels.append(0)  # 0 for real

        fake_path = os.path.join(data_path, "fake")
        for video in os.listdir(fake_path):
            if video.endswith(".mp4"):
                video_paths.append(os.path.join(fake_path, video))
                labels.append(1)  # 1 for fake

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
        test_dataset = VideoDataset(
            test_videos,
            test_labels,
            sequence_length=self.config.sequence_length,
            transform=transform,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def evaluate_model(self):
        """
        Evaluate the model on the test split.

        The model is loaded from the given model path, and the evaluation strategy
        is used to evaluate the model on the test loader. The evaluation metrics
        are logged to the console, and the model is saved to the given score path.
        The plots are saved to the given plots path.

        Attributes:
            test_loader (DataLoader): DataLoader for testing.
            evaluation_strategy (EvaluationStrategy): Strategy for evaluating the model.
            device (torch.device): Device to be used for evaluation.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        model = self.load_model()
        criterion = nn.CrossEntropyLoss()

        metrics, plots = self.evaluation_strategy.evaluate(
            model, self.test_loader, criterion, self.device
        )

        logger.info(f"Evaluation metrics: {metrics}")

        save_json(Path(self.config.score), metrics)
        logger.info(f"Evaluation metrics saved to {self.config.score}")

        self.save_plots(plots)

        # log metrics to mlflow
        if mlflow.active_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.log_artifact(str(Path(self.config.score)))

        return metrics

    def save_plots(self, plots):
        """
        Saves the evaluation plots to the directory specified in the configuration.

        Args:
            plots (dict): A dictionary with plot names as keys and plotly figures as values.
        """
        plots_dir = os.path.join(self.config.root_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for plot_name, plot_figure in plots.items():
            plot_path = os.path.join(plots_dir, f"{plot_name}.html")
            pio.write_html(plot_figure, file=plot_path)
            logger.info(f"Plot saved: {plot_path}")
            if mlflow.active_run():
                mlflow.log_artifact(plot_path)  # log the plot to mlflow

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
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise
        finally:
            if mlflow.active_run():
                mlflow.end_run()
                logger.info("MLflow run ended after evaluation.")
