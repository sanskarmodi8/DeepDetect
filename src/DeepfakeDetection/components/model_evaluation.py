import os
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import ModelEvaluationConfig
from DeepfakeDetection.utils.common import save_json


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, sequence_length=60, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = []
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > self.sequence_length:
            start = np.random.randint(0, frame_count - self.sequence_length)
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
        pass


class ResNextLSTMEvaluationStrategy(EvaluationStrategy):
    def evaluate(self, model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)

                _, outputs = model(inputs)  # Unpack the tuple, use only the second element
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)

                all_preds.extend(outputs.cpu().numpy()[:, 1])  # Probability of being fake
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        precision = precision_score(all_labels, (np.array(all_preds) > 0.5).astype(int), average="weighted")
        recall = recall_score(all_labels, (np.array(all_preds) > 0.5).astype(int), average="weighted")
        f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int), average="weighted")
        auc = roc_auc_score(all_labels, all_preds)

        return {
            "loss": epoch_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_strategy = (
            ResNextLSTMEvaluationStrategy()
        )  # Updated to reflect the new model strategy
        self._initialize_mlflow()

    def _initialize_mlflow(self):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        if not mlflow.active_run():
            mlflow.start_run()
        logger.info("MLflow experiment initialized for evaluation.")

    def load_model(self):
        # Assuming the ResNextLSTM model has been saved as a PyTorch .pt or .pth file
        model = torch.load(self.config.model_path, map_location=self.device)
        model.eval()
        return model

    def load_video_paths(self, data_path):
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
        model = self.load_model()
        criterion = nn.CrossEntropyLoss()

        metrics = self.evaluation_strategy.evaluate(
            model, self.test_loader, criterion, self.device
        )

        logger.info(f"Evaluation metrics: {metrics}")

        save_json(Path(self.config.score), metrics)
        logger.info(f"Evaluation metrics saved to {self.config.score}")

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_artifact(str(Path(self.config.score)))

        return metrics

    def execute(self):
        logger.info("Starting model evaluation...")
        try:
            self.prepare_data()
            metrics = self.evaluate_model()
            logger.info(f"Model evaluation completed with metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise
        finally:
            mlflow.end_run()
