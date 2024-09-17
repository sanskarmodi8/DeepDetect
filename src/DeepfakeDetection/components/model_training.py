import os
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import ModelTrainingConfig


# Define your VideoDataset class for loading video data
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

        # If we don't have enough frames, pad with zeros
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros_like(frames[0]))

        return torch.stack(frames), torch.tensor(label, dtype=torch.long)


# Abstract model strategy class
class ModelStrategy(ABC):
    @abstractmethod
    def build_model(self, config, num_classes=2):
        pass


# Abstract training strategy class
class TrainingStrategy(ABC):
    @abstractmethod
    def train_epoch(self, model, dataloader, criterion, optimizer, device):
        pass

    @abstractmethod
    def validate(self, model, dataloader, criterion, device):
        pass


# ResNext-LSTM Strategy using your custom model
class ResNextLSTMStrategy(ModelStrategy):
    def build_model(self, config, num_classes=2):
        class ResNextLSTMModel(nn.Module):
            def __init__(
                self,
                num_classes,
                latent_dim=2048,
                lstm_layers=1,
                hidden_dim=2048,
                bidirectional=False,
            ):
                super(ResNextLSTMModel, self).__init__()
                model = models.resnext50_32x4d(pretrained=True)  # Residual Network CNN
                self.model = nn.Sequential(*list(model.children())[:-2])
                self.lstm = nn.LSTM(
                    latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional
                )
                self.relu = nn.LeakyReLU()
                self.dp = nn.Dropout(0.4)
                self.linear1 = nn.Linear(2048, num_classes)
                self.avgpool = nn.AdaptiveAvgPool2d(1)

            def forward(self, x):
                batch_size, seq_length, c, h, w = x.shape
                x = x.view(batch_size * seq_length, c, h, w)
                fmap = self.model(x)
                x = self.avgpool(fmap)
                x = x.view(batch_size, seq_length, 2048)
                x_lstm, _ = self.lstm(x, None)
                return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

        return ResNextLSTMModel(num_classes=num_classes)


# Standard training strategy for training and validation
class StandardTrainingStrategy(TrainingStrategy):
    def train_epoch(self, model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(dataloader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            _, outputs = model(inputs)  # Using only the classifier output
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)

                _, outputs = model(inputs)  # Using only the classifier output
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return epoch_loss, epoch_acc, precision, recall, f1


# ModelTraining class that coordinates the process
class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.model_strategy = ResNextLSTMStrategy()
        self.training_strategy = StandardTrainingStrategy()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_video_paths(self, data_path, split):
        video_paths = []
        labels = []
        split_path = os.path.join(data_path, split)

        # Load original (real) videos
        original_path = os.path.join(split_path, "original")
        for video in os.listdir(original_path):
            if video.endswith(".mp4"):
                video_paths.append(os.path.join(original_path, video))
                labels.append(0)  # 0 for real

        # Load fake videos
        fake_path = os.path.join(split_path, "fake")
        for video in os.listdir(fake_path):
            if video.endswith(".mp4"):
                video_paths.append(os.path.join(fake_path, video))
                labels.append(1)  # 1 for fake

        return video_paths, labels

    def prepare_data(self):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_videos, train_labels = self.load_video_paths(
            self.config.data_path, "train"
        )
        val_videos, val_labels = self.load_video_paths(self.config.data_path, "val")

        train_dataset = VideoDataset(
            train_videos,
            train_labels,
            sequence_length=self.config.sequence_length,
            transform=transform,
        )
        val_dataset = VideoDataset(
            val_videos,
            val_labels,
            sequence_length=self.config.sequence_length,
            transform=transform,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def train(self):
        model = self.model_strategy.build_model(self.config).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )

        best_val_acc = 0.0
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.training_strategy.train_epoch(
                model, self.train_loader, criterion, optimizer, self.device
            )
            val_loss, val_acc, precision, recall, f1 = self.training_strategy.validate(
                model, self.val_loader, criterion, self.device
            )

            scheduler.step(val_loss)

            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.config.model_path)
                logger.info("Model saved!")

        logger.info("Training completed!")

    def execute(self):
        self.prepare_data()
        self.train()
