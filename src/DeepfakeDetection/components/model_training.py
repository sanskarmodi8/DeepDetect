import os
from abc import ABC, abstractmethod

import cv2
import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import ModelTrainingConfig

# set CUDA memory allocation configuration to expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

        # If we don't have enough frames, pad with zeros
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros_like(frames[0]))

        return torch.stack(frames), torch.tensor(label, dtype=torch.long)


class ResNextLSTMModel(nn.Module):
    def __init__(
        self,
        num_classes,
        latent_dim,
        lstm_layers=2,
        hidden_dim=None,
        bidirectional=True,
        dropout_rate=0.5,
    ):
        """
        Args:
            num_classes (int): The number of classes in the classification problem.
            latent_dim (int): The dimensionality of the latent space.
            lstm_layers (int, optional): The number of LSTM layers. Defaults to 2.
            hidden_dim (int, optional): The number of hidden units in the LSTM layers. Defaults to None.
            bidirectional (bool, optional): Whether to use a bidirectional LSTM. Defaults to True.
            dropout_rate (float, optional): The dropout rate for the LSTM layers. Defaults to 0.5.
        """
        super(ResNextLSTMModel, self).__init__()
        model = models.resnext101_32x8d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
        )
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(dropout_rate)
        if bidirectional:
            self.linear1 = nn.Linear(hidden_dim * 2, num_classes)
        else:
            self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Forward pass of the model. Applies the CNN to the input, followed by the LSTM, then applies the attention mechanism.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, seq_length, c, h, w).

        Returns:
            tuple: A tuple containing the feature maps after the CNN, and the final output of the model.
        """
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        x_lstm, _ = self.lstm(x, None)

        # Apply attention
        attn_weights = self.attention(x_lstm).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)
        x_attn = torch.bmm(attn_weights.unsqueeze(1), x_lstm).squeeze(1)

        return fmap, self.dp(self.linear1(x_attn))


class ModelStrategy(ABC):
    @abstractmethod
    def build_model(self, config, num_classes=2):
        """
        Builds a model based on the given configuration.

        Args:
            config (ModelTrainingConfig): The configuration object.
            num_classes (int, optional): The number of classes. Defaults to 2.

        Returns:
            nn.Module: The built model.
        """
        pass


class TrainingStrategy(ABC):
    @abstractmethod
    def train_epoch(self, model, dataloader, criterion, optimizer, device):
        """
        Train the model for one epoch.

        Args:
            model (nn.Module): The model to be trained.
            dataloader (DataLoader): The data loader for the training data.
            criterion (nn.Module): The loss function to be used.
            optimizer (nn.Module): The optimizer to be used.
            device (str): The device to be used.

        Returns:
            tuple: A tuple containing the epoch loss and epoch accuracy.
        """
        pass

    @abstractmethod
    def validate(self, model, dataloader, criterion, device):
        """
        Validates the model on the given dataloader and returns the validation loss, accuracy, precision, recall and f1 score.

        Args:
            model (nn.Module): The model to be validated.
            dataloader (DataLoader): The dataloader containing the validation data.
            criterion (nn.Module): The loss function to be used.
            device (str): The device to be used for validation.

        Returns:
            tuple: A tuple containing the validation loss, accuracy, precision, recall and f1 score.
        """
        pass


class ResNextLSTMStrategy(ModelStrategy):
    def build_model(self, config, num_classes=2):
        """
        Builds a ResNext-LSTM model based on the given configuration.

        Args:
            config (ModelTrainingConfig): The configuration object.
            num_classes (int, optional): The number of classes. Defaults to 2.

        Returns:
            nn.Module: The built ResNext-LSTM model.
        """
        return ResNextLSTMModel(
            num_classes=num_classes,
            latent_dim=config.units,
            lstm_layers=config.lstm_layers,
            hidden_dim=config.units,
            bidirectional=config.bidirectional,
            dropout_rate=config.dropout_rate,
        )


class StandardTrainingStrategy(TrainingStrategy):
    def train_epoch(self, model, dataloader, criterion, optimizer, device, scaler):
        """
        Train the model for one epoch.

        Args:
            model (nn.Module): The model to be trained.
            dataloader (DataLoader): The data loader for the training data.
            criterion (nn.Module): The loss function to be used.
            optimizer (nn.Module): The optimizer to be used.
            device (str): The device to be used.
            scaler (GradScaler): The gradient scaler to be used.

        Returns:
            tuple: A tuple containing the epoch loss and epoch accuracy.
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training")):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():
                _, outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            if (i + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, model, dataloader, criterion, device):
        """
        Validates the model on the given dataloader and returns the validation loss, accuracy, precision, recall and f1 score.

        Args:
            model (nn.Module): The model to be validated.
            dataloader (DataLoader): The dataloader containing the validation data.
            criterion (nn.Module): The loss function to be used.
            device (str): The device to be used for validation.

        Returns:
            tuple: A tuple containing the validation loss, accuracy, precision, recall and f1 score.
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)

                _, outputs = model(inputs)
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


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        """
        Initializes ModelTraining with a configuration object.

        Args:
            config (ModelTrainingConfig): Configuration object with model training settings.

        Attributes:
            model_strategy (ModelStrategy): Strategy for building the model.
            training_strategy (TrainingStrategy): Strategy for training the model.
            config (ModelTrainingConfig): Configuration object with model training settings.
            device (torch.device): Device to use for training.
        """
        self.model_strategy = ResNextLSTMStrategy()
        self.training_strategy = StandardTrainingStrategy()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_mlflow(self):
        """
        Initializes the MLflow experiment for model training.

        This function sets the MLflow tracking URI to the environment variable
        MLFLOW_TRACKING_URI and starts a new run if there is no active run.
        """
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        if not mlflow.active_run():
            mlflow.start_run()
        logger.info("MLflow experiment initialized for training.")

    def load_video_paths(self, data_path, split):
        """
        Loads video paths and labels for the given split.

        Args:
            data_path (str): Path to the dataset.
            split (str): The split to load. Can be "train", "val", or "test".

        Returns:
            tuple: A tuple of two lists. The first list contains the video paths,
            and the second list contains the corresponding labels.
        """
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
        """
        Prepare the data for training by creating DataLoaders.

        This function uses the configuration to load the data from the given data path,
        and creates DataLoaders for training and validation. The DataLoaders are stored
        in the ModelTraining instance as `train_loader` and `val_loader`.

        The data is loaded using the `load_video_paths` method, and the VideoDataset
        class is used to create the datasets. The datasets are then used to create
        DataLoaders.

        """

        # Data augmentation
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
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
            batch_size=self.config.batch_size // 4,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size // 4,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def train(self):
        """
        Train the model using the training strategy and the given data.

        Args:
            None

        Returns:
            None

        Steps:
        1. Create the model using the model strategy.
        2. Create the weighted loss criterion.
        3. Create the AdamW optimizer with weight decay.
        4. Create the cosine annealing learning rate scheduler.
        5. Train the model using the training strategy.
        6. Validate the model using the validation strategy.
        7. Save the model if the validation accuracy is improved.
        8. Log the training and validation metrics.
        9. Log the model with a valid signature to MLflow.
        """

        model = self.model_strategy.build_model(self.config).to(self.device)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.0]).to(self.device)
        )  # Weighted loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        best_val_acc = 0.0
        scaler = GradScaler()

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.training_strategy.train_epoch(
                model, self.train_loader, criterion, optimizer, self.device, scaler
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

        if mlflow.active_run():

            # Log metrics
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("train_acc", train_acc)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("val_acc", val_acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            # Log parameters
            mlflow.log_param("learning_rate", self.config.learning_rate)
            mlflow.log_param("weight_decay", self.config.weight_decay)
            mlflow.log_param("epochs", self.config.epochs)
            mlflow.log_param("sequence_length", self.config.sequence_length)

            # Define model signature
            input_example = next(iter(self.train_loader))[0][:1].to(self.device)
            model_signature = mlflow.models.infer_signature(
                input_example.cpu().numpy(), model(input_example).detach().cpu().numpy()
            )

            # Log model with signature
            mlflow.pytorch.log_model(
                model,
                "model",
                signature=model_signature,
                input_example=input_example.cpu().numpy(),
            )

        logger.info("Training completed!")

    def execute(self):
        """
        Execute the model training pipeline by preparing the data and then training the model.

        Steps:
        1. Prepare the data by creating the data loaders.
        2. Train the model.

        This function is wrapped in a try/except block to catch any exceptions that may occur during training.
        If an exception occurs, the error is logged and the exception is re-raised.

        Finally, if an MLflow run is active, the run is ended.
        """
        try:
            self.prepare_data()
            self.train()
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
        finally:
            if mlflow.active_run():
                mlflow.end_run()
                logger.info("MLflow run ended after training.")
