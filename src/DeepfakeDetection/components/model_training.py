from DeepfakeDetection.entity.config_entity import ModelTrainingConfig
from DeepfakeDetection.utils.common import save_h5py, load_h5py
from DeepfakeDetection import logger
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from vit_keras import vit
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()


class ModelTraining:

    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.model = self._build_model()
        self._initialize_mlflow()  # Uncomment if using MLflow
        self._prepare_for_resume()

    def _initialize_mlflow(self):
        """Initialize MLflow experiment."""
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.start_run()
        logger.info("MLflow experiment initialized.")

    def _build_model(self):
        """Builds and returns the model."""
        vit_model = vit.vit_b16(
            image_size=tuple(self.config.input_shape[0:2]),
            activation="softmax",
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )

        inputs = layers.Input(shape=tuple(self.config.input_shape))
        x2 = vit_model(inputs)  # This should be a 2D tensor

        x = layers.Flatten()(x2)  # Add Flatten to convert 2D tensor to 1D
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(2, activation="softmax")(x)

        model = models.Model(inputs, outputs)
        return model

    def _prepare_for_resume(self):
        """Load weights from the last checkpoint if available."""
        checkpoint_path = os.path.join(self.config.root_dir, "ckpt.weights.h5")
        if Path(checkpoint_path).exists():
            logger.info("Loading model weights from the latest checkpoint.")
            self.model.load_weights(checkpoint_path)

    def compile_model(self):
        """Compile the model."""
        self.model.compile(
            optimizer=optimizers.Adam(self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        # Uncomment if logging parameters with MLflow
        # mlflow.log_param("learning_rate", self.config.learning_rate)
        # mlflow.log_param("batch_size", self.config.batch_size)
        # mlflow.log_param("epochs", self.config.epochs)
        # mlflow.log_param("input_shape", self.config.input_shape)

    def train_model(self, train_data, val_data):
        """Train the model and save the best model checkpoint."""
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.config.root_dir, "ckpt.weights.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, mode="max"
        )

        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=[checkpoint, early_stop],
        )

        # Uncomment if logging metrics with MLflow
        # for key, value in history.history.items():
        #     mlflow.log_metric(key, value[-1])

        return history

    def load_data(self):
        """Load training and validation data."""

        def parse_function(data, labels):
            return data, labels

        train_data = load_h5py(Path(self.config.train_data_path), "data")
        val_data = load_h5py(Path(self.config.val_data_path), "data")
        train_labels = load_h5py(Path(self.config.train_labels_path), "labels")
        val_labels = load_h5py(Path(self.config.val_labels_path), "labels")

        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))

        train_dataset = (
            train_dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_dataset = (
            val_dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_dataset, val_dataset

    def execute(self):
        """Execute the model training and saving."""
        logger.info("Starting model training...")

        train_data, val_data = self.load_data()

        self.compile_model()
        history = self.train_model(train_data, val_data)

        # Load weights to verify if the best checkpoint is saved correctly
        self.model.load_weights(os.path.join(self.config.root_dir, "ckpt.weights.h5"))

        # Save the entire model if needed
        self.model.save(Path(self.config.model_path))

        logger.info("History: %s", history.history)
        logger.info("Model training completed successfully.")
        # Uncomment if using MLflow
        # mlflow.end_run()
