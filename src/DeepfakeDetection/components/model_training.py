import os
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers
from vit_keras import vit

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import ModelTrainingConfig
from DeepfakeDetection.utils.common import load_h5py

# Load environment variables (e.g., for MLflow tracking URI)
load_dotenv()


class ModelTraining:

    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.model = self._build_model()  # Initialize the model
        self._initialize_mlflow()  # Set up MLflow tracking
        self._prepare_for_resume()  # Resume from checkpoint if available

    def _initialize_mlflow(self):
        """Initialize MLflow experiment."""
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.start_run()  # Start an MLflow run
        logger.info("MLflow experiment initialized.")

    def _build_model(self):
        """Builds and returns the model."""
        # Load the pre-trained Vision Transformer model without the top layers
        vit_model = vit.vit_b16(
            image_size=tuple(self.config.input_shape[0:2]),
            activation="softmax",
            pretrained=self.config.pretrained,
            include_top=self.config.include_top,
            pretrained_top=self.config.pretrained_top,
        )

        inputs = layers.Input(shape=tuple(self.config.input_shape))  # Input layer
        x2 = vit_model(inputs)  # Output of the pre-trained model

        # Add custom layers
        x = layers.Flatten()(x2)  # Flatten the output
        x = layers.Dense(self.config.units, activation=self.config.activation)(
            x
        )  # Fully connected layer
        x = layers.BatchNormalization()(x)  # Batch normalization
        x = layers.Dropout(self.config.dropout_rate)(x)  # Dropout for regularization
        outputs = layers.Dense(2, activation="softmax")(x)  # Output layer

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
        class_weights = (
            self._compute_class_weights()
        )  # Compute class weights for imbalance
        self.model.compile(
            optimizer=optimizers.Adam(self.config.learning_rate),  # Adam optimizer
            loss="sparse_categorical_crossentropy",  # Loss function
            metrics=["accuracy"],  # Evaluation metric
        )
        # Log hyperparameters to MLflow
        mlflow.log_param("learning_rate", self.config.learning_rate)
        mlflow.log_param("batch_size", self.config.batch_size)
        mlflow.log_param("epochs", self.config.epochs)
        mlflow.log_param("units", self.config.units)
        mlflow.log_param("activation", self.config.activation)
        mlflow.log_param("dropout_rate", self.config.dropout_rate)

        return class_weights

    def _compute_class_weights(self):
        """Compute class weights to handle imbalance."""
        # Load the labels
        train_labels = load_h5py(Path(self.config.train_labels_path), "labels")
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
        return dict(enumerate(class_weights))

    def train_model(self, train_data, val_data):
        """Train the model and save the best model checkpoint."""
        class_weights = self.compile_model()  # Get class weights

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.config.root_dir, "ckpt.weights.h5"),
            monitor="val_accuracy",  # Monitor validation accuracy
            save_best_only=True,  # Save only the best model
            save_weights_only=True,  # Save only the model weights
            mode="max",
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, mode="max"  # Early stopping criteria
        )

        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            class_weight=class_weights,  # Include class weights
            callbacks=[checkpoint, early_stop],
        )

        # Log metrics to MLflow
        for key, value in history.history.items():
            mlflow.log_metric(key, value[-1])

        return history

    def load_data(self):
        """Load training and validation data."""

        # Load data from HDF5 files
        train_data = load_h5py(Path(self.config.train_data_path), "data")
        val_data = load_h5py(Path(self.config.val_data_path), "data")
        train_labels = load_h5py(Path(self.config.train_labels_path), "labels")
        val_labels = load_h5py(Path(self.config.val_labels_path), "labels")

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))

        # Preprocess and batch datasets
        train_dataset = train_dataset.batch(self.config.batch_size).prefetch(
            tf.data.AUTOTUNE
        )

        val_dataset = val_dataset.batch(self.config.batch_size).prefetch(
            tf.data.AUTOTUNE
        )

        return train_dataset, val_dataset

    def execute(self):
        """Execute the model training and saving."""
        logger.info("Starting model training...")

        # Load training and validation data
        train_data, val_data = self.load_data()

        # Compile and train the model
        self.compile_model()
        history = self.train_model(train_data, val_data)

        # Load weights from the best checkpoint
        self.model.load_weights(os.path.join(self.config.root_dir, "ckpt.weights.h5"))

        # Save the entire model for future use
        self.model.save(Path(self.config.model_path))

        # Infer the model's input signature and log the model to MLflow
        signature = mlflow.models.signature.infer_signature(
            train_data.map(lambda x, _: x).take(1).as_numpy_iterator().next(),
            self.model.predict(train_data.take(1)),
        )
        mlflow.keras.log_model(self.model, "model", signature=signature)

        logger.info("History: %s", history.history)
        logger.info("Model training completed successfully.")
        mlflow.end_run()  # End the MLflow run
