import os
from pathlib import Path

import h5py
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from vit_keras import layers

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import ModelEvaluationConfig
from DeepfakeDetection.utils.common import load_h5py, save_json

# Load environment variables from the .env file
load_dotenv()


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = self._load_model()  # Load the model
        self._initialize_mlflow()  # Initialize MLflow

    def _initialize_mlflow(self):
        """Initialize MLflow experiment."""
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI")
        )  # Set the MLflow tracking URI
        mlflow.start_run()  # Start an MLflow run
        logger.info(
            "MLflow experiment initialized for evaluation."
        )  # Log initialization

    def _load_model(self):
        """Load the model with custom object scope for 'ClassToken'."""
        custom_objects = {
            "ClassToken": layers.ClassToken,  # Register ClassToken layer
        }

        model = tf.keras.models.load_model(
            self.config.model_path,
            custom_objects=custom_objects,  # Pass the custom objects to the loader
        )
        return model

    def data_generator(self, data_path, labels_path, batch_size):
        """Generator for loading evaluation data in batches."""
        with h5py.File(data_path, "r") as data_file, h5py.File(
            labels_path, "r"
        ) as labels_file:
            data = data_file["data"]
            labels = labels_file["labels"]
            for i in range(0, len(data), batch_size):
                yield data[i : i + batch_size], labels[i : i + batch_size]

    def evaluate_model(self):
        """Evaluate the model on the test set using a generator and log metrics to MLflow."""
        batch_size = 32  # Adjust this based on your memory constraints

        # Create the data generator
        eval_data_gen = self.data_generator(
            self.config.data_path, self.config.labels_path, batch_size
        )

        # Calculate steps based on the size of the dataset
        with h5py.File(self.config.data_path, "r") as f:
            total_samples = f["data"].shape[0]

        # Include an extra step for any remaining samples that don't fit exactly into a batch
        steps = (total_samples + batch_size - 1) // batch_size  # This rounds up

        # Make predictions using the generator
        predictions = self.model.predict(eval_data_gen, steps=steps)

        # Load the true labels separately to compare
        eval_labels = load_h5py(Path(self.config.labels_path), "labels")

        # Trim predictions if they exceed the number of true samples
        predictions = predictions[:total_samples]

        # Evaluate metrics (ensure eval_labels is in the correct shape)
        if len(eval_labels.shape) == 1:  # Binary classification
            true_classes = eval_labels[
                :total_samples
            ]  # Ensure correct number of samples
            auc = roc_auc_score(true_classes, predictions)  # Calculate AUC
        else:
            raise ValueError("Unexpected shape for eval_labels")

        # Convert predicted probabilities to class labels
        predicted_classes = (predictions > self.config.threshold).astype(int)

        # Calculate accuracy
        accuracy = accuracy_score(true_classes, predicted_classes)
        logger.info(f"Evaluation accuracy: {accuracy}")

        # Generate classification report
        report = classification_report(
            true_classes, predicted_classes, output_dict=True
        )
        logger.info(f"Classification Report: {report}")

        # Save the evaluation metrics
        metrics = {"accuracy": accuracy, "auc": auc, "classification_report": report}
        save_json(Path(self.config.score), metrics)
        logger.info(f"Evaluation metrics saved to {self.config.score}")

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_artifact(str(Path(self.config.score)))

        return metrics

    def execute(self):
        """Execute the model evaluation."""
        logger.info("Starting model evaluation...")  # Log the start of evaluation
        metrics = self.evaluate_model()  # Evaluate the model
        logger.info(
            f"Model evaluation completed with metrics: {metrics}"
        )  # Log the completion of evaluation
        mlflow.end_run()  # End the MLflow run
