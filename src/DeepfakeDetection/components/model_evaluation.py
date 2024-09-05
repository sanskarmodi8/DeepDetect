import os
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

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
        """Loads the model from the given path."""
        model = tf.keras.models.load_model(
            self.config.model_path
        )  # Load the model from the specified path
        logger.info(
            f"Model loaded from {self.config.model_path}"
        )  # Log that the model was loaded
        return model

    def evaluate_model(self):
        """Evaluate the model on the test set and log metrics to MLflow."""
        # Load evaluation data
        eval_data = load_h5py(
            Path(self.config.data_path), "data"
        )  # Load evaluation data
        eval_labels = load_h5py(
            Path(self.config.labels_path), "labels"
        )  # Load evaluation labels

        # Make predictions
        predictions = self.model.predict(eval_data)  # Get model predictions

        # Ensure eval_labels is in the correct shape
        if len(eval_labels.shape) == 1:  # Binary classification
            true_classes = eval_labels  # Use the labels as is for binary classification
            if predictions.shape[1] == 2:
                auc = roc_auc_score(
                    true_classes, predictions[:, 1]
                )  # Calculate AUC for binary classification
            else:
                raise ValueError("Predictions shape mismatch for binary classification")
        elif len(eval_labels.shape) == 2:  # Multiclass classification
            true_classes = np.argmax(
                eval_labels, axis=1
            )  # Convert one-hot labels to class indices
            auc = roc_auc_score(
                true_classes, predictions, multi_class="ovr"
            )  # Calculate AUC for multiclass classification
        else:
            raise ValueError("Unexpected shape for eval_labels")

        # Calculate accuracy
        predicted_classes = np.argmax(
            predictions, axis=1
        )  # Get predicted class indices
        accuracy = accuracy_score(true_classes, predicted_classes)  # Calculate accuracy
        logger.info(f"Evaluation accuracy: {accuracy}")  # Log accuracy

        # Generate classification report
        report = classification_report(
            true_classes, predicted_classes, output_dict=True
        )  # Generate classification report
        logger.info(f"Classification Report: {report}")  # Log classification report

        # Save the evaluation metrics
        metrics = {"accuracy": accuracy, "auc": auc, "classification_report": report}
        save_json(Path(self.config.score), metrics)  # Save metrics to JSON file
        logger.info(
            f"Evaluation metrics saved to {self.config.score}"
        )  # Log saving of metrics

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)  # Log accuracy to MLflow
        mlflow.log_metric("auc", auc)  # Log AUC to MLflow
        mlflow.log_artifact(
            str(Path(self.config.score))
        )  # Log the metrics JSON as an artifact

        return metrics  # Return the metrics

    def execute(self):
        """Execute the model evaluation."""
        logger.info("Starting model evaluation...")  # Log the start of evaluation
        metrics = self.evaluate_model()  # Evaluate the model
        logger.info(
            f"Model evaluation completed with metrics: {metrics}"
        )  # Log the completion of evaluation
        mlflow.end_run()  # End the MLflow run
