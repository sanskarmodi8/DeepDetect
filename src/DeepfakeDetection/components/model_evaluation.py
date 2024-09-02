from DeepfakeDetection import logger
import os
from DeepfakeDetection.entity.config_entity import ModelEvaluationConfig
from DeepfakeDetection.utils.common import save_json, load_h5py
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = self._load_model()
        # self._initialize_mlflow()

    # def _initialize_mlflow(self):
    #     """Initialize MLflow experiment."""
    #     mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    #     mlflow.start_run()
    #     logger.info("MLflow experiment initialized for evaluation.")

    def _load_model(self):
        """Loads the model from the given path."""
        model = tf.keras.models.load_model(self.config.model_path)
        logger.info(f"Model loaded from {self.config.model_path}")
        return model

    def evaluate_model(self):
        # """Evaluate the model on the test set and log metrics to MLflow."""
        # Load evaluation data
        eval_data = load_h5py(Path(self.config.data_path), "data")
        eval_labels = load_h5py(Path(self.config.labels_path), "labels")

        # Make predictions
        predictions = self.model.predict(eval_data)

        # Ensure eval_labels is in the correct shape
        if len(eval_labels.shape) == 1:  # Binary classification
            true_classes = eval_labels
            # Predictions for binary classification should have shape [num_samples, 2]
            if predictions.shape[1] == 2:
                auc = roc_auc_score(true_classes, predictions[:, 1])
            else:
                raise ValueError("Predictions shape mismatch for binary classification")
        elif len(eval_labels.shape) == 2:  # Multiclass classification
            true_classes = np.argmax(eval_labels, axis=1)
            auc = roc_auc_score(true_classes, predictions, multi_class="ovr")
        else:
            raise ValueError("Unexpected shape for eval_labels")

        # Calculate accuracy
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(true_classes, predicted_classes)
        logger.info(f"Evaluation accuracy: {accuracy}")

        # Generate classification report
        report = classification_report(
            true_classes, predicted_classes, output_dict=True
        )
        logger.info(f"Classification Report: {report}")

        # Save the evaluation metrics
        metrics = {"accuracy": accuracy, "auc": auc, "classification_report": report}
        save_json(Path(self.config.score), metrics)  # Convert to Path object
        logger.info(f"Evaluation metrics saved to {self.config.score}")

        # Log metrics to MLflow
        # mlflow.log_metric("accuracy", accuracy)
        # mlflow.log_metric("auc", auc)
        # mlflow.log_artifacts(self.config.score)

        return metrics

    def execute(self):
        """Execute the model evaluation."""
        logger.info("Starting model evaluation...")
        metrics = self.evaluate_model()
        logger.info(f"Model evaluation completed with metrics: {metrics}")
        # mlflow.end_run()
