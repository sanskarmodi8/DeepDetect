import os
from pathlib import Path

import h5py
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow.keras.models import load_model

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import ModelEvaluationConfig
from DeepfakeDetection.utils.common import load_h5py, save_json

# Load environment variables from the .env file
load_dotenv()


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = self._load_model()
        self._initialize_mlflow()

    def _initialize_mlflow(self):
        """Initialize MLflow experiment."""
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        if not mlflow.active_run():
            mlflow.start_run()
        logger.info("MLflow experiment initialized for evaluation.")

    def _load_model(self):
        """Load the saved model using SavedModel format."""
        model = tf.saved_model.load(self.config.model_path)
        return model

    def load_data_generator(self, data_path, labels_path, batch_size):
        """Load data using a generator to avoid memory overload."""

        def data_generator():
            with h5py.File(data_path, "r") as data_file, h5py.File(
                labels_path, "r"
            ) as labels_file:
                data = data_file["data"]
                labels = labels_file["labels"]
                indices = np.arange(len(data))
                np.random.shuffle(indices)
                for i in indices:
                    x = data[i]
                    y = labels[i]
                    if y not in [0, 1]:
                        logger.warning(f"Invalid label encountered: {y}")
                        continue
                    yield x, y

        data_gen = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=self.config.input_shape, dtype=tf.float32, name="data"
                ),
                tf.TensorSpec(shape=(), dtype=tf.int32, name="labels"),
            ),
        )

        return data_gen.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def evaluate_model(self):
        """Evaluate the model on the test set using a generator and log metrics to MLflow."""
        eval_data = self.load_data_generator(
            self.config.data_path, self.config.labels_path, self.config.batch_size
        )

        # Calculate steps based on the size of the dataset
        eval_samples = sum(1 for _ in eval_data)
        steps = eval_samples // self.config.batch_size

        logger.info(f"Evaluation samples: {eval_samples}")
        logger.info(f"Evaluation steps: {steps}")

        # Make predictions using the generator
        predictions = []
        true_classes = []
        for x_batch, y_batch in eval_data.take(steps):
            batch_predictions = self.model(x_batch, training=False)
            predictions.extend(batch_predictions.numpy().flatten())
            true_classes.extend(y_batch.numpy())

        predictions = np.array(predictions)
        true_classes = np.array(true_classes)

        # Evaluate metrics
        auc = roc_auc_score(true_classes, predictions)

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
        logger.info("Starting model evaluation...")
        try:
            metrics = self.evaluate_model()
            logger.info(f"Model evaluation completed with metrics: {metrics}")

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise
        finally:
            mlflow.end_run()
