import os
from pathlib import Path

import h5py
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.metrics import AUC
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
        if not mlflow.active_run():  # Start a run if none exists
            mlflow.start_run()
        logger.info("MLflow experiment initialized.")

    def _build_model(self):
        """Builds and returns the model."""
        vit_model = vit.vit_b16(
            image_size=tuple(self.config.input_shape[0:2]),
            activation="sigmoid",
            pretrained=self.config.pretrained,
            include_top=self.config.include_top,
            pretrained_top=self.config.pretrained_top,
            classes=1
        )

        inputs = layers.Input(shape=tuple(self.config.input_shape))
        x = vit_model(inputs)
        if not self.config.include_top:
            x = layers.Flatten()(x)
            x = layers.Dense(
                self.config.units,
                activation=self.config.activation,
                kernel_regularizer=regularizers.l2(self.config.l2),
            )(x)
            x = layers.BatchNormalization()(x)  # Add BatchNormalization
            x = layers.Dropout(self.config.dropout_rate)(x)
            x = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs, x)
        return model

    def _prepare_for_resume(self):
        """Load weights from the last checkpoint if available."""
        checkpoint_path = os.path.join(self.config.root_dir, "ckpt.weights.h5")
        if Path(checkpoint_path).exists():
            logger.info("Loading model weights from the latest checkpoint.")
            self.model.load_weights(checkpoint_path)

    def compile_model(self):
        """Compile the model."""
        class_weights = self._compute_class_weights()
        optimizer = optimizers.Adam(
            learning_rate=optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.initial_learning_rate,
                decay_steps=self.config.decay_steps,
                decay_rate=self.config.decay_rate,
            )
        )
        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", AUC(name="auc")],
        )

        # Log hyperparameters to MLflow
        mlflow.log_param("initial_learning_rate", self.config.initial_learning_rate)
        mlflow.log_param("decay_steps", self.config.decay_steps)
        mlflow.log_param("decay_rate", self.config.decay_rate)
        mlflow.log_param("batch_size", self.config.batch_size)
        mlflow.log_param("l2", self.config.l2)
        mlflow.log_param("rotation", self.config.rotation)
        mlflow.log_param("zoom", self.config.zoom)
        mlflow.log_param("epochs", self.config.epochs)
        mlflow.log_param("units", self.config.units)
        mlflow.log_param("activation", self.config.activation)
        mlflow.log_param("dropout_rate", self.config.dropout_rate)
        mlflow.log_param("pretrained", self.config.pretrained)

        return class_weights

    def _compute_class_weights(self):
        """Compute class weights to handle imbalance."""
        train_labels = load_h5py(Path(self.config.train_labels_path), "labels")
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
        return dict(enumerate(class_weights))

    def train_model(self, train_data, val_data, train_samples, val_samples):
        """Train the model and save the best model checkpoint."""
        class_weights = self.compile_model()

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.config.root_dir, "ckpt.weights.h5"),
                monitor="val_auc",
                save_best_only=True,
                save_weights_only=True,
                mode="max",
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=5, mode="max", restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        # Data augmentation
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(self.config.rotation),
                layers.RandomZoom(self.config.zoom),
                layers.RandomContrast(self.config.contrast),
                layers.GaussianNoise(self.config.gnoise),
            ]
        )

        def augment(image, label):
            return data_augmentation(image, training=True), label

        train_data = train_data.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_data = train_data.shuffle(buffer_size=self.config.buffer).repeat()
        val_data = val_data.repeat()

        # Adjust steps per epoch and validation steps to handle partial batches
        steps_per_epoch = train_samples // self.config.batch_size
        validation_steps = val_samples // self.config.batch_size

        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            class_weight=class_weights,
            callbacks=callbacks,
        )

        # Log epoch-wise metrics to MLflow
        for epoch, metrics in enumerate(history.history["accuracy"]):
            mlflow.log_metric("accuracy", metrics, step=epoch)
        for epoch, metrics in enumerate(history.history["val_accuracy"]):
            mlflow.log_metric("val_accuracy", metrics, step=epoch)
        for epoch, auc in enumerate(history.history["auc"]):
            mlflow.log_metric("auc", auc, step=epoch)
        for epoch, val_auc in enumerate(history.history["val_auc"]):
            mlflow.log_metric("val_auc", val_auc, step=epoch)

        return history

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
                tf.TensorSpec(shape=self.config.input_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )

        with h5py.File(data_path, "r") as f:
            total_samples = f["data"].shape[0]

        return data_gen.batch(batch_size).prefetch(tf.data.AUTOTUNE), total_samples

    def load_data(self):
        """Load training and validation data using generators."""
        train_dataset, train_samples = self.load_data_generator(
            self.config.train_data_path,
            self.config.train_labels_path,
            self.config.batch_size,
        )
        val_dataset, val_samples = self.load_data_generator(
            self.config.val_data_path,
            self.config.val_labels_path,
            self.config.batch_size,
        )

        # Check class balance
        train_labels = load_h5py(Path(self.config.train_labels_path), "labels")
        val_labels = load_h5py(Path(self.config.val_labels_path), "labels")

        logger.info(f"Train class distribution: {np.bincount(train_labels)}")
        logger.info(f"Validation class distribution: {np.bincount(val_labels)}")

        return train_dataset, val_dataset, train_samples, val_samples

    def execute(self):
        """Execute the model training and saving."""
        logger.info("Starting improved model training...")

        try:
            # Load training and validation data
            train_data, val_data, train_samples, val_samples = self.load_data()

            # Verify data
            for x_batch, y_batch in train_data.take(1):
                logger.info(f"Sample input shape: {x_batch.shape}")
                logger.info(f"Sample label shape: {y_batch.shape}")
                logger.info(f"Sample label values: {y_batch.numpy()}")

            # Compile and train the model
            history = self.train_model(train_data, val_data, train_samples, val_samples)
            logger.info(f"History: {history}")

            # Load weights from the best checkpoint
            self.model.load_weights(
                os.path.join(self.config.root_dir, "ckpt.weights.h5")
            )

            # Save the entire model
            self.model.save(os.path.join(self.config.root_dir, "model.h5"))

            # Prepare for MLflow model logging
            for batch in train_data.take(1):  # Fetch the first batch of training data
                inputs = batch[0].numpy()
                break  # Exit after first batch

            # Log model to MLflow with inferred signature
            signature = mlflow.models.infer_signature(
                model_input=inputs, model_output=self.model.predict(inputs)
            )
            mlflow.tensorflow.log_model(self.model, "model", signature=signature)
            mlflow.end_run()
        except Exception as e:
            logger.exception(e)
            raise e
