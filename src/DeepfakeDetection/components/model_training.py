import matplotlib

matplotlib.use("Agg")
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import albumentations as A
import cv2
import h5py
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.metrics import AUC

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import ModelTrainingConfig
from DeepfakeDetection.utils.common import load_h5py

# Load environment variables (e.g., for MLflow tracking URI)
load_dotenv()

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("Memory growth set for GPUs")
    except RuntimeError as e:
        logger.info(e)


# Strategy Interfaces
class ModelArchitectureStrategy(ABC):
    @abstractmethod
    def build_model(self, input_shape):
        """Method to define the model architecture"""
        pass


class ModelTrainingStrategy(ABC):
    @abstractmethod
    def train(self, model, train_data, val_data, config):
        """Method to train the model"""
        pass


# Concrete Implementation of ModelArchitectureStrategy
class EfficientNetTransformerArchitecture(ModelArchitectureStrategy):
    def build_model(self, config):
        base_model = EfficientNetB2(
            include_top=False, input_shape=tuple(config.input_shape), weights="imagenet"
        )

        base_model.trainable = False if config.pretrained else True

        inputs = layers.Input(shape=tuple(config.input_shape))
        x = preprocess_input(inputs)
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)

        # Transformer-like architecture with deeper attention layers
        x = layers.Reshape((-1, x.shape[-1]))(x)
        for _ in range(config.attention_depth):
            attn_output = layers.MultiHeadAttention(
                num_heads=config.num_heads, key_dim=config.key_dim
            )(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward network
        ffn = layers.Dense(config.units, activation="relu")(x)
        ffn = layers.Dense(x.shape[-1])(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.GlobalAveragePooling1D()(x)

        # Final classification head
        x = layers.Dense(
            config.units,
            activation=config.activation,
            kernel_regularizer=regularizers.l2(config.l2),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout_rate)(x)
        x = layers.Dense(
            config.units/2,
            activation=config.activation,
            kernel_regularizer=regularizers.l2(config.l2),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout_rate)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = models.Model(inputs, outputs)
        return model


class AugmentedTrainingStrategy(ModelTrainingStrategy):
    def __init__(self, config):
        self.config = config
        self.augmented_samples_saved = False

    def coarse_dropout(self, image, max_holes=8, max_height=70, min_height=30, max_width=70, min_width=30, p=0.5):
        if tf.random.uniform(()) > p:
            return image

        height, width, channels = self.config.input_shape

        for _ in range(max_holes):
            h = tf.random.uniform((), min_height, max_height, dtype=tf.int32)
            w = tf.random.uniform((), min_width, max_width, dtype=tf.int32)

            x = tf.random.uniform((), 0, width - w, dtype=tf.int32)
            y = tf.random.uniform((), 0, height - h, dtype=tf.int32)

            mask = tf.pad(
                tf.ones((h, w, channels), dtype=image.dtype),
                [[y, height - y - h], [x, width - x - w], [0, 0]],
            )

            image = image * (1 - mask)

        return image

    def augmentations(self, image):
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = self.coarse_dropout(
            image, max_holes=5, max_height=60, min_height=20,max_width=60, min_width=20,p=0.7
        )
        return image

    def augment(self, image, label):
        image = self.augmentations(image)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1)
        image = tf.clip_by_value(image + noise, 0.0, 255.0)
        image = tf.image.resize(image, tuple(self.config.input_shape[:2]))
        return image, label

    def save_augmented_samples(self, images, labels):
        if self.augmented_samples_saved:
            return

        for i in range(4):
            augmented_image, _ = self.augment(images[i], labels[i])
            cv2.imwrite(
                f"{self.config.root_dir}/augmented_image_{i}.jpg",
                cv2.cvtColor(augmented_image.numpy() * 255, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                f"{self.config.root_dir}/original_image_{i}.jpg",
                cv2.cvtColor(images[i].numpy() * 255, cv2.COLOR_RGB2BGR),
            )

        self.augmented_samples_saved = True

    def train(self, model, train_data, val_data, config):
        for images, labels in train_data.take(1):
            self.save_augmented_samples(images, labels)

        train_labels = load_h5py(Path(config.train_labels_path), "labels")
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
        class_weights = dict(enumerate(class_weights))
        logger.info(f"Computed Class Weights: {class_weights}")

        optimizer = (
            optimizers.Adam(
                learning_rate=optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=config.initial_learning_rate,
                    decay_steps=config.decay_steps,
                    decay_rate=config.decay_rate,
                )
            )
            if not config.const_lr
            else optimizers.Adam(learning_rate=config.initial_learning_rate)
        )

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", AUC(name="auc")],
        )

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(config.ckpt_path),
                monitor="val_auc",
                save_best_only=True,
                save_weights_only=True,
                mode="max",
            ),
            # tf.keras.callbacks.EarlyStopping(
            #     monitor="val_auc", patience=20, mode="max", restore_best_weights=True
            # ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        # Apply augmentation to the training data
        train_data = train_data.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)

        if config.buffer > 0:
            train_data = train_data.shuffle(buffer_size=config.buffer)

        # Calculate steps_per_epoch and validation_steps
        train_samples, val_samples = sum(1 for _ in train_data), sum(
            1 for _ in val_data
        )
        steps_per_epoch = train_samples // config.batch_size
        validation_steps = val_samples // config.batch_size

        logger.info(f"Training samples: {train_samples}")
        logger.info(f"Validation samples: {val_samples}")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Validation steps: {validation_steps}")

        history = model.fit(
            train_data.repeat(),
            validation_data=val_data.repeat(),
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            class_weight=class_weights,
            callbacks=callbacks,
        )

        return history


# ModelTraining Class
class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.training_strategy = AugmentedTrainingStrategy(self.config)
        self.architecture_strategy = EfficientNetTransformerArchitecture()
        self.model = self._build_model()
        self._initialize_mlflow()
        self._prepare_for_resume()

    def _initialize_mlflow(self):
        """Initialize MLflow experiment."""
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        if not mlflow.active_run():
            mlflow.start_run()
        logger.info("MLflow experiment initialized.")

    def _build_model(self):
        """Builds and returns the model."""
        model = self.architecture_strategy.build_model(config=self.config)
        return model

    def _prepare_for_resume(self):
        """Load weights from the last checkpoint if available."""
        checkpoint_path = os.path.join(self.config.ckpt_path)
        if Path(checkpoint_path).exists():
            logger.info("Loading model weights from the latest checkpoint.")
            self.model.load_weights(checkpoint_path)

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

    def load_data(self):
        """Load training and validation data using generators."""
        train_dataset = self.load_data_generator(
            self.config.train_data_path,
            self.config.train_labels_path,
            self.config.batch_size,
        )
        val_dataset = self.load_data_generator(
            self.config.val_data_path,
            self.config.val_labels_path,
            self.config.batch_size,
        )

        # Check class balance
        train_labels = load_h5py(Path(self.config.train_labels_path), "labels")
        val_labels = load_h5py(Path(self.config.val_labels_path), "labels")

        logger.info(f"Train class distribution: {np.bincount(train_labels)}")
        logger.info(f"Validation class distribution: {np.bincount(val_labels)}")

        return train_dataset, val_dataset

    def execute(self):
        """Execute the model training and saving."""
        logger.info("Starting model training...")

        try:
            # Load training and validation data
            train_data, val_data = self.load_data()

            # Verify data and get a sample for signature inference
            for x_batch, y_batch in train_data.take(1):
                logger.info(f"Sample input shape: {x_batch.shape}")
                logger.info(f"Sample label shape: {y_batch.shape}")

            # Train the model
            history = self.training_strategy.train(
                self.model,
                train_data,
                val_data,
                self.config,
            )

            logger.info(f"History: {history.history}")

            # Load the best weights
            self.model.load_weights(self.config.ckpt_path)

            # Save the entire model using SavedModel format and log weights in mlflow
            saved_model_path = self.config.model_path
            tf.saved_model.save(self.model, saved_model_path)
            logger.info(f"Saved model to {saved_model_path}")
            mlflow.log_artifact(self.config.ckpt_path, "model_weights")

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

        finally:
            mlflow.end_run()
