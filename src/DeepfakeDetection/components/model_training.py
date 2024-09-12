import os
import json
from pathlib import Path
from abc import ABC, abstractmethod

import cv2
import h5py
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.metrics import AUC
from efficientnet.tfkeras import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
import albumentations as A

from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import ModelTrainingConfig
from DeepfakeDetection.utils.common import load_h5py

# Load environment variables (e.g., for MLflow tracking URI)
load_dotenv()

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
        # EfficientNetB7 as feature extractor
        base_model = EfficientNetB7(include_top=False, input_shape=tuple(config.input_shape), weights='imagenet')
        base_model.trainable = False
        
        inputs = layers.Input(shape=tuple(config.input_shape))
        x = preprocess_input(inputs)
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)

        # Transformer for temporal dynamics across frames
        x = layers.Reshape((-1, x.shape[-1]))(x)
        transformer_layer = layers.Transformer(num_heads=config.num_heads, key_dim=config.key_dim, dropout=config.dropout_rate)(x)
        x = layers.GlobalAveragePooling1D()(transformer_layer)

        # Final classification head
        x = layers.Dense(config.units, activation=config.activation, kernel_regularizer=regularizers.l2(config.l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout_rate)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = models.Model(inputs, outputs)
        return model

# Concrete Implementation of ModelTrainingStrategy
class AugmentedTrainingStrategy(ModelTrainingStrategy):
    def train(self, model, train_data, val_data, train_samples, val_samples, config):
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
                os.path.join(config.root_dir, "ckpt.weights.h5"),
                monitor="val_auc",
                save_best_only=True,
                save_weights_only=True,
                mode="max",
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=10, mode="max", restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        # Augmentations
        def dfdc_augmentations(image):
            transform = A.Compose([
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                A.GaussNoise(p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.05),
                A.HorizontalFlip(),
                A.OneOf([
                    A.Resize(300, 300, interpolation=cv2.INTER_AREA),
                    A.Resize(300, 300, interpolation=cv2.INTER_LINEAR),
                ], p=1),
                A.PadIfNeeded(min_height=300, min_width=300, border_mode=cv2.BORDER_CONSTANT),
                A.OneOf([A.RandomBrightnessContrast(), A.HueSaturationValue()], p=0.7),
                A.ToGray(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.5)
            ])
            augmented = transform(image=image)
            return augmented['image']

        def augment(image, label):
            image = tf.numpy_function(dfdc_augmentations, [image], tf.float32)
            return image, label

        train_data = train_data.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_data = train_data.shuffle(buffer_size=config.buffer)

        steps_per_epoch = train_samples // config.batch_size
        validation_steps = val_samples // config.batch_size

        history = model.fit(
            train_data,
            validation_data=val_data,
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
        self.training_strategy = AugmentedTrainingStrategy()
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
        checkpoint_path = os.path.join(self.config.root_dir, "ckpt.weights.h5")
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
                tf.TensorSpec(shape=self.config.input_shape, dtype=tf.float32, name="data"), 
                tf.TensorSpec(shape=(), dtype=tf.int32 , name="labels"), 
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
        logger.info("Starting model training...")

        try:
            # Load training and validation data
            train_data, val_data, train_samples, val_samples = self.load_data()

            # Verify data
            for x_batch, y_batch in train_data.take(1):
                logger.info(f"Sample input shape: {x_batch.shape}")
                logger.info(f"Sample label shape: {y_batch.shape}")

            # Train the model
            history = self.training_strategy.train(
                self.model, train_data, val_data, train_samples, val_samples, self.config
            )
            
            logger.info(f"History: {history}")

            # Save model weights after training
            self.model.load_weights(self.config.ckpt_path)
            self.model.save(self.config.model_path)
            logger.info(f"Model saved at {self.config.model_path}")

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

        finally:
            mlflow.end_run()

