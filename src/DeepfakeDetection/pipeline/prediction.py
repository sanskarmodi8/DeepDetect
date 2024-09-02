import tensorflow as tf
import numpy as np
import cv2
import io
import os
from typing import Tuple
from DeepfakeDetection.config.configuration import ConfigurationManager
from vit_keras import vit
from tensorflow.keras import layers, models


class Prediction:
    def __init__(self):
        # Load configuration
        config = ConfigurationManager().get_model_training_config()
        self.ckpt_path = os.path.join(config.root_dir, "ckpt.weights.h5")
        self.input_shape = tuple(
            config.input_shape
        )  # Make sure to load input_shape from config
        # Load the model
        self.model = self.load_model()

    def load_model(self):
        # Define model architecture
        vit_model = vit.vit_b16(
            image_size=self.input_shape[:2],  # Extract height and width
            activation="softmax",
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )

        inputs = layers.Input(shape=self.input_shape)
        x2 = vit_model(inputs)

        x = layers.Flatten()(x2)  # Flatten the output
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(2, activation="softmax")(x)

        model = models.Model(inputs, outputs)
        model.load_weights(self.ckpt_path)  # Load weights
        return model

    def preprocess_video(self, video_data: io.BytesIO) -> np.ndarray:
        cap = cv2.VideoCapture(video_data)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  # Resize frame
            frame = frame / 255.0  # Normalize
            frames.append(frame)

        cap.release()
        return np.array(frames)

    def predict(self, video_data: io.BytesIO) -> Tuple[str, float]:
        preprocessed_video = self.preprocess_video(video_data)

        predictions = self.model.predict(preprocessed_video)
        class_index = np.argmax(predictions, axis=-1)[0]
        confidence = np.max(predictions, axis=-1)[0]

        class_names = ["Real", "Fake"]
        prediction = class_names[class_index]

        return prediction, confidence
