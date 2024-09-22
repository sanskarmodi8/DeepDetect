import cv2
import face_recognition
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torchvision import transforms

from src.DeepfakeDetection.constants import PARAMS_FILE_PATH
from src.DeepfakeDetection.utils.common import read_yaml

load_dotenv()


class Prediction:
    def __init__(self):
        """
        Initialize the Prediction class with a pre-trained model and necessary parameters.
        """
        self.device = torch.device("cpu")
        self.model = mlflow.pytorch.load_model(
            "runs:/7e7834bd633249e69f35ec4c52288c73/model"
        )
        self.model.to(self.device)
        self.model.eval()
        self.expansion_factor = read_yaml(PARAMS_FILE_PATH).expansion_factor
        self.resolution = read_yaml(PARAMS_FILE_PATH).resolution
        self.frame_count = read_yaml(PARAMS_FILE_PATH).sequence_length

    def get_frames(self, video):
        """
        Yields frames from the given video file.
        """
        vidobj = cv2.VideoCapture(video)
        success, image = vidobj.read()
        while success:
            yield image
            success, image = vidobj.read()

    def get_face(self, frame):
        """
        Get the face locations from the given frames.
        """
        return face_recognition.face_locations(frame)

    def preprocess(self, video):
        """
        Preprocess the video by extracting frames, detecting faces, and resizing to the specified resolution.
        """
        frames = []
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    tuple(self.resolution),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        for idx, frame in enumerate(self.get_frames(video)):
            if idx < self.frame_count:
                face = self.get_face(frame)
                if len(face) > 0:
                    t, r, b, l = face[0]
                    h = b - t
                    w = l - r
                    t = max(0, t - int(h * self.expansion_factor / 2))
                    b = min(frame.shape[0], b + int(h * self.expansion_factor / 2))
                    l = max(0, l - int(w * self.expansion_factor / 2))
                    r = min(frame.shape[1], r + int(w * self.expansion_factor / 2))
                    cropped = cv2.resize(frame[t:b, l:r, :], tuple(self.resolution))
                    frames.append(cropped)
        frames = [transform(frame) for frame in frames]
        return frames

    def save_gradients(self, grad):
        """
        Hook function to capture gradients.
        """
        self.gradients = grad

    def grad_cam(self, fmap, grads):
        """
        Compute Grad-CAM using feature maps and gradients.
        """

        pooled_grads = torch.mean(grads, dim=[0])
        for i in range(fmap.shape[1]):
            fmap[:, i, :, :] *= pooled_grads[i]

        cam = torch.mean(fmap, dim=1).squeeze().cpu().detach().numpy()

        # Apply ReLU to retain only positive activations
        cam = np.maximum(cam, 0)

        # Normalize Grad-CAM
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam  # Prevent division by zero

        # Resize the cam to match the resolution of the original image
        cam = cv2.resize(cam, tuple(self.resolution))
        # Convert to single-channel by summing or taking one of the channels
        cam = np.sum(cam, axis=-1) if cam.shape[-1] > 1 else cam
        return cam

    def generate_gradcam(self, fmap, video_frame, grads):
        """
        Generate the Grad-CAM heatmap and overlay it on the frame.
        """
        cam = self.grad_cam(fmap, grads)
        # Ensure cam is a single-channel 8-bit image
        cam = np.uint8(255 * cam)  # Scale to 0-255
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # Apply colormap

        # Ensure video_frame is in the right format
        video_frame = np.float32(cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR))

        # Convert the normalized video_frame back to uint8 (0-255)
        video_frame = np.uint8(255 * video_frame)

        # Blend heatmap and original image with a weight to ensure the face is visible
        alpha = 0.01  # Lower weight for the heatmap to make face more visible
        beta = 1 - alpha  # Weight for the original frame
        overlayed_img = cv2.addWeighted(heatmap, alpha, video_frame, beta, 0)

        return overlayed_img

    def predict(self, video):
        frames = self.preprocess(video)
        input_tensor = torch.stack([frame.float() for frame in frames]).unsqueeze(0)
        input_tensor = input_tensor.view(1, self.frame_count, 3, *self.resolution)
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()

        # Forward pass to get feature maps and final output
        fmap, output = self.model(input_tensor)
        fmap.register_hook(self.save_gradients)

        frame_predictions = (
            F.softmax(output, dim=1).detach().cpu().numpy()[:, 1]
        )  # Get probability of class 1 (Fake)

        # Get the average prediction confidence across all frames
        avg_confidence = np.mean(frame_predictions)

        # Classify as REAL or FAKE based on confidence
        prediction = "REAL" if avg_confidence < 0.5 else "FAKE/MANIPULATED"
        confidence_percentage = (
            avg_confidence * 100
            if avg_confidence >= 0.5
            else (1 - avg_confidence) * 100
        )
        prediction_string = f"{prediction} : {confidence_percentage:.2f}% confidence"

        if prediction == "REAL":
            best_frame_index = np.argmin(frame_predictions)
        else:
            best_frame_index = np.argmax(frame_predictions)

        # Backpropagate to get gradients
        self.model.train()
        output[:, 1].backward()
        grads = self.gradients
        self.model.eval()

        # Generate Grad-CAM image
        gradcam_image = self.generate_gradcam(
            fmap, frames[best_frame_index].numpy().transpose(1, 2, 0), grads
        )

        return prediction_string, gradcam_image
