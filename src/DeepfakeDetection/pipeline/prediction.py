import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torchvision import transforms
import mediapipe as mp
import mlflow
from DeepfakeDetection.constants import PARAMS_FILE_PATH
from DeepfakeDetection.utils.common import read_yaml

load_dotenv()


class Prediction:
    def __init__(self):
        """
        Initialize the Prediction class with a pre-trained model and necessary parameters.
        """
        self.device = torch.device("cpu")
        self.model = mlflow.pytorch.load_model('runs:/6199e90b978f48fc868c514f78539d41/model', map_location=self.device)

        self.model.eval()
        params = read_yaml(PARAMS_FILE_PATH)
        self.expansion_factor = params.expansion_factor
        self.resolution = params.resolution
        self.default_frame_count = params.sequence_length

        # Initialize MediaPipe face detector
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

        # Define the classes for prediction
        self.classes = ["original", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

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
        Detect faces in a frame using MediaPipe.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (top, right, bottom, left) coordinates of the face or None if no face detected
        """
        try:
            # Convert frame from BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]  # Use the first detected face
                h, w, _ = frame.shape
                bboxC = detection.location_data.relative_bounding_box

                # Calculate absolute coordinates
                xmin = int(bboxC.xmin * w)
                ymin = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)

                # Return in top, right, bottom, left format
                top = max(ymin, 0)
                right = min(xmin + box_width, w)
                bottom = min(ymin + box_height, h)
                left = max(xmin, 0)
                
                return (top, right, bottom, left)
            
            return None  # No face detected

        except Exception as e:
            print(f"Error in get_face: {e}")
            print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            raise

    def color_jitter(self, image):
        """
        Applies color jitter to the given image for data augmentation.
        
        Args:
            image (np.ndarray): The input image
            
        Returns:
            np.ndarray: The color jittered image
        """
        rng = np.random.default_rng(seed=42)

        # Convert to HSV for easier manipulation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Adjust brightness
        value = rng.uniform(0.8, 1.2)
        v = cv2.multiply(v, value)

        # Adjust contrast
        mean = np.mean(v)
        value = rng.uniform(0.8, 1.2)
        v = cv2.addWeighted(v, value, mean, 1 - value, 0)

        # Adjust saturation
        value = rng.uniform(0.8, 1.2)
        s = cv2.multiply(s, value)

        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return image

    def preprocess(self, video, seq_length=None):
        """
        Preprocess the video by extracting frames, detecting faces, and resizing.
        Applies same preprocessing as training pipeline.
        
        Args:
            video (str): Path to the video file
            seq_length (int, optional): Number of frames to extract
            
        Returns:
            list: List of preprocessed frames
        """
        frames = []
        raw_frames = []  # Store original cropped frames for visualization
        
        # Use provided sequence length or default from params
        target_seq_length = seq_length if seq_length is not None else self.default_frame_count
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                tuple(self.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        buffer = []  # For processing in batches of 4 like training pipeline
        
        for idx, frame in enumerate(self.get_frames(video)):
            if len(frames) < target_seq_length:
                buffer.append(frame)
                
                if len(buffer) == 4:  # Process in batches of 4
                    faces = [self.get_face(f) for f in buffer]
                    
                    for i, face in enumerate(faces):
                        if face is not None:
                            top, right, bottom, left = face
                            face_height = bottom - top
                            face_width = right - left
                            
                            # Expand face region using expansion factor
                            expanded_top = max(0, top - int(self.expansion_factor / 2 * face_height))
                            expanded_bottom = min(buffer[i].shape[0], bottom + int(self.expansion_factor / 2 * face_height))
                            expanded_left = max(0, left - int(self.expansion_factor / 2 * face_width))
                            expanded_right = min(buffer[i].shape[1], right + int(self.expansion_factor / 2 * face_width))
                            
                            # Crop and resize
                            cropped_face = cv2.resize(
                                buffer[i][expanded_top:expanded_bottom, expanded_left:expanded_right, :],
                                tuple(self.resolution)
                            )
                            
                            # Store original cropped face for visualization
                            raw_frames.append(cropped_face.copy())
                            
                            # Apply color jitter like in training
                            cropped_face = self.color_jitter(cropped_face)
                            
                            # Transform for model input
                            transformed = transform(cropped_face)
                            frames.append(transformed)
                    
                    buffer = []  # Reset buffer
            else:
                break
                
        # Handle padding if we have fewer frames than required
        if len(frames) < target_seq_length:
            # If we have some frames, duplicate the last one
            if frames:
                while len(frames) < target_seq_length:
                    frames.append(frames[-1])
                    raw_frames.append(raw_frames[-1])
            else:
                return [], []  # No faces detected
                
        return frames[:target_seq_length], raw_frames[:target_seq_length]

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
        
        return cam

    def generate_gradcam(self, fmap, video_frame, grads):
        """
        Generate the Grad-CAM heatmap and overlay it on the frame.
        """
        cam = self.grad_cam(fmap, grads)
        
        # Ensure cam is a single-channel 8-bit image
        cam = np.uint8(255 * cam)  # Scale to 0-255
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # Apply colormap

        # Make sure video_frame is BGR for overlay
        if video_frame.shape[-1] == 3:  # Already BGR or RGB
            video_frame_bgr = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR) if video_frame.dtype != np.uint8 else video_frame
        else:
            video_frame_bgr = cv2.cvtColor(np.uint8(video_frame * 255), cv2.COLOR_RGB2BGR)

        # Blend heatmap and original image
        alpha = 0.4  # Weight for the heatmap
        beta = 1 - alpha  # Weight for the original frame
        overlayed_img = cv2.addWeighted(heatmap, alpha, video_frame_bgr, beta, 0)

        return overlayed_img

    def predict(self, video, seq_length=None):
        """
        Predict whether a video is real or fake.
        
        Args:
            video (str): Path to the video file
            seq_length (int, optional): Number of frames to use
            
        Returns:
            tuple: (prediction_result, gradcam_image, classification_details)
        """
        frames, raw_frames = self.preprocess(video, seq_length)
        
        if not frames:
            return "No faces detected in the video", None, None

        # Prepare input tensor for the model
        target_seq_length = seq_length if seq_length is not None else self.default_frame_count
        input_tensor = torch.stack(frames).unsqueeze(0)
        input_tensor = input_tensor.view(1, target_seq_length, 3, *self.resolution)
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()

        # Forward pass to get feature maps and final output
        fmap, attn_wts, output = self.model(input_tensor)
        fmap.register_hook(self.save_gradients)

        # Get predictions for all classes
        class_probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        
        # Get the predicted class
        predicted_class_idx = np.argmax(class_probs)
        predicted_class = self.classes[predicted_class_idx] if predicted_class_idx < len(self.classes) else "Unknown"
        
        confidence = class_probs[predicted_class_idx] * 100
        prediction_string = f"{predicted_class} : {confidence:.2f}% confidence"

        # Create detailed classification results
        classification_details = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": {
                self.classes[i]: class_probs[i] * 100 for i in range(len(self.classes))
            }
        }

        # Backpropagate for Grad-CAM
        self.model.zero_grad()
        output[0, predicted_class_idx].backward()
        grads = self.gradients

        # Generate Grad-CAM visualization for the best frame
        if raw_frames:
            # Choose middle frame for visualization
            middle_idx = len(raw_frames) // 2
            gradcam_image = self.generate_gradcam(
                fmap, raw_frames[middle_idx], grads
            )
        else:
            gradcam_image = None

        return prediction_string, gradcam_image, classification_details