import gradio as gr
import numpy as np
import cv2
from prediction import Prediction

# Initialize the Prediction class
predictor = Prediction()

def inference(video):
    """
    Gradio-compatible inference function.

    Args:
        video (str): Path to the uploaded video file

    Returns:
        tuple: (Prediction string, Grad-CAM image, Classification details)
    """
    prediction, gradcam_image, classification_details = predictor.predict(video)

    # Convert Grad-CAM image to RGB for display in Gradio
    if gradcam_image is not None:
        gradcam_image = cv2.cvtColor(gradcam_image, cv2.COLOR_BGR2RGB)
    else:
        gradcam_image = np.zeros((256, 256, 3), dtype=np.uint8)  # fallback image

    return prediction, gradcam_image, classification_details

# Define Gradio interface
demo = gr.Interface(
    fn=inference,
    inputs=gr.Video(label="Upload a video"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(label="Grad-CAM Visualization"),
        gr.Label(label="Classification Confidence Scores"),  # Updated label
    ],
    title="Deepfake Detection with Grad-CAM",
    description="Upload a video and detect whether it is real or a deepfake. Grad-CAM will highlight the most influential region.",
    theme="default",
)

if __name__ == "__main__":
    demo.launch(debug=True)
