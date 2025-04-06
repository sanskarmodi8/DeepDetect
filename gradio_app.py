import os
import tempfile
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import gradio as gr

from src.DeepfakeDetection.pipeline.prediction import Prediction

# Initialize model
predictor = Prediction()

def predict_deepfake(
    video_file: str, 
    sequence_length: int = None
) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    """
    Process a video file and predict if it's a deepfake
    
    Args:
        video_file: path to uploaded video file
        sequence_length: number of frames to use for prediction
        
    Returns:
        prediction result, explanation image, and detailed information
    """
    try:
        # Get prediction and explanation image
        prediction_str, explanation_image, details = predictor.predict(
            video_file, sequence_length
        )
        
        return prediction_str, explanation_image, details
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"Error: {str(e)}", None, {"error_details": error_detail}

def process_prediction(
    video_file, 
    sequence_length: int = None
) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    """
    Wrapper function to handle the prediction process and format results for Gradio
    """
    if video_file is None:
        return "No video uploaded", None, {}
    
    # Convert sequence_length to int or None
    if sequence_length:
        try:
            sequence_length = int(sequence_length)
        except:
            sequence_length = None
    
    # Get prediction results
    prediction_str, explanation_image, details = predict_deepfake(video_file, sequence_length)
    
    # Format the detailed information for display
    formatted_details = ""
    if isinstance(details, dict):
        for key, value in details.items():
            formatted_details += f"**{key}**: {value}\n"
    
    return prediction_str, explanation_image, formatted_details

# Create Gradio interface
with gr.Blocks(title="Deepfake Detection") as demo:
    gr.Markdown("# Deepfake Detection System")
    gr.Markdown(
        """Upload a video to check if it's real or a manipulated deepfake 
        (Face2Face, FaceShifter, FaceSwap, or NeuralTextures)."""
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video")
            sequence_length = gr.Number(
                label="Sequence Length (Optional)", 
                info="Number of frames to use for prediction",
                precision=0
            )
            submit_btn = gr.Button("Analyze Video", variant="primary")
        
        with gr.Column(scale=2):
            result_label = gr.Label(label="Prediction Result")
            explanation_image = gr.Image(label="Explanation Visualization")
            details_md = gr.Markdown(label="Analysis Details")
    
    submit_btn.click(
        fn=process_prediction,
        inputs=[video_input, sequence_length],
        outputs=[result_label, explanation_image, details_md]
    )
    
    gr.Examples(
        examples=[
            ["sample_videos/real_video.mp4", None],
            ["sample_videos/deepfake_example.mp4", 32],
        ],
        inputs=[video_input, sequence_length],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)