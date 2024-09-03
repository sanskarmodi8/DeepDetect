import io

import gradio as gr

from DeepfakeDetection.pipeline.prediction import Prediction

predictor = Prediction()


def predict_video(file):
    """Predict the class of the uploaded video and return the prediction and confidence."""

    prediction, confidence = predictor.predict(file)
    return prediction, confidence


# Define Gradio interface
iface = gr.Interface(
    fn=predict_video,
    inputs=gr.File(label="Upload Video", file_types=["video"]),
    outputs=[gr.Textbox(label="Prediction"), gr.Number(label="Confidence")],
    title="Deepfake Detection",
    description="Upload a video to detect if it is a deepfake. The prediction and confidence score will be provided.",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)
