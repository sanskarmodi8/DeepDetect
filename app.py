import gradio as gr

from src.DeepfakeDetection.pipeline.prediction import Prediction

# initialize prediction class
pred = Prediction()


def deepfake_detection(video):
    """
    Interface function for Gradio that takes a video as input and calls the Prediction class to make a prediction and generate an explainability image.
    """
    prediction, explainibility = pred.predict(video)
    return prediction, explainibility


interface = gr.Interface(
    fn=deepfake_detection,
    inputs=gr.Video(),
    outputs=["text", "image"],
    title="Deepfake Detection",
    description="Upload a video to check if it's real or fake/manipulated.",
)

if __name__ == "__main__":
    interface.launch()
