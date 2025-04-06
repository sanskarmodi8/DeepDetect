import tempfile
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from src.DeepfakeDetection.pipeline.prediction import Prediction

app = FastAPI(
    title="Deepfake Detection API",
    description="Upload a video to check if it's real or a manipulated deepfake (Face2Face, FaceShifter, FaceSwap, or NeuralTextures).",
)

# CORS (optional if using frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
predictor = Prediction()


@app.post("/predict/")
async def predict_deepfake(
    video: UploadFile = File(...),
    sequence_length: Optional[int] = Query(
        None, description="Number of frames to use for prediction"
    ),
):
    try:
        # Save video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await video.read())
            temp_video_path = temp_video.name

        # Get prediction and explanation image
        prediction_str, explanation_image, details = predictor.predict(
            temp_video_path, sequence_length
        )

        response = {"prediction": prediction_str, "details": details}

        # Convert explanation image (np array) to JPEG bytes if available
        if explanation_image is not None:
            _, img_encoded = cv2.imencode(".jpg", explanation_image)
            img_bytes = BytesIO(img_encoded.tobytes())
            return StreamingResponse(
                content=img_bytes,
                media_type="image/jpeg",
                headers={"X-Prediction-Result": prediction_str},
            )
        else:
            return JSONResponse(content=response)

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        return JSONResponse(
            status_code=500, content={"error": str(e), "detail": error_detail}
        )


@app.get("/")
def root():
    return {
        "message": "Deepfake Detection API is running!",
        "usage": "POST to /predict/ with a video file and optional sequence_length parameter",
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)