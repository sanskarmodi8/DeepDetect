import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from DeepfakeDetection.pipeline.prediction import Prediction

app = FastAPI(
    title="DeepDetect API",
    description="Deepfake Detection API with Grad-CAM support",
    version="1.0.0",
)

predictor = Prediction()


@app.get("/")
def health_check():
    return {"status": "ok", "message": "DeepDetect API is running"}


@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    """
    Accepts a video file and returns deepfake prediction + confidence.
    """

    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        prediction, gradcam_image, class_details = predictor.predict(tmp_path)

        response = {
            "prediction": prediction,
            "class_probabilities": class_details,
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        file.file.close()
