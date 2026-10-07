"""TireGuard inference API.

Run:  uvicorn ml.api.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs

The app sends one photo to POST /predict and gets back a condition (good / okay / defective),
or status "retake" if the photo is not usable.
"""
import json
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from ml.config import DEFAULT_THRESHOLDS, ONNX_PATH, THRESHOLDS_PATH
from ml.preprocessing import load_image, to_model_input
from ml.quality import check_quality

MODEL_VERSION = "effnet-b0-v1"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

MESSAGES = {
    "good": "No visible defects detected. Keep checking your tyres regularly.",
    "okay": "Possible wear or damage detected. We recommend a professional inspection soon.",
    "defective": "Visible defect detected (e.g. cracks, bulges or heavy wear). Get this tyre inspected "
                 "before your next long drive.",
}
DISCLAIMER = ("TireGuard is a visual screening aid, not a professional inspection. "
              "Always consult a qualified technician about tyre safety.")

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["session"] = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    state["thresholds"] = (json.loads(THRESHOLDS_PATH.read_text()) if THRESHOLDS_PATH.exists()
                           else DEFAULT_THRESHOLDS)
    yield
    state.clear()


app = FastAPI(title="TireGuard ML API", version=MODEL_VERSION, lifespan=lifespan)


def classify(p_defective: float, t: dict) -> str:
    if p_defective < t["good_below"]:
        return "good"
    if p_defective > t["defective_above"]:
        return "defective"
    return "okay"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION, "thresholds": state["thresholds"]}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image larger than 15 MB")
    try:
        img = load_image(data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File is not a readable image")

    quality = check_quality(img)
    if not quality.ok:
        return {"status": "retake", "message": quality.reason, "quality": quality.to_dict(),
                "model_version": MODEL_VERSION}

    probs = state["session"].run(None, {"image": to_model_input(img)})[0][0]
    p_defective = float(probs[1])
    condition = classify(p_defective, state["thresholds"])
    # Confidence in the reported level: distance-based for good/defective, uncertainty for okay.
    confidence = {"good": 1 - p_defective, "defective": p_defective,
                  "okay": 1 - abs(p_defective - 0.5) * 2}[condition]

    return {
        "status": "ok",
        "condition": condition,
        "confidence": round(float(np.clip(confidence, 0, 1)), 3),
        "probabilities": {"good": round(float(probs[0]), 4), "defective": round(p_defective, 4)},
        "message": MESSAGES[condition],
        "disclaimer": DISCLAIMER,
        "quality": quality.to_dict(),
        "model_version": MODEL_VERSION,
    }
