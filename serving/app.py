"""FastAPI inference service.

Serves one or more exported models registered via the MODEL_REGISTRY env/JSON.
Each model directory contains `model.onnx` + `model_card.json`.

Run:  py -m uvicorn serving.app:app --host 0.0.0.0 --port 8080
Docs: http://localhost:8080/docs

DISCLAIMER: This service provides clinical *decision support* (screening triage).
Outputs are not a diagnosis and must be reviewed by a qualified clinician.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .inference import OphthalmologyPredictor

app = FastAPI(title="Occuwise AI — Ophthalmology Inference", version="0.1.0")

# Registry: {model_key: model_dir}. Point MODEL_DIR at an exported model, or
# extend to a JSON manifest for multiple models.
_MODELS: dict[str, OphthalmologyPredictor] = {}
_DEFAULT_DIR = os.environ.get("MODEL_DIR", "models/production")


def _get(model_key: str) -> OphthalmologyPredictor:
    if model_key not in _MODELS:
        d = Path(_DEFAULT_DIR) if model_key == "default" else Path("models") / model_key
        if not (d / "model.onnx").exists():
            raise HTTPException(404, f"Model '{model_key}' not found at {d}")
        _MODELS[model_key] = OphthalmologyPredictor(d)
    return _MODELS[model_key]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models/{model_key}/card")
def card(model_key: str = "default"):
    return _get(model_key).card


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_key: str = "default"):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(415, "Expected an image upload.")
    raw = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not decode image: {e}") from e
    result = _get(model_key).predict(image)
    result["filename"] = file.filename
    result["disclaimer"] = "Decision support only — not a diagnosis. Review by a clinician required."
    return result
