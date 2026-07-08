"""Occuwise POC web app.

Two pages, matching the corereqs.md flow:

  GET /prepare   — Admin: prepare/ready a model for use, and run the smoke tests.
  GET /          — Execution: upload an eye image, run the model, display the result.

Run:
    py -m uvicorn serving.webapp:app --port 8080 --reload
    # then open http://localhost:8080

Prepared models live in memory (fast, no dataset needed). Preparing a model with
no checkpoint gives an ImageNet-pretrained backbone sized to the dataset's clinical
classes — the full pipeline runs and returns structured output, but it is untrained
(demo mode). Pass a checkpoint path to serve real predictions.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from occuwise.data.registry import DATASETS
from occuwise.models.registry import CLASSIFICATION_ARCHS, SEGMENTATION_ENCODERS

from .poc_predictor import PocPredictor

app = FastAPI(title="Occuwise POC", version="0.1.0")

WEB_DIR = Path(__file__).parent / "web"
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
REGISTRY: dict[str, PocPredictor] = {}

# Serve sample images (for the gallery thumbnails) if they've been fetched.
if SAMPLES_DIR.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


def _page(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def predict_page():
    return _page("predict.html")


@app.get("/prepare", response_class=HTMLResponse)
def prepare_page():
    return _page("prepare.html")


@app.get("/health")
def health():
    return {"status": "ok", "prepared_models": list(REGISTRY)}


@app.get("/api/options")
def options():
    """Datasets + compatible architectures for the prepare form."""
    datasets = []
    for name, spec in DATASETS.items():
        archs = (
            list(CLASSIFICATION_ARCHS) if spec.task == "classification"
            else list(SEGMENTATION_ENCODERS)
        )
        datasets.append({
            "name": name, "task": spec.task, "modality": spec.modality,
            "num_classes": spec.num_classes, "class_names": spec.class_names,
            "archs": archs, "description": spec.description,
        })
    return {"datasets": datasets}


@app.get("/api/models")
def list_models():
    return {"models": [p.summary() for p in REGISTRY.values()]}


@app.post("/api/prepare")
def prepare(payload: dict):
    dataset = payload.get("dataset")
    arch = payload.get("arch")
    checkpoint = payload.get("checkpoint") or None
    if not dataset or not arch:
        raise HTTPException(400, "dataset and arch are required")
    if dataset not in DATASETS:
        raise HTTPException(404, f"Unknown dataset '{dataset}'")
    try:
        predictor = PocPredictor(dataset, arch, checkpoint)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Failed to prepare model: {e}") from e
    REGISTRY[predictor.key] = predictor
    return {"status": "ready", "model": predictor.summary()}


def _run(model_key: str, image: np.ndarray, filename: str, explain: bool = False) -> dict:
    if model_key not in REGISTRY:
        raise HTTPException(404, f"Model '{model_key}' is not prepared. Prepare it first.")
    result = REGISTRY[model_key].predict(image, explain=explain)
    result["filename"] = filename
    result["disclaimer"] = "Decision support only — not a diagnosis. Clinician review required."
    return result


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model_key: str = Form(...),
                  explain: bool = Form(False)):
    raw = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not decode image: {e}") from e
    return _run(model_key, image, file.filename, explain)


@app.get("/api/samples")
def list_samples():
    """Bundled test images (empty until `py scripts/fetch_samples.py` is run)."""
    idx = SAMPLES_DIR / "samples.json"
    if not idx.exists():
        return {"samples": []}
    return {"samples": json.loads(idx.read_text(encoding="utf-8"))}


@app.post("/api/predict_sample")
def predict_sample(payload: dict):
    model_key = payload.get("model_key")
    rel = payload.get("path", "")
    # Prevent path traversal: the resolved file must stay under data/samples.
    target = (SAMPLES_DIR / rel).resolve()
    if not str(target).startswith(str(SAMPLES_DIR.resolve())) or not target.exists():
        raise HTTPException(404, f"Sample not found: {rel}")
    image = np.array(Image.open(target).convert("RGB"))
    return _run(model_key, image, target.name, bool(payload.get("explain", False)))


@app.post("/api/run-tests")
def run_tests():
    """Run the offline smoke tests and return the output."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=900,
    )
    return JSONResponse({
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-4000:],
    })
