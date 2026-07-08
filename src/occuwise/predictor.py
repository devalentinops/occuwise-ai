"""In-process PyTorch predictor (shared by the CLI and the POC web app).

Unlike serving/inference.py (which serves an exported ONNX model in production),
this runs the PyTorch model directly in memory. That lets you predict with one
call — no dataset, no training run, no export step required — so the full
image -> process -> structured output flow works immediately.

A model is backed by either:
  * a fresh ImageNet-pretrained backbone with a head sized to the dataset's
    clinical classes (demo mode — structurally correct output, not yet trained), or
  * a trained checkpoint if you pass one (real predictions).
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from occuwise.data.registry import DatasetSpec, get_spec
from occuwise.data.transforms import build_transforms
from occuwise.models import ModelConfig, build_model

# Distinct colours for segmentation class overlays (RGB).
_SEG_COLOURS = np.array(
    [[0, 0, 0], [255, 64, 64], [64, 160, 255], [64, 255, 128], [255, 200, 0]],
    dtype=np.uint8,
)


def _input_size(arch: str, spec: DatasetSpec) -> int:
    # ViT backbones are fixed at 224px; everything else uses the dataset default.
    return 224 if "vit" in arch else spec.image_size


class InProcessPredictor:
    def __init__(self, dataset: str, arch: str, checkpoint: str | None = None):
        self.spec = get_spec(dataset)
        self.arch = arch
        self.input_size = _input_size(arch, self.spec)
        self.trained = checkpoint is not None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if checkpoint:
            self.model = self._load_checkpoint(checkpoint)
        else:
            cfg = ModelConfig(
                task=self.spec.task, arch=arch,
                num_classes=self.spec.num_classes, pretrained=True,
                decoder="unet" if self.spec.task == "segmentation" else "unet",
            )
            self.model = build_model(cfg)
        self.model.eval().to(self.device)
        self.transform = build_transforms(
            self.spec.task, self.spec.modality, self.input_size, train=False
        )

    def _load_checkpoint(self, ckpt: str):
        from occuwise.engine import LitClassifier, LitSegmenter

        cls = LitClassifier if self.spec.task == "classification" else LitSegmenter
        return cls.load_from_checkpoint(ckpt, map_location="cpu").model

    @property
    def key(self) -> str:
        return f"{self.spec.name}__{self.arch}"

    def _tensor(self, image_rgb: np.ndarray) -> torch.Tensor:
        x = self.transform(image=image_rgb)["image"]
        return x.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> dict:
        x = self._tensor(image_rgb)
        if self.spec.task == "classification":
            return self._predict_classification(x)
        return self._predict_segmentation(x, image_rgb)

    def _predict_classification(self, x: torch.Tensor) -> dict:
        probs = F.softmax(self.model(x), dim=1)[0].cpu().numpy()
        order = np.argsort(probs)[::-1]
        idx = int(order[0])
        return {
            "task": "classification",
            "dataset": self.spec.name,
            "modality": self.spec.modality,
            "model": self.arch,
            "trained": self.trained,
            "predicted_class": self.spec.class_names[idx],
            "predicted_index": idx,
            "confidence": float(probs[idx]),
            "probabilities": [
                {"class": self.spec.class_names[i], "prob": float(probs[i])} for i in order
            ],
        }

    def _predict_segmentation(self, x: torch.Tensor, image_rgb: np.ndarray) -> dict:
        logits = self.model(x)
        mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # HxW
        total = mask.size
        areas = {
            self.spec.class_names[c]: float((mask == c).sum() / total)
            for c in range(self.spec.num_classes)
        }
        result = {
            "task": "segmentation",
            "dataset": self.spec.name,
            "modality": self.spec.modality,
            "model": self.arch,
            "trained": self.trained,
            "class_area_fraction": areas,
            "overlay_png": self._overlay(image_rgb, mask),
        }
        # Clinically useful derived metric for REFUGE optic disc/cup.
        if {"optic_disc", "optic_cup"} <= set(self.spec.class_names):
            disc = (mask >= self.spec.class_names.index("optic_disc")).sum()
            cup = (mask == self.spec.class_names.index("optic_cup")).sum()
            result["cup_to_disc_ratio"] = float(cup / disc) if disc > 0 else None
        return result

    def _overlay(self, image_rgb: np.ndarray, mask: np.ndarray) -> str:
        base = np.array(
            Image.fromarray(image_rgb).resize((mask.shape[1], mask.shape[0]))
        ).astype(np.float32)
        colour = _SEG_COLOURS[np.clip(mask, 0, len(_SEG_COLOURS) - 1)].astype(np.float32)
        alpha = (mask > 0)[..., None] * 0.45
        blended = (base * (1 - alpha) + colour * alpha).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def summary(self) -> dict:
        return {
            "key": self.key,
            "dataset": self.spec.name,
            "arch": self.arch,
            "task": self.spec.task,
            "modality": self.spec.modality,
            "num_classes": self.spec.num_classes,
            "class_names": self.spec.class_names,
            "input_size": self.input_size,
            "trained": self.trained,
            "device": self.device,
        }


# Backwards-compatible alias.
PocPredictor = InProcessPredictor
