"""Export a trained checkpoint to production formats (TorchScript + ONNX).

Validates numerical parity between the PyTorch model and the exported ONNX graph,
then writes a `model_card.json` sidecar with provenance for the audit trail.

    py -m occuwise.export --ckpt outputs/aptos/resnet50/best.ckpt --dataset aptos --onnx --torchscript
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data.registry import get_spec
from .engine import load_trained_module


def load_model(ckpt: str, task: str):
    # Load trained weights without re-fetching (possibly gated) pretrained backbones,
    # then return the underlying nn.Module. Tracing/exporting the LightningModule wrapper
    # trips torch.jit's attribute introspection on its `.trainer` property.
    module = load_trained_module(ckpt, task, map_location="cpu")
    return module.model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--torchscript", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    size = args.image_size or spec.image_size
    out_dir = Path(args.out or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, spec.task)
    example = torch.randn(1, 3, size, size)
    with torch.no_grad():
        ref = model(example).cpu().numpy()

    if args.torchscript:
        ts = torch.jit.trace(model, example)
        ts_path = out_dir / "model.torchscript.pt"
        ts.save(str(ts_path))
        print(f"TorchScript -> {ts_path}")

    if args.onnx:
        onnx_path = out_dir / "model.onnx"
        torch.onnx.export(
            model, example, str(onnx_path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=args.opset,
            dynamo=False,  # legacy TorchScript exporter (no onnxscript dependency)
        )
        # Parity check.
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": example.numpy()})[0]
        max_diff = float(np.abs(ref - ort_out).max())
        print(f"ONNX -> {onnx_path}  (max |pytorch-onnx| = {max_diff:.2e})")
        assert max_diff < 1e-3, "ONNX parity check failed!"

    card = {
        "dataset": spec.name,
        "task": spec.task,
        "modality": spec.modality,
        "num_classes": spec.num_classes,
        "class_names": spec.class_names,
        "image_size": size,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "primary_metric": spec.primary_metric,
        "checkpoint": str(args.ckpt),
        "intended_use": "Clinical decision support (screening triage). NOT a standalone diagnosis.",
        "limitations": "Trained on public datasets; validate on local population before deployment.",
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"model_card.json -> {out_dir/'model_card.json'}")


if __name__ == "__main__":
    main()
