"""Command-line inference — run a model on an image, a folder, or the sample set.

Examples
--------
Single image with an untrained demo model (downloads pretrained backbone):
    py -m occuwise.predict --dataset aptos --arch resnet50 --image data/samples/fundus/Fundus_photograph_of_normal_left_eye.jpg

A whole folder:
    py -m occuwise.predict --dataset aptos --arch resnet50 --dir data/samples/fundus

Every bundled sample whose modality matches the dataset (fundus/OCT):
    py -m occuwise.predict --dataset aptos --arch efficientnet_b4 --samples

With real trained weights:
    py -m occuwise.predict --dataset aptos --arch resnet50 --ckpt outputs/aptos/resnet50/best.ckpt --samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data.registry import get_spec
from .predictor import InProcessPredictor

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _gather(args, spec) -> list[Path]:
    if args.image:
        return [Path(args.image)]
    if args.dir:
        return sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.samples:
        idx_file = Path("data/samples/samples.json")
        if not idx_file.exists():
            raise SystemExit("No samples found. Run:  py scripts/fetch_samples.py")
        idx = json.loads(idx_file.read_text())
        # Match the dataset's modality (fundus vs oct).
        return [Path("data/samples") / s["path"] for s in idx if s["modality"] == spec.modality]
    raise SystemExit("Provide one of --image, --dir, or --samples.")


def _format(result: dict) -> str:
    if result["task"] == "classification":
        top = result["predicted_class"]
        conf = result["confidence"]
        runners = ", ".join(f"{p['class']} {p['prob']*100:.0f}%" for p in result["probabilities"][:3])
        return f"{top:<16} ({conf*100:5.1f}%)   [{runners}]"
    areas = ", ".join(f"{k} {v*100:.0f}%" for k, v in result["class_area_fraction"].items())
    cdr = result.get("cup_to_disc_ratio")
    extra = f"  CDR={cdr:.3f}" if cdr is not None else ""
    return f"segmentation  [{areas}]{extra}"


def main():
    ap = argparse.ArgumentParser(description="Run an ophthalmology model on image(s).")
    ap.add_argument("--dataset", required=True, help="registered dataset name (sets task & classes)")
    ap.add_argument("--arch", required=True, help="architecture alias, e.g. resnet50")
    ap.add_argument("--ckpt", default=None, help="trained checkpoint (omit for untrained demo mode)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single image path")
    src.add_argument("--dir", help="folder of images")
    src.add_argument("--samples", action="store_true", help="bundled samples matching the modality")
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of a table")
    args = ap.parse_args()

    spec = get_spec(args.dataset)
    paths = _gather(args, spec)
    if not paths:
        raise SystemExit("No images to run.")

    print(f"Loading {args.arch} for {args.dataset} "
          f"({'trained' if args.ckpt else 'UNTRAINED demo'})...")
    predictor = InProcessPredictor(args.dataset, args.arch, args.ckpt)

    results = []
    for p in paths:
        res = predictor.predict(_load_rgb(p))
        res["image"] = str(p)
        results.append(res)
        if not args.json:
            print(f"  {p.name:<52} {_format(res)}")

    if args.json:
        print(json.dumps(results, indent=2))
    if not args.ckpt:
        print("\nNote: demo mode - predictions are from an untrained head and are not "
              "clinically meaningful. Train a model and pass --ckpt for real results.")


if __name__ == "__main__":
    main()
