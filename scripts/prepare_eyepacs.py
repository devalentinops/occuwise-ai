"""Build the EyePACS (Kaggle DR Detection) manifest.

Download: https://www.kaggle.com/c/diabetic-retinopathy-detection/data
Raw layout under data/eyepacs/:
    trainLabels.csv           (columns: image, level)
    train/<image>.jpeg

Usage:  py scripts/prepare_eyepacs.py --root data/eyepacs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _manifest import stratified_splits, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/eyepacs")
    ap.add_argument("--manifest", default=None,
                    help="Where to write the manifest CSV (default: <root>/manifest.csv). "
                         "Set this when --root is read-only, e.g. Kaggle /kaggle/input.")
    args = ap.parse_args()
    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.csv"

    df = pd.read_csv(root / "trainLabels.csv")
    df["image_path"] = df["image"].apply(lambda i: f"train/{i}.jpeg")
    df["label"] = df["level"].astype(int)
    df = df[df["image_path"].apply(lambda p: (root / p).exists())].reset_index(drop=True)
    df = stratified_splits(df, "label")
    write_manifest(df, manifest, ["image_path", "label", "split"])


if __name__ == "__main__":
    main()
