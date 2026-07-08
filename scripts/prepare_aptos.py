"""Build the APTOS 2019 manifest.

Download (Kaggle): https://www.kaggle.com/c/aptos2019-blindness-detection/data
Expected raw layout under data/aptos/:
    train.csv                 (columns: id_code, diagnosis)
    train_images/<id_code>.png

Usage:  py scripts/prepare_aptos.py --root data/aptos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _manifest import stratified_splits, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/aptos")
    args = ap.parse_args()
    root = Path(args.root)

    df = pd.read_csv(root / "train.csv")
    df["image_path"] = df["id_code"].apply(lambda i: f"train_images/{i}.png")
    df["label"] = df["diagnosis"].astype(int)
    df = df[df["image_path"].apply(lambda p: (root / p).exists())].reset_index(drop=True)
    df = stratified_splits(df, "label")
    write_manifest(df, root / "manifest.csv", ["image_path", "label", "split"])


if __name__ == "__main__":
    main()
