"""Build the OCT2017 (Kermany) manifest from its folder-per-class layout.

Download: https://data.mendeley.com/datasets/rscbjbr9sj  (or Kaggle "kermany2018")
Raw layout under data/oct2017/:
    OCT2017/train/<CLASS>/*.jpeg
    OCT2017/test/<CLASS>/*.jpeg          (CLASS in CNV, DME, DRUSEN, NORMAL)

We keep the official test split and carve a val split out of train.
Usage:  py scripts/prepare_oct2017.py --root data/oct2017
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from _manifest import write_manifest

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def _scan(split_dir: Path, root: Path) -> pd.DataFrame:
    rows = []
    for label, cls in enumerate(CLASSES):
        for img in (split_dir / cls).glob("*"):
            if img.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                rows.append({"image_path": str(img.relative_to(root)).replace("\\", "/"),
                             "label": label})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/oct2017")
    args = ap.parse_args()
    root = Path(args.root)
    base = root / "OCT2017"

    train = _scan(base / "train", root)
    test = _scan(base / "test", root)
    train, val = train_test_split(train, test_size=0.1, stratify=train["label"], random_state=42)
    train["split"], val["split"], test["split"] = "train", "val", "test"
    df = pd.concat([train, val, test], ignore_index=True)
    write_manifest(df, root / "manifest.csv", ["image_path", "label", "split"])


if __name__ == "__main__":
    main()
