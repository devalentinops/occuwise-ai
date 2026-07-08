"""Build REFUGE manifests — both the classification and segmentation views.

Download: https://refuge.grand-challenge.org/
REFUGE ships fundus images + optic disc/cup masks + a glaucoma label per image.
This script expects a pre-collated CSV `data/refuge/index.csv` with columns:
    image_path, mask_path, glaucoma   (glaucoma in {0,1})
(Adjust the loader below to your download's exact layout.)

Masks must be single-channel: 0=background, 1=optic disc, 2=optic cup.

Usage:  py scripts/prepare_refuge.py --root data/refuge
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _manifest import stratified_splits, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/refuge")
    args = ap.parse_args()
    root = Path(args.root)

    df = pd.read_csv(root / "index.csv")

    # Classification manifest (glaucoma vs not).
    cls = df.copy()
    cls["label"] = cls["glaucoma"].astype(int)
    cls = stratified_splits(cls, "label")
    write_manifest(cls, root / "manifest_cls.csv", ["image_path", "label", "split"])

    # Segmentation manifest (disc/cup masks). Reuse the same split assignment.
    seg = cls[["image_path", "mask_path", "split"]]
    write_manifest(seg, root / "manifest_seg.csv", ["image_path", "mask_path", "split"])


if __name__ == "__main__":
    main()
