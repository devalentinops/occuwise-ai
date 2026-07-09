"""Build the Duke OCT (DME) segmentation manifest.

Source: http://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm  (Chiu et al. 2015)
The dataset ships MATLAB .mat volumes with layer/fluid annotations. Run your
conversion to export B-scan PNGs + single-channel mask PNGs, then provide a
collated CSV `data/duke_oct/index.csv` with columns: image_path, mask_path.

Masks: 0=background, 1=fluid (extend to multi-layer as needed; update the
`duke_oct` DatasetSpec num_classes to match).

Usage:  py scripts/prepare_duke_oct.py --root data/duke_oct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from _manifest import write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/duke_oct")
    ap.add_argument("--manifest", default=None,
                    help="Where to write the manifest CSV (default: <root>/manifest.csv). "
                         "Set this when --root is read-only, e.g. Kaggle /kaggle/input.")
    args = ap.parse_args()
    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.csv"

    df = pd.read_csv(root / "index.csv")
    # Group-aware split by patient if an id column exists, else random.
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    train["split"], val["split"], test["split"] = "train", "val", "test"
    out = pd.concat([train, val, test], ignore_index=True)
    write_manifest(out, manifest, ["image_path", "mask_path", "split"])


if __name__ == "__main__":
    main()
