"""Build the Messidor / Messidor-2 manifest.

Download: https://www.adcis.net/en/third-party/messidor/  (and Messidor-2)
Messidor grades retinopathy 0..3. Provide a collated CSV `data/messidor/index.csv`
with columns: image_path, grade   (grade in {0,1,2,3}).

Usage:  py scripts/prepare_messidor.py --root data/messidor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _manifest import stratified_splits, write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/messidor")
    ap.add_argument("--manifest", default=None,
                    help="Where to write the manifest CSV (default: <root>/manifest.csv). "
                         "Set this when --root is read-only, e.g. Kaggle /kaggle/input.")
    args = ap.parse_args()
    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.csv"

    df = pd.read_csv(root / "index.csv")
    df["label"] = df["grade"].astype(int)
    df = df[df["image_path"].apply(lambda p: (root / p).exists())].reset_index(drop=True)
    df = stratified_splits(df, "label")
    write_manifest(df, manifest, ["image_path", "label", "split"])


if __name__ == "__main__":
    main()
