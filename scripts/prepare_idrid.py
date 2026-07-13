"""Build the IDRiD Disease-Grading manifest — for EXTERNAL validation.

IDRiD (516 fundus images, ICDR grade 0-4) is not part of `combined_dr`, so it gives
an honest generalisation number for a model trained on the combined DR set.

The grading labels live in CSVs ("Image name", "Retinopathy grade", "Risk of macular
edema") with the images under a "Disease Grading" tree whose exact folder names vary
by Kaggle mirror. This script auto-finds every grading CSV, detects the id + DR-grade
columns by keyword, indexes all images by filename, and joins them.

By default EVERY image goes into the `test` split (evaluation-only), so:
    py -m occuwise.evaluate ckpt=<combined_dr checkpoint> data=idrid
runs the whole set as an external test. Pass --stratify to instead make a normal
train/val/test split (e.g. if you want to fine-tune on IDRiD).

Usage:
    py scripts/prepare_idrid.py --root /kaggle/input/idrid-dataset \
        --manifest /kaggle/working/idrid_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _manifest import stratified_splits, write_manifest
from prepare_eyepacs import _index_images


def _norm(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())


def _grade_col(cols) -> str | None:
    for c in cols:                                   # prefer "Retinopathy grade"
        n = _norm(c)
        if "retinopathy" in n and "grade" in n:
            return c
    for c in cols:                                   # any grade that isn't the edema one
        n = _norm(c)
        if "grade" in n and "edema" not in n and "macular" not in n and "risk" not in n:
            return c
    return None


def _id_col(cols) -> str | None:
    for c in cols:
        n = _norm(c)
        if "imagename" in n or n == "image" or "imageid" in n:
            return c
    for c in cols:
        n = _norm(c)
        if "image" in n or "name" in n or n == "id" or "idcode" in n:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/idrid")
    ap.add_argument("--manifest", default=None,
                    help="Where to write the manifest CSV (default: <root>/manifest.csv).")
    ap.add_argument("--stratify", action="store_true",
                    help="Make a train/val/test split instead of all-test (eval-only).")
    args = ap.parse_args()
    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.csv"

    frames = []
    for csv in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:  # noqa: BLE001
            continue
        gc, ic = _grade_col(df.columns), _id_col(df.columns)
        if gc and ic:
            sub = df[[ic, gc]].copy()
            sub.columns = ["_id", "_grade"]
            frames.append(sub)
            print(f"labels: {csv.name}  (id='{ic}', grade='{gc}', rows={len(sub)})")
    if not frames:
        raise SystemExit(f"No DR-grading CSV found under {root}. Pass the right --root.")

    labels = pd.concat(frames, ignore_index=True).dropna(subset=["_id", "_grade"])
    labels = labels.drop_duplicates(subset="_id")

    index = _index_images(root)
    print(f"found {len(index)} image files")

    def resolve(image_id):
        p = index.get(Path(str(image_id)).stem)
        return p.relative_to(root).as_posix() if p is not None else None

    labels["image_path"] = labels["_id"].map(resolve)
    labels["label"] = pd.to_numeric(labels["_grade"], errors="coerce")
    n_before = len(labels)
    df = labels.dropna(subset=["image_path", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    print(f"matched {len(df)}/{n_before} labelled images to files")
    if len(df) == 0:
        raise SystemExit("Matched 0 images — check --root points at the IDRiD folder.")

    if args.stratify:
        df = stratified_splits(df, "label")
    else:
        df["split"] = "test"      # evaluation-only external set

    write_manifest(df, manifest, ["image_path", "label", "split"])
    print("class distribution:\n" + df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
