"""Build the EyePACS (Kaggle DR Detection) manifest — robust to any mirror layout.

EyePACS is distributed in several shapes on Kaggle:
  * raw competition ("diabetic-retinopathy-detection"):   train/<id>.jpeg          + trainLabels.csv
  * tanlikesmath/diabetic-retinopathy-resized:            resized_train/resized_train/<id>.jpeg + trainLabels.csv
  * various 512/224 mirrors:                              <some folder>/<id>.{jpg,png} + labels csv
and the labels CSV uses different column names (image/level, image_id/dr_grade, ...).

Rather than hardcode one layout, this script:
  1. finds the labels CSV (auto or --labels),
  2. detects the id + grade columns from a known set,
  3. indexes EVERY image under --root by filename stem (any folder depth, any ext),
  4. joins labels to real files, reports how many matched, and writes the manifest
     with paths RELATIVE to --root (so it stays valid when data.data_root points there).

Usage (Kaggle, read-only input):
    py scripts/prepare_eyepacs.py \
        --root /kaggle/input/diabetic-retinopathy-resized \
        --manifest /kaggle/working/eyepacs_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _manifest import stratified_splits, write_manifest

IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp"}
LABELS_CSV_CANDIDATES = ["trainLabels.csv", "trainLabels_cropped.csv", "train.csv", "labels.csv"]
ID_COL_CANDIDATES = ["image", "image_id", "id_code", "id", "name", "filename", "image_name"]
LABEL_COL_CANDIDATES = ["level", "diagnosis", "dr_grade", "grade", "label", "class"]


def _find_labels_csv(root: Path, override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            raise SystemExit(f"--labels not found or is not a file: {p}")
        return p
    # NOTE: some datasets contain a *directory* whose name ends in ".csv"
    # (e.g. dreamer07/eyepacs). rglob matches those too, so we must keep only
    # real files here — never hand a directory to pd.read_csv.
    for name in LABELS_CSV_CANDIDATES:
        hits = sorted(q for q in root.rglob(name) if q.is_file())
        # Prefer the non-cropped / shallowest match.
        hits.sort(key=lambda q: ("crop" in str(q).lower(), len(q.parts)))
        if hits:
            return hits[0]
    for csv in sorted((q for q in root.rglob("*.csv") if q.is_file()), key=lambda q: len(q.parts)):
        try:
            cols = pd.read_csv(csv, nrows=0).columns.str.lower()
        except Exception:  # noqa: BLE001 - skip anything unreadable, keep scanning
            continue
        if any(c in cols for c in ID_COL_CANDIDATES):
            return csv
    raise SystemExit(
        f"No labels CSV found under {root}. Pass --labels <file> explicitly, or — if this "
        f"dataset stores the grade in class subfolders (0..4 / No_DR..Proliferative) rather "
        f"than a CSV — use scripts/prepare_combined_dr.py instead.")


def _pick_col(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise SystemExit(f"Could not find a {what} column in {list(df.columns)}. "
                     f"Expected one of {candidates}.")


def _index_images(root: Path) -> dict[str, Path]:
    """Map filename-stem -> path, for every image under root.

    When the same stem appears more than once (e.g. a dataset ships both
    resized_train/ and resized_train_cropped/, or train/ and test/), prefer the
    plain (non-'crop', non-'test') and shallowest path so results are deterministic.
    """
    # is_file() guards against directories whose names carry an image-like suffix.
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    files.sort(key=lambda p: ("crop" in str(p).lower(), "test" in str(p).lower(), len(p.parts)))
    index: dict[str, Path] = {}
    for p in files:
        index.setdefault(p.stem, p)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/eyepacs")
    ap.add_argument("--manifest", default=None,
                    help="Where to write the manifest CSV (default: <root>/manifest.csv). "
                         "Set this when --root is read-only, e.g. Kaggle /kaggle/input.")
    ap.add_argument("--labels", default=None,
                    help="Path to the labels CSV (auto-detected if omitted).")
    args = ap.parse_args()
    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.csv"

    labels_csv = _find_labels_csv(root, args.labels)
    df = pd.read_csv(labels_csv)
    id_col = _pick_col(df, ID_COL_CANDIDATES, "image-id")
    label_col = _pick_col(df, LABEL_COL_CANDIDATES, "grade")
    print(f"labels: {labels_csv}  (id='{id_col}', grade='{label_col}', rows={len(df)})")

    print(f"indexing images under {root} ...")
    index = _index_images(root)
    print(f"found {len(index)} image files")

    def resolve(image_id) -> str | None:
        stem = Path(str(image_id)).stem  # tolerate ids with or without extension
        p = index.get(stem)
        return p.relative_to(root).as_posix() if p is not None else None

    df["image_path"] = df[id_col].map(resolve)
    df["label"] = pd.to_numeric(df[label_col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["image_path", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    matched = len(df)
    print(f"matched {matched}/{n_before} label rows to image files "
          f"({n_before - matched} unmatched/dropped)")
    if matched == 0:
        raise SystemExit("Matched 0 images — check --root points at the folder that "
                         "contains the images, and that the CSV ids match filenames.")
    if matched < 0.9 * n_before:
        print("WARNING: >10% of labelled images were not found — double-check --root.")

    df = stratified_splits(df, "label")
    write_manifest(df, manifest, ["image_path", "label", "split"])
    print("class distribution:\n" + df["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
