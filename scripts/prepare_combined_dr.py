"""Build a manifest for a combined DR dataset (EyePACS+APTOS+Messidor and similar).

These merged Kaggle datasets encode the DR grade either as the image's parent
FOLDER name (0..4 or No_DR/Mild/Moderate/Severe/Proliferative) or, less often, in
a labels CSV. This script auto-detects which and works regardless of nesting:

  * class-folder mode: label = the first path component that names a DR class
    (handles `<root>/No_DR/x.jpg`, `<root>/train/0/x.jpg`, `<root>/aug/severe/x.jpg`)
  * csv mode (fallback): reuses the flexible EyePACS CSV matcher.

It prints the discovered class distribution and example paths so you can sanity-check
before a long run. Manifest paths are RELATIVE to --root.

Usage (Kaggle):
    py scripts/prepare_combined_dr.py \
        --root /kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy \
        --manifest /kaggle/working/combined_dr_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _manifest import stratified_splits, write_manifest

# Reuse the EyePACS CSV helpers for the fallback path.
from prepare_eyepacs import (  # noqa: E402
    IMAGE_EXTS,
    _find_labels_csv,
    _index_images,
    _pick_col,
    ID_COL_CANDIDATES,
    LABEL_COL_CANDIDATES,
)

# Normalised folder-name token -> DR grade (ICDR 0..4).
CLASS_TOKENS: dict[str, int] = {
    "0": 0, "nodr": 0, "healthy": 0, "normal": 0, "class0": 0,
    "1": 1, "mild": 1, "milddr": 1, "class1": 1,
    "2": 2, "moderate": 2, "moderatedr": 2, "class2": 2,
    "3": 3, "severe": 3, "severedr": 3, "class3": 3,
    "4": 4, "proliferate": 4, "proliferative": 4, "pdr": 4,
    "proliferatedr": 4, "proliferativedr": 4, "class4": 4,
}


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _label_from_path(rel: Path) -> int | None:
    # Look at folder components (deepest first); first DR-class token wins.
    for part in reversed(rel.parts[:-1]):
        grade = CLASS_TOKENS.get(_norm(part))
        if grade is not None:
            return grade
    return None


def _from_folders(root: Path) -> tuple[pd.DataFrame, int]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    rows = []
    for p in files:
        rel = p.relative_to(root)
        grade = _label_from_path(rel)
        if grade is not None:
            rows.append({"image_path": rel.as_posix(), "label": grade})
    return pd.DataFrame(rows), len(files)


def _from_csv(root: Path, labels: str | None) -> pd.DataFrame:
    csv = _find_labels_csv(root, labels)
    df = pd.read_csv(csv)
    id_col = _pick_col(df, ID_COL_CANDIDATES, "image-id")
    label_col = _pick_col(df, LABEL_COL_CANDIDATES, "grade")
    print(f"labels: {csv}  (id='{id_col}', grade='{label_col}', rows={len(df)})")
    index = _index_images(root)

    def resolve(image_id):
        p = index.get(Path(str(image_id)).stem)
        return p.relative_to(root).as_posix() if p is not None else None

    df["image_path"] = df[id_col].map(resolve)
    df["label"] = pd.to_numeric(df[label_col], errors="coerce")
    return df.dropna(subset=["image_path", "label"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/combined_dr")
    ap.add_argument("--manifest", default=None,
                    help="Where to write the manifest CSV (default: <root>/manifest.csv).")
    ap.add_argument("--labels", default=None, help="Force CSV mode with this labels file.")
    args = ap.parse_args()
    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.csv"

    if args.labels:
        df, mode = _from_csv(root, args.labels), "csv"
    else:
        print(f"scanning {root} for class-folder labels ...")
        folder_df, n_imgs = _from_folders(root)
        if n_imgs and len(folder_df) >= 0.5 * n_imgs:
            df, mode = folder_df, "class-folders"
            print(f"class-folder mode: labelled {len(folder_df)}/{n_imgs} images from folder names")
        else:
            print(f"class-folder mode covered only {len(folder_df)}/{n_imgs} — trying a labels CSV")
            df, mode = _from_csv(root, None), "csv"

    df["label"] = df["label"].astype(int)
    if len(df) == 0:
        raise SystemExit(
            "Found 0 labelled images. Check --root points at the dataset folder. "
            "If labels are in a CSV, pass --labels <file>. If folders use unusual class "
            "names, tell me the folder names and I'll extend the mapping.")

    df = stratified_splits(df, "label")
    write_manifest(df, manifest, ["image_path", "label", "split"])
    print(f"mode={mode}  total={len(df)}")
    print("class distribution:\n" + df["label"].value_counts().sort_index().to_string())
    print("example paths:")
    for pth in df["image_path"].head(3):
        print("   ", pth)


if __name__ == "__main__":
    main()
