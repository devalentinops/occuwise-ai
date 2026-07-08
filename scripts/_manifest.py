"""Shared helpers for turning raw datasets into standard manifest CSVs.

Manifest schema:
    classification -> image_path,label,split
    segmentation   -> image_path,mask_path,split
Paths are RELATIVE to the dataset root so the folder is relocatable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_splits(df: pd.DataFrame, label_col: str, val_frac=0.15, test_frac=0.15,
                      seed: int = 42) -> pd.DataFrame:
    """Add a `split` column with stratified train/val/test assignment."""
    train, temp = train_test_split(
        df, test_size=val_frac + test_frac, stratify=df[label_col], random_state=seed
    )
    rel_test = test_frac / (val_frac + test_frac)
    val, test = train_test_split(
        temp, test_size=rel_test, stratify=temp[label_col], random_state=seed
    )
    df = df.copy()
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def write_manifest(df: pd.DataFrame, out_path: str | Path, cols: list[str]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(df)} rows)")
    print(df["split"].value_counts().to_string())
