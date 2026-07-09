"""Torch Datasets that read a normalised manifest CSV.

Every raw dataset is converted to a manifest by `scripts/prepare_*.py`, so these
two classes handle *all* datasets regardless of their original on-disk layout.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ClassificationDataset(Dataset):
    """Manifest columns: image_path, label, split."""

    def __init__(self, manifest: str | Path, root: str | Path, split: str, transform=None):
        df = pd.read_csv(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = _read_rgb(self.root / row["image_path"])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, int(row["label"])

    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy()


class SegmentationDataset(Dataset):
    """Manifest columns: image_path, mask_path, split.

    Masks are single-channel integer label maps (0=background, 1..K classes).
    """

    def __init__(self, manifest: str | Path, root: str | Path, split: str, transform=None):
        df = pd.read_csv(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = _read_rgb(self.root / row["image_path"])
        mask = cv2.imread(str(self.root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {row['mask_path']}")
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask.long()
