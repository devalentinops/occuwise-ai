"""LightningDataModule that wires a registered dataset to train/val/test loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import ClassificationDataset, SegmentationDataset
from .registry import get_spec
from .transforms import build_transforms


class OphthalmologyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        data_root: str,
        manifest: str | None = None,
        image_size: int | None = None,
        batch_size: int = 16,
        num_workers: int = 4,
        balance_classes: bool = True,
    ):
        super().__init__()
        self.spec = get_spec(dataset)
        self.data_root = Path(data_root)
        self.manifest = manifest or str(self.data_root / "manifest.csv")
        self.image_size = image_size or self.spec.image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_classes = balance_classes
        self._ds_cls = (
            ClassificationDataset if self.spec.task == "classification" else SegmentationDataset
        )

    def _tf(self, train: bool):
        return build_transforms(self.spec.task, self.spec.modality, self.image_size, train)

    def setup(self, stage: str | None = None):
        self.train_ds = self._ds_cls(self.manifest, self.data_root, "train", self._tf(True))
        self.val_ds = self._ds_cls(self.manifest, self.data_root, "val", self._tf(False))
        self.test_ds = self._ds_cls(self.manifest, self.data_root, "test", self._tf(False))

    def _train_sampler(self):
        if not (self.balance_classes and self.spec.task == "classification"):
            return None
        labels = self.train_ds.labels()
        counts = np.bincount(labels, minlength=self.spec.num_classes).astype(float)
        counts[counts == 0] = 1.0
        weights = (1.0 / counts)[labels]
        return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights))

    def train_dataloader(self):
        sampler = self._train_sampler()
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, sampler=sampler,
            shuffle=sampler is None, num_workers=self.num_workers,
            pin_memory=True, drop_last=True, persistent_workers=self.num_workers > 0,
        )

    def _eval_loader(self, ds):
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return self._eval_loader(self.val_ds)

    def test_dataloader(self):
        return self._eval_loader(self.test_ds)
