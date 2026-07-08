"""Tests for the dataset registry and manifest-driven datasets."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from occuwise.data import get_spec
from occuwise.data.datasets import ClassificationDataset
from occuwise.data.registry import DATASETS
from occuwise.data.transforms import build_transforms


def test_registry_specs_consistent():
    for name, spec in DATASETS.items():
        assert spec.name == name
        assert spec.task in {"classification", "segmentation"}
        assert len(spec.class_names) == spec.num_classes


def test_classification_dataset_reads_manifest(tmp_path):
    # Create two tiny fake images + a manifest.
    for i in range(2):
        Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(tmp_path / f"img{i}.png")
    pd.DataFrame({
        "image_path": ["img0.png", "img1.png"],
        "label": [0, 3],
        "split": ["train", "train"],
    }).to_csv(tmp_path / "manifest.csv", index=False)

    tf = build_transforms("classification", "fundus", 64, train=False)
    ds = ClassificationDataset(tmp_path / "manifest.csv", tmp_path, "train", tf)
    assert len(ds) == 2
    img, label = ds[1]
    assert img.shape == (3, 64, 64)
    assert label == 3


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError):
        get_spec("does_not_exist")
