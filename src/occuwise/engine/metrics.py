"""Task-appropriate metric collections (torchmetrics).

DR grading is ordinal, so quadratic-weighted Cohen's kappa is the headline metric
(matches the Kaggle EyePACS/APTOS evaluation). Glaucoma is imbalanced binary, so
AUROC leads. OCT2017 uses macro-F1. Segmentation uses Dice + IoU.
"""

from __future__ import annotations

import torchmetrics as tm
from torchmetrics import MetricCollection


def classification_metrics(num_classes: int, prefix: str = "") -> MetricCollection:
    task = "binary" if num_classes == 2 else "multiclass"
    metrics = {
        "acc": tm.Accuracy(task=task, num_classes=num_classes),
        "macro_f1": tm.F1Score(task=task, num_classes=num_classes, average="macro"),
        "auroc": tm.AUROC(task=task, num_classes=num_classes),
        "quadratic_kappa": tm.CohenKappa(task=task, num_classes=num_classes, weights="quadratic"),
    }
    return MetricCollection(metrics, prefix=prefix)


def segmentation_metrics(num_classes: int, prefix: str = "") -> MetricCollection:
    metrics = {
        "dice": tm.Dice(num_classes=num_classes, average="macro", ignore_index=0),
        "iou": tm.JaccardIndex(task="multiclass", num_classes=num_classes),
    }
    return MetricCollection(metrics, prefix=prefix)
