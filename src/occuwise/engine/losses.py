"""Loss functions.

Classification: label-smoothed CE by default; focal loss for heavy class imbalance
(EyePACS DR grade 0 dominates). Segmentation: Dice + CE, the standard robust combo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class DiceCELoss(nn.Module):
    """Multiclass Dice + cross-entropy for segmentation."""

    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        cardinality = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


def build_classification_loss(name: str, num_classes: int,
                              class_weights: torch.Tensor | None = None) -> nn.Module:
    if name == "focal":
        return FocalLoss(gamma=2.0, weight=class_weights)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
