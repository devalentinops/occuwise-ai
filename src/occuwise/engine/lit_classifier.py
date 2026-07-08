"""LightningModule for classification (ResNet / EfficientNet / DenseNet / ViT)."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..models import ModelConfig, build_model
from .losses import build_classification_loss
from .metrics import classification_metrics


class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        task_modality: str = "fundus",
        pretrained: bool = True,
        loss: str = "ce",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        warmup_epochs: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(
            ModelConfig(task="classification", arch=arch, num_classes=num_classes,
                        pretrained=pretrained)
        )
        self.criterion = build_classification_loss(loss, num_classes)
        self.train_metrics = classification_metrics(num_classes, prefix="train/")
        self.val_metrics = classification_metrics(num_classes, prefix="val/")
        self.test_metrics = classification_metrics(num_classes, prefix="test/")

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, metrics, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=1)
        # torchmetrics binary AUROC expects the positive-class score.
        preds = probs[:, 1] if probs.shape[1] == 2 else probs
        metrics.update(preds, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=stage == "train")
        return loss

    def training_step(self, batch, _):
        return self._step(batch, self.train_metrics, "train")

    def validation_step(self, batch, _):
        return self._step(batch, self.val_metrics, "val")

    def test_step(self, batch, _):
        return self._step(batch, self.test_metrics, "test")

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute()); self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True); self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute()); self.test_metrics.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
