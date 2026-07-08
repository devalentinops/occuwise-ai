"""LightningModule for segmentation (U-Net & friends)."""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from ..models import ModelConfig, build_model
from .losses import DiceCELoss
from .metrics import segmentation_metrics


class LitSegmenter(pl.LightningModule):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        decoder: str = "unet",
        pretrained: bool = True,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 80,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(
            ModelConfig(task="segmentation", arch=arch, num_classes=num_classes,
                        pretrained=pretrained, decoder=decoder)
        )
        self.criterion = DiceCELoss(num_classes=num_classes)
        self.val_metrics = segmentation_metrics(num_classes, prefix="val/")
        self.test_metrics = segmentation_metrics(num_classes, prefix="test/")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def _eval_step(self, batch, metrics, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        metrics.update(logits.argmax(1), y)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        return self._eval_step(batch, self.val_metrics, "val")

    def test_step(self, batch, _):
        return self._eval_step(batch, self.test_metrics, "test")

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True); self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute()); self.test_metrics.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
