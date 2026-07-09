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
        weights_repo: str | None = None,
        weights_file: str | None = None,
        weights_key: str | None = None,
        backbone_lr_scale: float = 1.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(
            ModelConfig(task="classification", arch=arch, num_classes=num_classes,
                        pretrained=pretrained, weights_repo=weights_repo,
                        weights_file=weights_file, weights_key=weights_key)
        )
        if freeze_backbone:
            for p in self._backbone_params():
                p.requires_grad_(False)
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
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True,
                 on_step=stage == "train", sync_dist=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, self.train_metrics, "train")

    def validation_step(self, batch, _):
        return self._step(batch, self.val_metrics, "val")

    def test_step(self, batch, _):
        return self._step(batch, self.test_metrics, "test")

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), sync_dist=True); self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True); self.test_metrics.reset()

    def _head_params(self):
        head = self.model.get_classifier() if hasattr(self.model, "get_classifier") else None
        return list(head.parameters()) if head is not None else []

    def _backbone_params(self):
        head_ids = {id(p) for p in self._head_params()}
        return [p for p in self.model.parameters() if id(p) not in head_ids]

    def configure_optimizers(self):
        lr, scale = self.hparams.lr, self.hparams.backbone_lr_scale
        head_ids = {id(p) for p in self._head_params()}
        # Discriminative fine-tuning: the fresh head trains at `lr`; the pretrained
        # backbone at `lr * backbone_lr_scale` (set <1 for foundation models on small
        # datasets to adapt features gently and curb overfitting). Frozen params drop out.
        head = [p for p in self.model.parameters() if id(p) in head_ids and p.requires_grad]
        backbone = [p for p in self.model.parameters() if id(p) not in head_ids and p.requires_grad]
        groups = [{"params": head, "lr": lr}]
        if backbone:
            groups.append({"params": backbone, "lr": lr * scale})
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
