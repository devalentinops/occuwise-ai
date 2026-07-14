"""Training entrypoint (Hydra-driven).

Examples
--------
Single run:
    py -m occuwise.train model=resnet50 data=aptos
Named experiment:
    py -m occuwise.train +experiment=dr_baseline
Sweep (architecture x dataset), one process per combination:
    py -m occuwise.train -m model=resnet50,efficientnet_b4,vit_base data=aptos,eyepacs

Every run is logged to MLflow (./mlruns) and the best checkpoint is saved under
outputs/<dataset>/<arch>/.
"""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from .data import OphthalmologyDataModule, get_spec
from .engine import build_lit_module

# DDP keeps AccumulateGrad nodes alive across iterations, which trips a benign
# stream-mismatch warning on PyTorch >= 2.8 (multi-GPU only). Correctness is
# unaffected and the mismatch is outside our control (Lightning DDP init).
if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


def _monitor_for(metric: str) -> tuple[str, str]:
    """Map a dataset's primary metric to the (monitored key, mode)."""
    # Higher-is-better for every metric we track.
    return f"val/{metric}", "max"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    pl.seed_everything(cfg.seed, workers=True)
    spec = get_spec(cfg.data.dataset)
    print(OmegaConf.to_yaml(cfg))

    dm = OphthalmologyDataModule(
        dataset=cfg.data.dataset,
        data_root=cfg.data.data_root,
        manifest=cfg.data.get("manifest"),
        image_size=cfg.data.get("image_size"),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        balance_classes=cfg.train.get("balance_classes", True),
    )

    common = dict(
        arch=cfg.model.arch,
        num_classes=spec.num_classes,
        pretrained=cfg.model.pretrained,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        max_epochs=cfg.train.max_epochs,
    )
    if spec.task == "classification":
        common.update(
            task_modality=spec.modality,
            loss=cfg.train.get("loss", "ce"),
            # Domain-pretrained backbone weights (e.g. RETFound), optional.
            weights_repo=cfg.model.get("weights_repo"),
            weights_file=cfg.model.get("weights_file"),
            weights_key=cfg.model.get("weights_key"),
            # Discriminative fine-tuning knobs (model config wins, else train default).
            backbone_lr_scale=cfg.model.get("backbone_lr_scale",
                                            cfg.train.get("backbone_lr_scale", 1.0)),
            freeze_backbone=cfg.model.get("freeze_backbone",
                                          cfg.train.get("freeze_backbone", False)),
        )
    else:
        common.update(decoder=cfg.model.get("decoder", "unet"))

    model = build_lit_module(spec.task, **common)

    monitor, mode = _monitor_for(spec.primary_metric)
    run_name = f"{spec.name}__{cfg.model.arch}"
    ckpt_cb = ModelCheckpoint(
        dirpath=f"outputs/{spec.name}/{cfg.model.arch}",
        filename="best-{epoch:02d}-{" + monitor.replace("/", "_") + ":.4f}",
        monitor=monitor, mode=mode, save_top_k=1, save_last=True, auto_insert_metric_name=False,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor=monitor, mode=mode, patience=cfg.train.patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    loggers = [
        MLFlowLogger(experiment_name=f"occuwise/{spec.task}", run_name=run_name,
                     tracking_uri=cfg.tracking_uri),
        CSVLogger("outputs", name=run_name),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        # Stop gracefully before a session wall-clock limit (e.g. Kaggle's 12h cap),
        # leaving last.ckpt/best.ckpt intact so the run can be resumed next session.
        max_time=cfg.train.get("max_time"),
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        deterministic=False,
        gradient_clip_val=cfg.train.get("gradient_clip_val", 1.0),
        check_val_every_n_epoch=cfg.train.get("check_val_every_n_epoch", 1),
        limit_val_batches=cfg.train.get("limit_val_batches", 1.0),
    )
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.train.get("resume"))
    test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
    print("TEST:", test_results)

    # Return the monitored metric so Hydra sweepers (e.g. Optuna) can optimise it.
    return float(trainer.callback_metrics.get(monitor, 0.0))


if __name__ == "__main__":
    main()
