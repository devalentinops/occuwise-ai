"""Evaluate a trained checkpoint on a dataset's test split.

    py -m occuwise.evaluate ckpt=outputs/aptos/resnet50/best-....ckpt data=aptos
"""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from .data import OphthalmologyDataModule, get_spec
from .engine import load_trained_module


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    spec = get_spec(cfg.data.dataset)
    dm = OphthalmologyDataModule(
        dataset=cfg.data.dataset,
        data_root=cfg.data.data_root,
        manifest=cfg.data.get("manifest"),
        image_size=cfg.data.get("image_size"),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )
    # Loads the trained weights without re-fetching (possibly gated) pretrained backbones.
    model = load_trained_module(cfg.ckpt, spec.task)
    trainer = pl.Trainer(accelerator=cfg.train.accelerator, devices=1,
                         precision=cfg.train.precision, logger=False)
    results = trainer.test(model, datamodule=dm)
    print(results)
    return results


if __name__ == "__main__":
    main()
