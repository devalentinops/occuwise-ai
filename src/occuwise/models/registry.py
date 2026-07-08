"""Model factory.

One entry point, `build_model(cfg)`, produces either:
  * a classification backbone (timm) — ResNet / EfficientNet / DenseNet / ViT, or
  * a U-Net segmentation model (segmentation-models-pytorch) with a pretrained encoder.

`cfg.arch` is a short alias resolved here to the concrete library model name. This
keeps experiment configs terse (`model=resnet50`) while allowing any timm/smp model.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except Exception:  # pragma: no cover - smp optional at import time
    smp = None


# Alias -> timm model name for classification backbones.
# Any timm name works directly too; these are just the benchmarked defaults.
CLASSIFICATION_ARCHS: dict[str, str] = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b4": "efficientnet_b4",
    "densenet121": "densenet121",
    "densenet201": "densenet201",
    "vit_base": "vit_base_patch16_224",
    "vit_small": "vit_small_patch16_224",
    # Load HF-hub weights directly, e.g. microsoft/resnet-50 style, via timm:
    #   "resnet50_hf": "hf_hub:timm/resnet50.a1_in1k"
}

# Alias -> smp encoder name for U-Net segmentation.
SEGMENTATION_ENCODERS: dict[str, str] = {
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "efficientnet_b4": "efficientnet-b4",
    "densenet121": "densenet121",
    # ViT/transformer encoders are available via smp's `mit_b*` (SegFormer) too.
    "mit_b2": "mit_b2",
}


@dataclass
class ModelConfig:
    task: str                       # "classification" | "segmentation"
    arch: str                       # alias from the maps above (or raw library name)
    num_classes: int
    pretrained: bool = True
    in_channels: int = 3
    drop_rate: float = 0.0          # classification head dropout
    # segmentation-only:
    decoder: str = "unet"           # "unet" | "unetpp" | "fpn" | "segformer"


def build_classifier(cfg: ModelConfig) -> nn.Module:
    name = CLASSIFICATION_ARCHS.get(cfg.arch, cfg.arch)
    model = timm.create_model(
        name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        in_chans=cfg.in_channels,
        drop_rate=cfg.drop_rate,
    )
    return model


def build_segmenter(cfg: ModelConfig) -> nn.Module:
    if smp is None:
        raise ImportError("segmentation-models-pytorch is required for segmentation tasks.")
    encoder = SEGMENTATION_ENCODERS.get(cfg.arch, cfg.arch)
    weights = "imagenet" if cfg.pretrained else None
    factory = {
        "unet": smp.Unet,
        "unetpp": smp.UnetPlusPlus,
        "fpn": smp.FPN,
        "segformer": smp.Segformer,
    }[cfg.decoder]
    return factory(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=cfg.in_channels,
        classes=cfg.num_classes,
    )


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.task == "classification":
        return build_classifier(cfg)
    if cfg.task == "segmentation":
        return build_segmenter(cfg)
    raise ValueError(f"Unknown task: {cfg.task}")
