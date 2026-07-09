"""Model factory.

One entry point, `build_model(cfg)`, produces either:
  * a classification backbone (timm) — ResNet / EfficientNet / DenseNet / ViT, or
  * a U-Net segmentation model (segmentation-models-pytorch) with a pretrained encoder.

`cfg.arch` is a short alias resolved here to the concrete library model name. This
keeps experiment configs terse (`model=resnet50`) while allowing any timm/smp model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import timm
import torch
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
    # Modern strong CNNs / literature baselines — all resolve straight from timm+HF.
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "vgg16": "vgg16",              # explainable-fundus literature baseline (weaker; for comparison)
    "xception": "legacy_xception",  # multi-ocular-disease Grad-CAM baseline
    # RETFound is a ViT-L/16 encoder; its retinal-pretrained weights load via the
    # `weights_repo`/`weights_file` mechanism below (see configs/model/retfound.yaml).
    "retfound": "vit_large_patch16_224",
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
    # Domain-pretrained backbone weights (e.g. RETFound). When set, the backbone is
    # created WITHOUT ImageNet weights and this checkpoint is loaded into it instead;
    # the classification head is always freshly initialised for the target task.
    weights_repo: str | None = None   # HF-hub repo id, e.g. "YukunZhou/RETFound_mae_natureCFP"
    weights_file: str | None = None   # filename within the repo, e.g. "RETFound_mae_natureCFP.pth"
    weights_key: str | None = None    # nested key holding the state_dict (e.g. "model"); auto if None


def _load_backbone_weights(model: nn.Module, cfg: ModelConfig) -> None:
    """Load a domain-pretrained encoder checkpoint into `model` (backbone only).

    Non-fatal and loud: mismatched/decoder/head keys are dropped, the class head is
    left freshly initialised, and the matched-parameter count is reported so a silent
    "loaded nothing → training from scratch" can never pass unnoticed.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=cfg.weights_repo, filename=cfg.weights_file)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Unwrap common nesting (RETFound/MAE store under "model"; others "state_dict"/"teacher").
    for key in ([cfg.weights_key] if cfg.weights_key else ["model", "state_dict", "teacher"]):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt.items()}

    # Drop the checkpoint's own head / MAE decoder — we train a fresh task head.
    model_keys = set(model.state_dict().keys())
    filtered = {
        k: v for k, v in state.items()
        if k in model_keys and model.state_dict()[k].shape == v.shape
        and not k.startswith(("head", "fc", "decoder", "mask_token"))
    }
    result = model.load_state_dict(filtered, strict=False)

    matched, total = len(filtered), len(model_keys)
    if matched == 0:
        warnings.warn(
            f"RETFound/custom weights from {cfg.weights_repo}:{cfg.weights_file} matched 0 of "
            f"{total} backbone tensors — check the repo id, filename, and key layout. "
            f"The backbone is currently RANDOM, not retinal-pretrained.",
            stacklevel=2,
        )
    else:
        print(f"[weights] loaded {matched}/{total} backbone tensors from "
              f"{cfg.weights_repo}:{cfg.weights_file} "
              f"(missing={len(result.missing_keys)} head/new params left fresh).")


def build_classifier(cfg: ModelConfig) -> nn.Module:
    name = CLASSIFICATION_ARCHS.get(cfg.arch, cfg.arch)
    custom = bool(cfg.weights_repo)
    model = timm.create_model(
        name,
        pretrained=cfg.pretrained and not custom,  # don't fetch ImageNet if loading domain weights
        num_classes=cfg.num_classes,
        in_chans=cfg.in_channels,
        drop_rate=cfg.drop_rate,
    )
    if custom:
        _load_backbone_weights(model, cfg)
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
