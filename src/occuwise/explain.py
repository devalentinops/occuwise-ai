"""Grad-CAM saliency for classification models.

Produces a heatmap over the input image showing which regions most influenced the
predicted class — red = strongly influential, blue = ignored. This is essential for
clinical trust: it lets a reviewer check the model is looking at real pathology
(haemorrhages, lesions) rather than an artefact (border, glare).

Works for the CNN backbones (ResNet / EfficientNet / DenseNet) by hooking the last
convolutional layer, and for ViT via a token-to-grid reshape. If anything about a
given architecture doesn't cooperate, `gradcam_overlay` returns None rather than
breaking prediction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _last_conv(model: nn.Module) -> nn.Module | None:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def _vit_target(model: nn.Module):
    """Target layer + reshape transform for a timm ViT, or (None, None)."""
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        return None, None
    target = blocks[-1].norm1

    def reshape_transform(tensor):
        # tensor: [B, tokens, dim]; drop the leading class token, fold to a grid.
        n = tensor.shape[1] - 1
        side = int(round(n ** 0.5))
        grid = tensor[:, 1:1 + side * side, :].reshape(tensor.shape[0], side, side, -1)
        return grid.permute(0, 3, 1, 2)

    return target, reshape_transform


def gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    base_rgb01: np.ndarray,
    arch: str,
    image_weight: float = 0.5,
) -> np.ndarray | None:
    """Return an HxWx3 uint8 RGB overlay for `class_idx`, or None if unsupported.

    `base_rgb01` is the display image (same H,W as the model input) in [0,1] float.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        if "vit" in arch:
            target_layer, reshape = _vit_target(model)
        else:
            target_layer, reshape = _last_conv(model), None
        if target_layer is None:
            return None

        # Grad-CAM needs gradients, so run outside any no_grad context.
        with torch.enable_grad():
            with GradCAM(model=model, target_layers=[target_layer],
                         reshape_transform=reshape) as cam:
                grayscale = cam(input_tensor=input_tensor,
                                targets=[ClassifierOutputTarget(class_idx)])[0]

        base = np.ascontiguousarray(base_rgb01.astype(np.float32))
        return show_cam_on_image(base, grayscale, use_rgb=True, image_weight=image_weight)
    except Exception:  # noqa: BLE001 — explainability must never break prediction
        return None
