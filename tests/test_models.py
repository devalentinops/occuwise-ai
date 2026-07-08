"""Smoke tests: every architecture builds and produces the right output shape.

These run on CPU with tiny inputs and no pretrained download (pretrained=False),
so they are fast and offline-safe.
"""

import pytest
import torch

from occuwise.models import ModelConfig, build_model

CLS_ARCHS = ["resnet18", "resnet50", "efficientnet_b0", "densenet121", "vit_base"]


@pytest.mark.parametrize("arch", CLS_ARCHS)
def test_classifier_forward(arch):
    size = 224 if "vit" in arch else 128
    model = build_model(ModelConfig(task="classification", arch=arch,
                                    num_classes=5, pretrained=False)).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 5)


def test_gradcam_overlay():
    pytest.importorskip("pytorch_grad_cam")
    import numpy as np

    from occuwise.explain import gradcam_overlay

    model = build_model(ModelConfig(task="classification", arch="resnet18",
                                    num_classes=5, pretrained=False)).eval()
    base = np.random.rand(128, 128, 3).astype("float32")
    overlay = gradcam_overlay(model, torch.randn(1, 3, 128, 128), 2, base, "resnet18")
    assert overlay is not None and overlay.shape == (128, 128, 3)
    assert overlay.dtype == np.uint8


@pytest.mark.parametrize("decoder", ["unet", "unetpp", "fpn"])
def test_segmenter_forward(decoder):
    smp = pytest.importorskip("segmentation_models_pytorch")  # noqa: F841
    model = build_model(ModelConfig(task="segmentation", arch="resnet34",
                                    num_classes=3, pretrained=False, decoder=decoder)).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 3, 128, 128)
