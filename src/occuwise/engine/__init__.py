from __future__ import annotations

import warnings

import torch

from .lit_classifier import LitClassifier
from .lit_segmenter import LitSegmenter


def build_lit_module(task: str, **kwargs):
    """Return the LightningModule appropriate for the task."""
    if task == "classification":
        return LitClassifier(**kwargs)
    if task == "segmentation":
        return LitSegmenter(**kwargs)
    raise ValueError(f"Unknown task: {task}")


def load_trained_module(ckpt_path: str, task: str, map_location: str = "cpu"):
    """Load a trained checkpoint for inference/export WITHOUT re-fetching backbone weights.

    `LightningModule.load_from_checkpoint` rebuilds the model from its saved
    hyperparameters, which re-runs `build_model` and re-downloads any ImageNet or
    domain-pretrained backbone weights (e.g. RETFound, which is *gated* on the HF Hub).
    That download is both wasteful and auth-gated, because the checkpoint's own
    state_dict overwrites those weights immediately afterwards. Here we reconstruct the
    module with every pretrained-weight fetch disabled, then load the trained state_dict
    on top — no network access required to evaluate or export a trained model.

    Returns the module in eval() mode. Warns loudly (never silently loads nothing) if the
    state_dict doesn't populate the model as expected.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    # Disable every pretrained-weight fetch; the state_dict below supplies the real weights.
    hp["pretrained"] = False
    for key in ("weights_repo", "weights_file"):
        if key in hp:  # only classifiers carry these; segmenter __init__ has no such arg
            hp[key] = None

    module = build_lit_module(task, **hp)
    result = module.load_state_dict(ckpt["state_dict"], strict=False)
    matched = len(module.state_dict()) - len(result.missing_keys)
    if matched == 0 or result.unexpected_keys:
        warnings.warn(
            f"load_trained_module({ckpt_path}): matched {matched} tensors "
            f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}). "
            f"Unexpected (first 5): {result.unexpected_keys[:5]}. "
            f"A near-zero match means the checkpoint doesn't fit this architecture.",
            stacklevel=2,
        )
    module.eval()
    return module


__all__ = ["LitClassifier", "LitSegmenter", "build_lit_module", "load_trained_module"]
