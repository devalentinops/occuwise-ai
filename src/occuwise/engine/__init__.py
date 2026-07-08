from .lit_classifier import LitClassifier
from .lit_segmenter import LitSegmenter


def build_lit_module(task: str, **kwargs):
    """Return the LightningModule appropriate for the task."""
    if task == "classification":
        return LitClassifier(**kwargs)
    if task == "segmentation":
        return LitSegmenter(**kwargs)
    raise ValueError(f"Unknown task: {task}")


__all__ = ["LitClassifier", "LitSegmenter", "build_lit_module"]
