"""Augmentation / preprocessing pipelines (Albumentations).

Two families:
  * fundus  — retinal photographs. We apply a circular crop + CLAHE to normalise
              illumination, then geometric/photometric augmentation.
  * oct     — grayscale B-scans. Lighter photometric aug; no colour jitter.

ImageNet normalisation is used because all backbones are ImageNet-pretrained.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def circle_crop(image: np.ndarray) -> np.ndarray:
    """Crop a fundus image to its circular field-of-view and trim black borders."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > 7
    if mask.sum() == 0:
        return image
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


class FundusPreprocess(A.ImageOnlyTransform):
    """Circle-crop wrapper usable inside an Albumentations pipeline."""

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def apply(self, img, **params):
        return circle_crop(img)


def build_transforms(task: str, modality: str, image_size: int, train: bool) -> A.Compose:
    resize = [A.Resize(image_size, image_size)]
    normalize = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]

    pre: list = []
    if modality == "fundus":
        pre.append(FundusPreprocess())
        pre.append(A.CLAHE(clip_limit=2.0, p=1.0))

    if not train:
        return A.Compose(pre + resize + normalize, **_seg_kwargs(task))

    if modality == "fundus":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=180,
                               border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
        ]
    else:  # oct
        aug = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
                               border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        ]

    return A.Compose(pre + aug + resize + normalize, **_seg_kwargs(task))


def _seg_kwargs(task: str) -> dict:
    # For segmentation, masks must be transformed jointly with the image.
    if task == "segmentation":
        return {"additional_targets": {}}
    return {}
