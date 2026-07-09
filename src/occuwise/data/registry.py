"""Dataset registry.

Central catalog of every dataset we benchmark against. Each entry describes the
*task* it supports and the metadata the training/eval code needs. Raw downloads
are heterogeneous, so every dataset is normalised by a `scripts/prepare_*.py`
script into a standard manifest CSV before training:

    classification manifest columns:  image_path, label, split
    segmentation   manifest columns:  image_path, mask_path, split

`split` is one of {train, val, test}. Paths are relative to the dataset root.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str                      # "classification" | "segmentation"
    modality: str                  # "fundus" | "oct"
    num_classes: int
    class_names: list[str]
    # Primary metric used to rank models on this dataset's leaderboard.
    primary_metric: str
    # Sensible default input resolution for this modality/dataset.
    image_size: int = 512
    description: str = ""
    # For segmentation: whether masks are multi-class (layers) or binary.
    seg_multiclass: bool = False


# fmt: off
DATASETS: dict[str, DatasetSpec] = {
    # ---------------- Diabetic Retinopathy grading (fundus, 5-class 0..4) ---------------
    "eyepacs": DatasetSpec(
        name="eyepacs", task="classification", modality="fundus", num_classes=5,
        class_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        primary_metric="quadratic_kappa", image_size=512,
        description="Kaggle EyePACS Diabetic Retinopathy Detection (~88k fundus images).",
    ),
    "aptos": DatasetSpec(
        name="aptos", task="classification", modality="fundus", num_classes=5,
        class_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        primary_metric="quadratic_kappa", image_size=512,
        description="APTOS 2019 Blindness Detection (~3.6k fundus images).",
    ),
    "messidor": DatasetSpec(
        name="messidor", task="classification", modality="fundus", num_classes=4,
        class_names=["R0", "R1", "R2", "R3"],
        primary_metric="quadratic_kappa", image_size=512,
        description="Messidor / Messidor-2 DR grading (retinopathy grade 0..3).",
    ),
    # ---------------------------- OCT disease classification --------------------------
    "oct2017": DatasetSpec(
        name="oct2017", task="classification", modality="oct", num_classes=4,
        class_names=["CNV", "DME", "DRUSEN", "NORMAL"],
        primary_metric="macro_f1", image_size=224,
        description="Kermany OCT2017 (~84k OCT B-scans, 4 classes).",
    ),
    # ---------------------------- Glaucoma (fundus) -----------------------------------
    "refuge_cls": DatasetSpec(
        name="refuge_cls", task="classification", modality="fundus", num_classes=2,
        class_names=["Non-Glaucoma", "Glaucoma"],
        primary_metric="auroc", image_size=512,
        description="REFUGE glaucoma classification label.",
    ),
    "refuge_seg": DatasetSpec(
        name="refuge_seg", task="segmentation", modality="fundus", num_classes=3,
        class_names=["background", "optic_disc", "optic_cup"],
        primary_metric="dice", image_size=512, seg_multiclass=True,
        description="REFUGE optic disc & cup segmentation (for cup-to-disc ratio).",
    ),
    # ---------------------------- OCT layer segmentation ------------------------------
    "duke_oct": DatasetSpec(
        name="duke_oct", task="segmentation", modality="oct", num_classes=2,
        class_names=["background", "fluid"],
        primary_metric="dice", image_size=512, seg_multiclass=False,
        description="Duke OCT (DME) — retinal fluid / layer segmentation.",
    ),
}
# fmt: on


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(DATASETS)}")
    return DATASETS[name]
