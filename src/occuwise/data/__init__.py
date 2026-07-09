from .datamodule import OphthalmologyDataModule
from .registry import DATASETS, DatasetSpec, get_spec

__all__ = ["OphthalmologyDataModule", "DATASETS", "DatasetSpec", "get_spec"]
