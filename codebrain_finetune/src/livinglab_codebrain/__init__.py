"""livinglab_codebrain — data pipeline for CodeBrain ON/OFF levodopa fine-tuning.

Public API for building cohort-correct, model-ready data loaders over the
Stage-0 window cache. See ``codebrain_finetune/config/*.yaml`` for configuration.
"""
from __future__ import annotations

from .config import (
    CohortSpec,
    DataConfig,
    LoaderSpec,
    OverlapSpec,
    Paths,
    WindowSpec,
    load_data_config,
    load_paths,
    run_root,
)
from .dataset import LABEL_MAP, CohortManifest, ManifestValidationError, WindowDataset
from .loader import (
    LopoFold,
    NoSingletonBatchSampler,
    build_lopo_folds,
    collate_windows,
    make_dataloader,
    make_fold_loaders,
)

__all__ = [
    # config
    "DataConfig", "Paths", "WindowSpec", "CohortSpec", "OverlapSpec", "LoaderSpec",
    "load_data_config", "load_paths", "run_root",
    # dataset
    "CohortManifest", "WindowDataset", "ManifestValidationError", "LABEL_MAP",
    # loader
    "make_dataloader", "make_fold_loaders", "build_lopo_folds", "LopoFold",
    "collate_windows", "NoSingletonBatchSampler",
]
