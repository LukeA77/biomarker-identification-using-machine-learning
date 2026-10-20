"""codebrain_ft — CodeBrain LoRA fine-tuning for ON/OFF levodopa classification.

Builds on the audited data loader package ``livinglab_codebrain`` (cohort,
folds, window loaders). This package adds the model (backbone + head + LoRA),
the training/eval harness, patient-level metrics, and run tracking for the
nested-LOPO experiments described in ``FINE_TUNING_PLAN.md`` §4-8.
"""
from __future__ import annotations

from .ft_config import FinetuneConfig, load_finetune_config
from .model import build_classifier_model, trainable_parameter_report, CodeBrainClassifier
from .data import (
    make_patient_loader, inner_group_split,
    NormalizedWindowDataset, INPUT_NORM_SCHEMES, cohort_median_scale,
    SessionBalancedBatchSampler, duration_matched_mask,
)
from .align import compute_ea_whiteners, load_or_compute_whiteners, EAWindowDataset
from .train import train_model, evaluate_session_level, predict_windows, TrainResult
from .probe import run_probe_fold, extract_features
from .seeding import seed_everything, fold_seed
from .control_features import (
    ControlsConfig, load_controls_config, FEATURE_BLOCKS, CONTROL_DESCRIPTIONS,
    load_or_compute_features, control_matrix, control_target,
)

__all__ = [
    "FinetuneConfig", "load_finetune_config",
    "build_classifier_model", "trainable_parameter_report", "CodeBrainClassifier",
    "make_patient_loader", "inner_group_split",
    "NormalizedWindowDataset", "INPUT_NORM_SCHEMES", "cohort_median_scale",
    "SessionBalancedBatchSampler", "duration_matched_mask",
    "compute_ea_whiteners", "load_or_compute_whiteners", "EAWindowDataset",
    "train_model", "evaluate_session_level", "predict_windows", "TrainResult",
    "run_probe_fold", "extract_features",
    "seed_everything", "fold_seed",
    "ControlsConfig", "load_controls_config", "FEATURE_BLOCKS", "CONTROL_DESCRIPTIONS",
    "load_or_compute_features", "control_matrix", "control_target",
]
