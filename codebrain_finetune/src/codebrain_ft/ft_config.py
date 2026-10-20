"""Typed configuration for the fine-tuning experiments (``config/finetune.yaml``).

Kept deliberately thin: a small tree of frozen dataclasses mirroring the YAML,
so downstream code reads ``cfg.lora.r`` rather than juggling raw dicts, and an
unknown/missing key fails loudly at load time instead of deep in a training run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from livinglab_codebrain.config import run_root

__all__ = ["FinetuneConfig", "load_finetune_config"]


def _require(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise KeyError(f"finetune.yaml: missing required key '{key}' under {ctx}")
    return d[key]


@dataclass(frozen=True)
class SeedCfg:
    base: int
    permutation_offset: int
    deterministic: bool


@dataclass(frozen=True)
class HeadCfg:
    type: str
    hidden1: int
    hidden2: int
    pool_hidden: int
    dropout: float


@dataclass(frozen=True)
class ModelCfg:
    backbone: dict  # verbatim SSSM kwargs (kept as dict; keys are CodeBrain's)
    head: HeadCfg


@dataclass(frozen=True)
class LoraCfg:
    target: str
    r: int
    alpha_ratio: int
    dropout: float
    bias: str

    @property
    def alpha(self) -> int:
        return self.alpha_ratio * self.r


@dataclass(frozen=True)
class EarlyStopCfg:
    monitor: str
    mode: str
    patience: int
    min_delta: float


@dataclass(frozen=True)
class TrainCfg:
    optimizer: str
    weight_decay: float
    lr: float
    lr_full_ft: float
    batch_size: int
    max_epochs: int
    warmup_frac: float
    precision: str
    grad_clip: float
    early_stopping: EarlyStopCfg


@dataclass(frozen=True)
class ProbeCfg:
    pooling: str
    classifier: str
    C: float
    max_iter: int
    standardize: bool


@dataclass(frozen=True)
class InnerSearchCfg:
    method: str
    n_configs: int
    space: dict


@dataclass(frozen=True)
class NestedLopoCfg:
    inner_folds: int
    inner_search: InnerSearchCfg


@dataclass(frozen=True)
class SamplerCfg:
    session_balanced: bool


@dataclass(frozen=True)
class BootstrapCfg:
    n_resamples: int
    ci: float


@dataclass(frozen=True)
class PermutationCfg:
    n_permutations: int


@dataclass(frozen=True)
class EvalCfg:
    patient_agg: str
    threshold: float
    primary_metric: str
    bootstrap: BootstrapCfg
    permutation: PermutationCfg


@dataclass(frozen=True)
class TrackingCfg:
    results_dir: Path
    master_csv: Path


@dataclass(frozen=True)
class FinetuneConfig:
    seed: SeedCfg
    model: ModelCfg
    lora: LoraCfg
    train: TrainCfg
    probe: ProbeCfg
    nested_lopo: NestedLopoCfg
    sampler: SamplerCfg
    eval: EvalCfg
    tracking: TrackingCfg
    raw: dict = field(repr=False, default_factory=dict)  # full dict for provenance dumps


def load_finetune_config(root: Path | None = None) -> FinetuneConfig:
    """Load and validate ``config/finetune.yaml`` into a typed tree."""
    root = root or run_root()
    path = root / "config" / "finetune.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"finetune config not found: {path}")
    d = yaml.safe_load(path.read_text())

    seed = SeedCfg(**_require(d, "seed", "root"))

    m = _require(d, "model", "root")
    head = HeadCfg(**_require(m, "head", "model"))
    if head.type not in {"pool_mlp", "flatten_mlp"}:
        raise ValueError(f"model.head.type must be pool_mlp or flatten_mlp, got {head.type!r}")
    model = ModelCfg(backbone=dict(_require(m, "backbone", "model")), head=head)

    lora = LoraCfg(**_require(d, "lora", "root"))
    if lora.target not in {"attention", "out_proj"}:
        raise ValueError(f"lora.target must be 'attention' or 'out_proj', got {lora.target!r}")

    t = _require(d, "train", "root")
    train = TrainCfg(
        optimizer=t["optimizer"], weight_decay=t["weight_decay"], lr=t["lr"],
        lr_full_ft=t["lr_full_ft"], batch_size=t["batch_size"], max_epochs=t["max_epochs"],
        warmup_frac=t["warmup_frac"], precision=t["precision"], grad_clip=t["grad_clip"],
        early_stopping=EarlyStopCfg(**t["early_stopping"]),
    )

    probe = ProbeCfg(**_require(d, "probe", "root"))
    if probe.pooling not in {"mean", "flatten"}:
        raise ValueError(f"probe.pooling must be 'mean' or 'flatten', got {probe.pooling!r}")

    nl = _require(d, "nested_lopo", "root")
    nested = NestedLopoCfg(
        inner_folds=nl["inner_folds"],
        inner_search=InnerSearchCfg(**nl["inner_search"]),
    )

    sampler = SamplerCfg(**_require(d, "sampler", "root"))

    e = _require(d, "eval", "root")
    ev = EvalCfg(
        patient_agg=e["patient_agg"], threshold=e["threshold"],
        primary_metric=e["primary_metric"],
        bootstrap=BootstrapCfg(**e["bootstrap"]),
        permutation=PermutationCfg(**e["permutation"]),
    )

    tr = _require(d, "tracking", "root")
    tracking = TrackingCfg(
        results_dir=root / tr["results_dir"],
        master_csv=root / tr["master_csv"],
    )

    return FinetuneConfig(
        seed=seed, model=model, lora=lora, train=train, probe=probe,
        nested_lopo=nested, sampler=sampler, eval=ev, tracking=tracking, raw=d,
    )
