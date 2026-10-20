"""Typed configuration for the CodeBrain fine-tuning data pipeline.

All paths and tunables are read from the two YAML files under
``codebrain_finetune/config/`` (``paths.yaml`` and ``dataloader.yaml``) — the
single sources of truth — and exposed as frozen, type-checked dataclasses so
downstream code never touches raw dicts or hardcoded paths.

The window contract (18 channels x 10 patches x 200 samples, x/100 normalisation)
was verified end-to-end by the Stage-2 forward-pass gate; treat it as load-bearing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = [
    "Paths",
    "WindowSpec",
    "CohortSpec",
    "OverlapSpec",
    "LoaderSpec",
    "DataConfig",
    "load_paths",
    "load_data_config",
    "run_root",
]


def run_root() -> Path:
    """Absolute path to ``codebrain_finetune/`` (this package lives at src/livinglab_codebrain)."""
    # config.py -> livinglab_codebrain -> src -> codebrain_finetune
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Resolved absolute filesystem locations for the run."""

    project_root: Path
    codebrain_repo: Path
    checkpoint: Path
    manifest: Path
    cache_dir: Path


@dataclass(frozen=True)
class WindowSpec:
    """The (verified) per-window tensor contract."""

    n_channels: int
    n_patches: int
    samples_per_patch: int
    norm_divisor: float
    expected_sfreq: int

    @property
    def flat_samples(self) -> int:
        """Length of the stored (unreshaped) time axis, e.g. 10 * 200 = 2000."""
        return self.n_patches * self.samples_per_patch

    @property
    def raw_shape(self) -> tuple[int, int]:
        """Shape of ``X`` as stored on disk, e.g. (18, 2000)."""
        return (self.n_channels, self.flat_samples)

    @property
    def model_shape(self) -> tuple[int, int, int]:
        """Shape CodeBrain's SSSM expects per window, e.g. (18, 10, 200)."""
        return (self.n_channels, self.n_patches, self.samples_per_patch)


@dataclass(frozen=True)
class CohortSpec:
    """Which patients are dropped, and which are still under review."""

    exclude_patients: tuple[str, ...]
    pending_review: tuple[str, ...]


@dataclass(frozen=True)
class OverlapSpec:
    eval_nonoverlap_only: bool


@dataclass(frozen=True)
class LoaderSpec:
    batch_size: int
    num_workers: int
    pin_memory: bool
    seed: int
    min_batch_size: int


@dataclass(frozen=True)
class DataConfig:
    """The complete, resolved data-pipeline configuration."""

    paths: Paths
    window: WindowSpec
    cohort: CohortSpec
    overlap: OverlapSpec
    loader: LoaderSpec


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text())


def load_paths(root: Path | None = None) -> Paths:
    """Resolve absolute paths from ``config/paths.yaml``."""
    root = root or run_root()
    project_root = root.parent
    cfg = _read_yaml(root / "config" / "paths.yaml")
    return Paths(
        project_root=project_root,
        codebrain_repo=project_root / cfg["run"]["codebrain"],
        checkpoint=root / "checkpoints" / "CodeBrain.pth",
        manifest=project_root / cfg["stage0"]["manifest"],
        cache_dir=project_root / cfg["stage0"]["cache_dir"],
    )


def load_data_config(root: Path | None = None) -> DataConfig:
    """Load and validate the full data configuration from both YAML files."""
    root = root or run_root()
    dl = _read_yaml(root / "config" / "dataloader.yaml")

    window = WindowSpec(
        n_channels=int(dl["window"]["n_channels"]),
        n_patches=int(dl["window"]["n_patches"]),
        samples_per_patch=int(dl["window"]["samples_per_patch"]),
        norm_divisor=float(dl["window"]["norm_divisor"]),
        expected_sfreq=int(dl["window"]["expected_sfreq"]),
    )
    cohort = CohortSpec(
        exclude_patients=tuple(dl["cohort"].get("exclude_patients") or ()),
        pending_review=tuple(dl["cohort"].get("pending_review") or ()),
    )
    overlap = OverlapSpec(
        eval_nonoverlap_only=bool(dl["overlap"]["eval_nonoverlap_only"]),
    )
    loader = LoaderSpec(
        batch_size=int(dl["loader"]["batch_size"]),
        num_workers=int(dl["loader"]["num_workers"]),
        pin_memory=bool(dl["loader"]["pin_memory"]),
        seed=int(dl["loader"]["seed"]),
        min_batch_size=int(dl["loader"]["min_batch_size"]),
    )

    if loader.min_batch_size < 2:
        raise ValueError(
            "loader.min_batch_size must be >= 2: CodeBrain's SSSM.forward() "
            "collapses a size-1 batch dimension via squeeze()."
        )

    return DataConfig(
        paths=load_paths(root),
        window=window,
        cohort=cohort,
        overlap=overlap,
        loader=loader,
    )
