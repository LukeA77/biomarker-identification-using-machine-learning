"""Manifest handling and the per-window ``Dataset`` for CodeBrain fine-tuning.

Two pieces:

* :class:`CohortManifest` — loads ``manifest.parquet``, validates the label map
  and window contract, applies the configured patient exclusions *loudly*, and
  exposes clean subsetting (by patient, by overlap flag). It is the single source
  of truth for "which windows are in play".
* :class:`WindowDataset` — a ``torch.utils.data.Dataset`` over a manifest slice
  that loads each ``.pkl``, reshapes ``(18, 2000) -> (18, 10, 200)``, normalises
  by ``/100``, and returns ``(x, y, info)`` where ``info`` carries the metadata
  (``patient_id`` etc.) needed for patient-level LOPO aggregation downstream.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import CohortSpec, DataConfig, WindowSpec

log = logging.getLogger(__name__)

__all__ = ["CohortManifest", "WindowDataset", "ManifestValidationError", "LABEL_MAP"]

# Verified label map: condition A = ON = 0, condition C = OFF = 1.
LABEL_MAP: dict[str, int] = {"A": 0, "C": 1}

# Columns the pipeline relies on; the manifest may carry more (harmless).
_REQUIRED_COLUMNS = {
    "path",
    "patient_id",
    "condition",
    "y_cond",
    "t_cont",
    "is_nonoverlap_subset",
    "stride_s",
    "sfreq",
}


class ManifestValidationError(RuntimeError):
    """Raised when the manifest violates an assumption the pipeline depends on."""


class CohortManifest:
    """The validated, exclusion-filtered set of windows for the study.

    Use :meth:`from_config` to build one. ``frame`` is the working (post-exclusion)
    manifest; ``full_frame`` is the untouched original for provenance/reporting.
    """

    def __init__(self, full_frame: pd.DataFrame, frame: pd.DataFrame, config: DataConfig) -> None:
        self.full_frame = full_frame
        self.frame = frame
        self.config = config

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, config: DataConfig) -> "CohortManifest":
        path = config.paths.manifest
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")
        full = pd.read_parquet(path)
        cls._validate(full, config.window)
        frame = cls._apply_exclusions(full, config.cohort)
        return cls(full, frame, config)

    # -------------------------------------------------------------- validation
    @staticmethod
    def _validate(df: pd.DataFrame, window: WindowSpec) -> None:
        """Fail loudly if the manifest breaks a load-bearing assumption."""
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ManifestValidationError(f"Manifest missing required columns: {sorted(missing)}")

        bad_cond = set(df["condition"].unique()) - set(LABEL_MAP)
        if bad_cond:
            raise ManifestValidationError(
                f"Unexpected condition values {sorted(bad_cond)}; expected {sorted(LABEL_MAP)}"
            )

        expected_y = df["condition"].map(LABEL_MAP)
        mismatched = int((df["y_cond"] != expected_y).sum())
        if mismatched:
            raise ManifestValidationError(
                f"{mismatched} rows where y_cond disagrees with the A=0/C=1 label map."
            )

        bad_sfreq = sorted(set(df["sfreq"].unique()) - {window.expected_sfreq})
        if bad_sfreq:
            raise ManifestValidationError(
                f"Non-uniform sfreq {bad_sfreq}; expected only {window.expected_sfreq} Hz."
            )
        log.info(
            "Manifest validated: %d windows, %d patients, sfreq=%d, label map A=0/C=1 consistent.",
            len(df), df["patient_id"].nunique(), window.expected_sfreq,
        )

    # --------------------------------------------------------------- exclusions
    @staticmethod
    def _apply_exclusions(df: pd.DataFrame, cohort: CohortSpec) -> pd.DataFrame:
        """Drop excluded patients, logging exactly what goes and what remains."""
        present = set(df["patient_id"].unique())
        exclude = list(cohort.exclude_patients)

        not_found = [p for p in exclude if p not in present]
        if not_found:
            # A configured exclusion that isn't in the manifest is suspicious — surface it.
            log.warning("Exclusion list names patients absent from the manifest: %s", not_found)

        if exclude:
            counts = df[df["patient_id"].isin(exclude)]["patient_id"].value_counts().to_dict()
            log.info("Applying cohort exclusions (loud):")
            for p in exclude:
                log.info("    EXCLUDE %-9s -> %5d windows dropped", p, counts.get(p, 0))

        kept = df[~df["patient_id"].isin(exclude)].reset_index(drop=True)

        still_pending = [p for p in cohort.pending_review if p in set(kept["patient_id"].unique())]
        if still_pending:
            log.warning(
                "PENDING REVIEW patients still IN the cohort (contributing windows): %s. "
                "Confirm with supervisor before locking folds.", still_pending,
            )

        log.info(
            "Cohort after exclusions: %d patients (was %d), %d windows (was %d).",
            kept["patient_id"].nunique(), df["patient_id"].nunique(), len(kept), len(df),
        )
        return kept

    # ---------------------------------------------------------------- accessors
    @property
    def patients(self) -> list[str]:
        """Sorted, deterministic list of patients remaining in the cohort."""
        return sorted(self.frame["patient_id"].unique().tolist())

    @property
    def n_windows(self) -> int:
        return len(self.frame)

    def label_balance(self) -> dict[str, int]:
        """Window counts per condition (A/ON, C/OFF) over the working cohort."""
        return self.frame["condition"].value_counts().to_dict()

    def patient_window_counts(self, nonoverlap_only: bool = False) -> pd.Series:
        """Windows per patient (optionally restricted to the non-overlap subset)."""
        frame = self.frame[self.frame["is_nonoverlap_subset"]] if nonoverlap_only else self.frame
        return frame.groupby("patient_id").size().sort_values()

    def subset(
        self,
        patients: list[str] | None = None,
        *,
        nonoverlap_only: bool = False,
    ) -> pd.DataFrame:
        """Return a copy of the manifest filtered to ``patients`` and/or the non-overlap subset."""
        frame = self.frame
        if patients is not None:
            frame = frame[frame["patient_id"].isin(patients)]
        if nonoverlap_only:
            frame = frame[frame["is_nonoverlap_subset"]]
        return frame.reset_index(drop=True)


class WindowDataset(Dataset):
    """A ``Dataset`` over a manifest slice.

    Each item is ``(x, y, info)``:

    * ``x``    — ``float32`` tensor, shape ``(n_channels, n_patches, samples_per_patch)``,
      already divided by ``norm_divisor``;
    * ``y``    — ``int`` label (0=ON/A, 1=OFF/C);
    * ``info`` — dict with ``patient_id``, ``condition``, ``t_cont``, ``path``,
      ``is_nonoverlap_subset`` for patient-level aggregation and audit.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        cache_dir: Path,
        window: WindowSpec,
        *,
        check_meta: bool = False,
    ) -> None:
        if len(frame) == 0:
            raise ValueError("WindowDataset built on an empty manifest slice.")
        self._rows = frame.reset_index(drop=True)
        self._cache_dir = Path(cache_dir)
        self._window = window
        self._check_meta = check_meta

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        row = self._rows.iloc[idx]
        file_path = self._cache_dir / row["path"]
        with open(file_path, "rb") as fh:
            obj = pickle.load(fh)

        x = obj["X"]
        if x.shape != self._window.raw_shape:
            raise ValueError(
                f"{file_path}: expected raw shape {self._window.raw_shape}, got {x.shape}."
            )

        x = x.reshape(self._window.model_shape).astype(np.float32) / self._window.norm_divisor
        y = int(row["y_cond"])

        if self._check_meta:
            meta = obj.get("meta", {})
            if meta.get("patient_id") != row["patient_id"] or meta.get("condition") != row["condition"]:
                raise ValueError(
                    f"{file_path}: meta ({meta.get('patient_id')},{meta.get('condition')}) "
                    f"disagrees with manifest ({row['patient_id']},{row['condition']})."
                )

        info = {
            "patient_id": row["patient_id"],
            "condition": row["condition"],
            "t_cont": float(row["t_cont"]),
            "path": row["path"],
            "is_nonoverlap_subset": bool(row["is_nonoverlap_subset"]),
        }
        return torch.from_numpy(x), y, info
