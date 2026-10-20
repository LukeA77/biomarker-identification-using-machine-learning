"""Config system for the QC audit (config/qc.yaml).

Mirrors config.py's conventions (validated, frozen pydantic model, unknown
keys rejected) but is deliberately a SEPARATE, smaller config from the
pipeline's: this audit reads pipeline.yaml too (for channel list, reref
scheme, filter/units settings) but every QC-specific threshold, path, and
subject list lives here, never as a literal in qc_*.py.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import PROJECT_ROOT


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QcSource(_Strict):
    drive_root: str
    processed_dir_name: str


class QcOutput(_Strict):
    dir: str


class RepresentativeSessions(_Strict):
    patients: list[str]


class ImpedanceSession(_Strict):
    patient: str
    condition: str


class ImpedanceInventorySample(_Strict):
    sessions: list[ImpedanceSession]


class QcPsd(_Strict):
    welch_segment_s: float = Field(gt=0)
    welch_overlap_frac: float = Field(ge=0, lt=1)
    flat_low_band_hz: list[float]
    flat_high_band_hz: list[float]
    flat_ratio_db_threshold: float

    @model_validator(mode="after")
    def _check_bands(self) -> "QcPsd":
        for name, band in (("flat_low_band_hz", self.flat_low_band_hz),
                           ("flat_high_band_hz", self.flat_high_band_hz)):
            if len(band) != 2 or band[0] >= band[1]:
                raise ValueError(f"psd.{name} must be [lo, hi] with lo < hi, got {band}")
        return self


class QcAperiodic(_Strict):
    fit_freq_range_hz: list[float]
    peak_width_limits_hz: list[float]
    max_n_peaks: int = Field(gt=0)
    min_peak_height: float = Field(gt=0)
    peak_threshold: float = Field(gt=0)
    beta_band_hz: list[float]
    exponent_near_zero_threshold: float = Field(gt=0)
    r2_poor_threshold: float = Field(gt=0, le=1)
    flat_candidate_exponent_threshold: float = Field(gt=0)
    flat_candidate_min_fraction: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _check_ranges(self) -> "QcAperiodic":
        for name, band in (("fit_freq_range_hz", self.fit_freq_range_hz),
                           ("peak_width_limits_hz", self.peak_width_limits_hz),
                           ("beta_band_hz", self.beta_band_hz)):
            if len(band) != 2 or band[0] >= band[1]:
                raise ValueError(f"aperiodic.{name} must be [lo, hi] with lo < hi, got {band}")
        return self


class QcPlots(_Strict):
    dpi: int = Field(gt=0)


class QcConfig(_Strict):
    source: QcSource
    output: QcOutput
    representative_sessions: RepresentativeSessions
    impedance_inventory_sample: ImpedanceInventorySample
    psd: QcPsd
    aperiodic: QcAperiodic
    plots: QcPlots

    @property
    def drive_root(self) -> Path:
        return Path(self.source.drive_root)

    @property
    def output_dir(self) -> Path:
        p = Path(self.output.dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_qc_config(path: str | Path | None = None) -> QcConfig:
    """Load and validate the QC audit config.

    Parameters
    ----------
    path : str or Path, optional
        Path to the QC yaml file. Defaults to ``config/qc.yaml`` under the
        project root.

    Returns
    -------
    QcConfig
    """
    cfg_path = Path(path) if path else (PROJECT_ROOT / "config" / "qc.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"QC config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return QcConfig(**raw)
