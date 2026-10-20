"""Signal access for the QC audit: reconstructs a session's full preprocessed
continuous signal from the window files the production run already wrote to
disk, and loads the corresponding pre-preprocessing raw-CSV signal for the
before/after comparison.

Read-only against `processed_csv_continuous/` (or whatever `qc.source.
processed_dir_name` points at) and the shared drive. Preprocessing is NOT
re-run at cohort scale here -- the per-channel-per-session PSD/aperiodic
checks (Part A/B) read the SAME serialized windows that will go into
fine-tuning, by stitching the non-overlapping window subset back into one
continuous array. Only the small representative-sample before/after plots
load anything fresh, and only the pre-filter raw side (the processed side
reuses the same stitched-window reconstruction).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort_discovery_csv import discover_csv_cohort
from .config import Config, load_config
from .csv_signal import load_raw_csv
from .preprocess import select_channels
from .qc_config import QcConfig
from .reference import apply_reference


@dataclass
class SessionSignal:
    """A session's signal at one processing stage, all channels in
    `cfg.channels.keep` canonical order.

    Attributes
    ----------
    key : str
        ``"<patient_id>-<condition>"`` (condition lower-case), e.g. ``sub-34-a``.
    patient_id, condition : str
    ch_names : list of str
    sfreq : float
    data_uV : np.ndarray
        Shape ``(n_channels, n_samples)``, microvolts.
    """
    key: str
    patient_id: str
    condition: str
    ch_names: list[str]
    sfreq: float
    data_uV: np.ndarray


def pipeline_cfg_for_qc(qc: QcConfig, pipeline_cfg: Config | None = None) -> Config:
    """Pipeline `Config` whose `run_dir` resolves under `qc.source.processed_dir_name`.

    Mirrors `run_cohort_continuous_csv.py::_cfg_for_csv_continuous`'s
    `Config.model_copy` technique exactly (a standard pydantic method, not a
    `config.py` edit), so this audit reads exactly the output tree the
    continuous-CSV cohort run wrote -- without importing that top-level
    script as a module.
    """
    base = pipeline_cfg if pipeline_cfg is not None else load_config()
    new_paths = base.paths.model_copy(update={"processed_dir": qc.source.processed_dir_name})
    return base.model_copy(update={"paths": new_paths})


def load_manifest(run_dir: Path) -> pd.DataFrame:
    """Load the production run's `manifest.parquet`.

    Parameters
    ----------
    run_dir : Path
        `cfg.run_dir` of the pipeline config built by `pipeline_cfg_for_qc`.

    Returns
    -------
    pd.DataFrame
    """
    manifest_path = run_dir / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found at {manifest_path} -- is qc.yaml's "
            f"source.processed_dir_name pointing at a completed run?")
    return pd.read_parquet(manifest_path)


def cohort_sessions(manifest: pd.DataFrame) -> list[tuple[str, str]]:
    """Every distinct `(patient_id, condition)` pair present in the manifest.

    Discovered from the manifest itself (not a hardcoded subject list), so
    the audit always reflects whatever the production run actually processed.
    """
    pairs = manifest[["patient_id", "condition"]].drop_duplicates()
    return sorted(tuple(row) for row in pairs.itertuples(index=False, name=None))


def reconstruct_processed_session(run_dir: Path, manifest: pd.DataFrame,
                                  patient_id: str, condition: str,
                                  ch_names: list[str]) -> SessionSignal:
    """Stitch a session's non-overlapping window subset back into one array.

    `is_nonoverlap_subset` windows tile the recording end-to-end with no gaps
    and no overlap (their stride equals the window length -- see
    `window_continuous.py`); a trailing partial window shorter than
    `window.length_s` is dropped, matching production behaviour exactly (the
    windowing stage never emits it either).

    Parameters
    ----------
    run_dir : Path
        `cfg.run_dir` of the production run.
    manifest : pd.DataFrame
        As returned by `load_manifest`.
    patient_id, condition : str
    ch_names : list of str
        Canonical channel order (`cfg.channels.keep`); window files store
        `X` in this order but do not repeat the channel names per-window.

    Returns
    -------
    SessionSignal
    """
    sub = manifest[(manifest["patient_id"] == patient_id)
                   & (manifest["condition"] == condition)
                   & (manifest["is_nonoverlap_subset"])].sort_values("window_idx")
    if sub.empty:
        raise ValueError(
            f"{patient_id}-{condition}: no non-overlap-subset windows found in manifest")

    sfreq = float(sub["sfreq"].iloc[0])
    chunks: list[np.ndarray] = []
    for rel_path in sub["path"]:
        with open(run_dir / rel_path, "rb") as fh:
            rec = pickle.load(fh)
        chunks.append(rec["X"])
    data = np.concatenate(chunks, axis=1).astype(np.float64)

    if data.shape[0] != len(ch_names):
        raise ValueError(
            f"{patient_id}-{condition}: reconstructed {data.shape[0]} channels, "
            f"expected {len(ch_names)}")

    key = f"{patient_id}-{condition.lower()}"
    return SessionSignal(key=key, patient_id=patient_id, condition=condition,
                        ch_names=list(ch_names), sfreq=sfreq, data_uV=data)


def load_raw_session(cfg: Config, csv_path: Path, key: str) -> SessionSignal:
    """Pre-preprocessing signal: raw CSV -> re-reference -> channel-select only
    (no filter/notch/ICA/resample).

    Isolates the cleaning fingerprint the same way `psd_csv.py`'s
    `_raw_scalp_uV_csv` does, reusing the same three functions unmodified.

    Parameters
    ----------
    cfg : Config
        Pipeline config (as from `pipeline_cfg_for_qc`); only reref/channel
        settings are used, no path on it needs to exist.
    csv_path : Path
        The session's raw-signal CSV (`CsvRecording.csv_path`).
    key : str
        Recording key, e.g. ``sub-34-a``, for logging/error messages.

    Returns
    -------
    SessionSignal
    """
    raw = load_raw_csv(cfg, csv_path, key)
    raw = apply_reference(cfg, raw, key)
    raw = select_channels(cfg, raw, key)
    data = raw.get_data(units=cfg.signal.extract_units)
    patient_id, condition = key.rsplit("-", 1)
    return SessionSignal(key=key, patient_id=patient_id, condition=condition.upper(),
                        ch_names=list(raw.ch_names), sfreq=float(raw.info["sfreq"]),
                        data_uV=data)


def find_raw_csv_paths(drive_root: Path, patients: list[str]) -> dict[str, Path]:
    """Locate each `(patient, condition)` session's raw-signal CSV.

    Parameters
    ----------
    drive_root : Path
        Root of the shared LivingLab_PD drive.
    patients : list of str
        Patient IDs to look up (both conditions each).

    Returns
    -------
    dict
        Maps session key (e.g. ``sub-34-a``) -> raw-signal CSV path, for
        every session that was found unambiguously. Sessions that could not
        be located are omitted; call sites must check for missing keys
        explicitly rather than assume completeness.
    """
    recordings, missing = discover_csv_cohort(drive_root, patients)
    if missing:
        print(f"[qc_reconstruct] WARNING: could not locate raw CSV for "
              f"{len(missing)} session(s): {missing}")
    return {rec.key: rec.csv_path for rec in recordings}
