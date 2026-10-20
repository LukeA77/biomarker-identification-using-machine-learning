"""CSV counterpart to preprocess.py::load_raw (STAGE 1), for the raw-signal-CSV
cohort (see cohort_discovery_csv.py). Reads the DSI-Streamer raw-signal export
directly instead of the EDF, and builds an mne.io.RawArray whose channel names
already use the same `cfg.channels.rename_from_pattern` convention (e.g.
"EEG {name}-Pz") that `mne.io.read_raw_edf` produces for this dataset's EDF
exports -- see the "channel-naming trick" note in the module docstring of
run_cohort_continuous_csv.py. Everything downstream (apply_reference, filter_,
clean_with_ica, select_channels -- all imported unmodified) then works exactly
as it does on an EDF-loaded Raw, because those functions only ever look up
channel names, never the file format.

CSV structure (confirmed on multiple sessions of this cohort): a header block
of ~12-16 `# key =,value,...` metadata rows, then a row starting with `Time,`
that is the real column header, then data rows. Columns are plain 10-20 names
(Fp1, Fp2, ... A1, A2, ...); order and the exact extra column set (CM, Pz, X1..,
Trigger, Event, ...) both vary session to session and are irrelevant here -- we
only ever select by name, never by position.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .config import Config


def _parse_header(csv_path: Path) -> tuple[int, list[str], dict[str, str]]:
    """Scan the metadata block up to the `Time,...` column-header row.

    Returns (header_row_idx, column_names, meta):
      header_row_idx : 0-based line index of the 'Time,...' row -- usable
                        directly as `pandas.read_csv(..., skiprows=header_row_idx)`
                        (pandas then treats the next line as the header).
      column_names   : the actual column header list, in raw CSV order.
      meta           : key -> value for every '# Key =,Value,...' metadata row
                        seen before the header row.
    """
    meta: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for idx, row in enumerate(reader):
            if not row:
                continue
            first = row[0]
            if first.strip() == "Time":
                return idx, row, meta
            stripped = first.lstrip()
            if stripped.startswith("#"):
                key = stripped.lstrip("#").strip()
                if key.endswith("="):
                    key = key[:-1].strip()
                if key.endswith(":"):
                    key = key[:-1].strip()
                value = row[1].strip() if len(row) > 1 else ""
                meta[key] = value
            if idx > 60:
                break
    raise ValueError(
        f"{csv_path}: could not find the 'Time,...' column-header row in the "
        f"first 60 lines (expected a DSI-Streamer raw-signal CSV)")


def _meta_exact(meta: dict[str, str], key: str) -> str | None:
    for k, v in meta.items():
        if k.strip().lower() == key.lower():
            return v
    return None


def _meta_contains(meta: dict[str, str], needle: str) -> str | None:
    needle_l = needle.lower()
    for k, v in meta.items():
        if needle_l in k.lower():
            return v
    return None


def _parse_meas_date(meta: dict[str, str], key: str) -> datetime | None:
    """Best-effort UTC-labeled meas_date from the header's Date+Time fields.

    Not functionally required (no clock-alignment stage runs for continuous
    windowing); kept for provenance/consistency only. Falls back to None
    (raw.set_meas_date(None) is valid) on any parse failure rather than
    aborting the whole recording over metadata.
    """
    date_str = _meta_exact(meta, "date")
    time_str = _meta_exact(meta, "time")
    if not date_str or not time_str:
        print(f"[csv_signal] {key}: WARNING no Date/Time metadata found; meas_date left unset")
        return None
    date_str, time_str = date_str.strip(), time_str.strip()
    try:
        d = datetime.strptime(date_str, "%d/%m/%Y").date()
        fmt = "%H:%M:%S.%f" if "." in time_str else "%H:%M:%S"
        t = datetime.strptime(time_str, fmt).time()
        return datetime.combine(d, t, tzinfo=timezone.utc)
    except ValueError as e:
        print(f"[csv_signal] {key}: WARNING could not parse Date='{date_str}' "
              f"Time='{time_str}' ({e}); meas_date left unset")
        return None


def load_raw_csv(cfg: Config, csv_path: Path, key: str) -> mne.io.BaseRaw:
    """STAGE 1 (CSV variant): load the raw-signal CSV, verify sfreq/units,
    return an mne.io.RawArray with the 18 keep channels + reref reference
    channels (A1/A2), patterned-named exactly like the EDF loader's channels,
    ALL channels present (no selection here -- mirrors load_raw's contract:
    channel selection happens only after re-referencing).
    """
    header_row_idx, columns, meta = _parse_header(csv_path)

    sf_str = _meta_contains(meta, "sample_frequency")
    if sf_str is None:
        raise ValueError(f"{key}: no Sample_Frequency metadata found in {csv_path}")
    try:
        sf = float(sf_str)
    except ValueError:
        raise ValueError(f"{key}: unparsable Sample_Frequency value {sf_str!r} in {csv_path}")
    if int(round(sf)) != cfg.signal.orig_sfreq:
        raise ValueError(
            f"{key}: sfreq {sf} != expected orig_sfreq {cfg.signal.orig_sfreq} ({csv_path})")

    units_str = _meta_contains(meta, "sensor_data_units")
    if units_str is None or units_str.strip().lower() != cfg.signal.extract_units.lower():
        raise ValueError(
            f"{key}: Sensor_Data_Units {units_str!r} != expected "
            f"{cfg.signal.extract_units!r} ({csv_path})")

    # Channels needed: the 18 scalp channels + any reref reference channels not
    # already among them (A1/A2) -- the same pre-select, pre-reref channel set
    # load_raw (EDF) hands to apply_reference.
    needed_std: list[str] = list(cfg.channels.keep)
    for ref in cfg.reref.ref_channels:
        if ref not in needed_std:
            needed_std.append(ref)

    available = set(columns)
    if "Time" not in available:
        raise ValueError(f"{key}: no 'Time' column in {csv_path} (columns: {columns})")
    missing = [c for c in needed_std if c not in available]
    if missing:
        raise ValueError(
            f"{key}: missing required channel column(s) {missing} in {csv_path} "
            f"(available columns: {columns})")

    try:
        df = pd.read_csv(csv_path, skiprows=header_row_idx, usecols=needed_std,
                         dtype=np.float32, engine="c")
    except Exception as e:
        raise ValueError(f"{key}: failed to parse data rows of {csv_path}: {e}") from e

    if len(df) == 0:
        raise ValueError(f"{key}: no data rows read from {csv_path}")

    # A handful of sessions in this cohort end in one truncated/short data row
    # (a known DSI-Streamer export artifact, confirmed separately by a full-file
    # audit of this dataset: exactly 1 malformed row in each of 3 files cohort-
    # wide, always the final line). pandas parses a short row's missing trailing
    # fields as NaN rather than raising, so we must check for and drop these
    # explicitly. Only a SMALL number of NaN rows are treated as this known,
    # benign artifact and dropped with a loud warning; a large fraction is a
    # genuine corruption signal and still aborts the recording (no silent mass
    # coercion).
    nan_row_mask = ~np.isfinite(df[needed_std].to_numpy(dtype=np.float64)).all(axis=1)
    n_nan_rows = int(nan_row_mask.sum())
    if n_nan_rows:
        frac = n_nan_rows / len(df)
        if frac > 0.0001:  # >0.01% of rows non-finite: treat as real corruption
            raise ValueError(
                f"{key}: {n_nan_rows}/{len(df)} ({frac:.4%}) non-finite raw rows in "
                f"{csv_path} -- too many to treat as a benign truncated-line artifact")
        print(f"[csv_signal] {key}: WARNING dropping {n_nan_rows}/{len(df)} "
              f"non-finite row(s) (benign truncated-line artifact) from {csv_path}")
        df = df.loc[~nan_row_mask].reset_index(drop=True)

    data_uV = df[needed_std].to_numpy(dtype=np.float64).T   # (n_ch, n_samples)
    data_V = data_uV / 1e6                                   # RawArray wants SI units

    if not np.all(np.isfinite(data_V)):
        raise ValueError(f"{key}: non-finite raw values after loading {csv_path}")

    pattern = cfg.channels.rename_from_pattern
    ch_names = [pattern.format(name=std) for std in needed_std]
    info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types="eeg")
    raw = mne.io.RawArray(data_V, info, verbose="ERROR")

    meas_date = _parse_meas_date(meta, key)
    if meas_date is not None:
        raw.set_meas_date(meas_date)

    if int(round(raw.info["sfreq"])) != cfg.signal.orig_sfreq:
        raise ValueError(
            f"{key}: post-construction sfreq {raw.info['sfreq']} != {cfg.signal.orig_sfreq}")
    return raw
