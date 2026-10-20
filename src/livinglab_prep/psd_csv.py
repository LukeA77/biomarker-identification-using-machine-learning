"""CSV-cohort counterpart to psd.py::compute_psd_check (Stage 4 PSD sanity check).

Verified before writing this module (per the task brief's "if it works
standalone off pre, verify" instruction): `compute_psd_check` is NOT reusable
unmodified for the CSV cohort. Its raw-baseline loader
(`psd.py::_raw_scalp_uV` -> `preprocess.py::load_reref_select` ->
`preprocess.py::load_raw`) calls `mne.io.read_raw_edf(rec.edf_path, ...)`, and
`CsvRecording` (cohort_discovery_csv.py) deliberately has no `edf_path` -- this
cohort is processed from the raw-signal CSV alone, no EDF is read at all.

This module reuses every reusable piece of psd.py UNMODIFIED (the `PSDCheck`
dataclass, the Welch/ratio helpers `_welch_mean`/`_median_ratio`, and the
plotting/logging helpers `_plot`/`_log`) and only swaps the raw-baseline
loader for the CSV equivalent: `load_raw_csv` -> `apply_reference` ->
`select_channels` (the same three-stage sequence `load_reref_select` uses,
just swapping the load step), so the PSD comparison isolates the filter
fingerprint exactly the same way as the EDF-cohort's Stage 4.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .csv_signal import load_raw_csv
from .preprocess import PreprocessedRecording, select_channels
from .psd import PSDCheck, _log, _median_ratio, _plot, _welch_mean
from .reference import apply_reference


def _raw_scalp_uV_csv(cfg: Config, pre: PreprocessedRecording) -> tuple[np.ndarray, float]:
    """Load the raw CSV (pre-filter) and return the 18 scalp channels in uV.

    Uses the SAME re-reference + channel selection as the processed path
    (load_raw_csv -> apply_reference -> select_channels), so the processed/raw
    PSD ratio isolates the FILTER fingerprint rather than mixing in the
    reference change -- mirrors psd.py::_raw_scalp_uV's contract exactly.
    """
    raw = load_raw_csv(cfg, pre.rec.csv_path, pre.rec.key)
    raw = apply_reference(cfg, raw, pre.rec.key)
    raw = select_channels(cfg, raw, pre.rec.key)
    return raw.get_data(units=cfg.signal.extract_units), float(raw.info["sfreq"])


def compute_psd_check_csv(cfg: Config, pre: PreprocessedRecording) -> PSDCheck:
    bp = cfg.signal.bandpass
    seg_s = cfg.window.length_s

    raw_data, fs_raw = _raw_scalp_uV_csv(cfg, pre)
    fr, praw = _welch_mean(raw_data, fs_raw, fs_raw * seg_s)
    fp, ppre = _welch_mean(pre.data_uV, pre.sfreq, pre.sfreq * seg_s)

    praw_i = np.interp(fp, fr, praw)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ppre / np.where(praw_i > 0, praw_i, np.nan)

    pass_ratio = _median_ratio(fp, ratio, 1.0, min(40.0, bp.h_freq))
    low_atten = _median_ratio(fp, ratio, 0.1, max(0.25, bp.l_freq - 0.05))
    high_lo = bp.h_freq + 5.0
    high_atten = _median_ratio(fp, ratio, high_lo, min(high_lo + 15.0, pre.sfreq / 2.0))

    notch = cfg.signal.notch_freq
    peak = ppre[(fp >= notch - 0.6) & (fp <= notch + 0.6)]
    neigh = ppre[((fp >= notch - 2) & (fp <= notch - 1)) | ((fp >= notch + 1) & (fp <= notch + 2))]
    notch_ratio = (float(peak.max() / neigh.mean())
                  if (peak.size and neigh.size and neigh.mean() > 0) else float("nan"))

    ok = (0.5 <= pass_ratio <= 2.0) and (low_atten < 0.5) and (high_atten < 0.1) and (notch_ratio < 2.0)
    check = PSDCheck(pre.rec.key, pass_ratio, low_atten, high_atten, notch_ratio, ok)

    _plot(cfg, fr, praw, fp, ppre, check)
    _log(check, cfg)
    return check
