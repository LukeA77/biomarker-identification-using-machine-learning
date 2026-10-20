"""QC Part A - PSD sanity checks.

For every channel of every session, computes a Welch PSD of the preprocessed
continuous signal and a FOOOF/specparam-independent "flatness" heuristic (see
module docstring of `qc_config.py`'s `QcPsd` fields for the rationale). For a
small representative sample of sessions, also renders:

  - `psd_examples/<key>.png`   -- 18-channel grid, log-log axes.
  - `psd_before_after_examples/<key>.png` -- 18-channel grid, raw (pre-
    preprocessing) vs processed overlay, log-log axes.

A channel's PSD is "flat" if it shows no discernible 1/f-like decline: the
mean power in a low reference band is not meaningfully higher than in a high
reference band. This is deliberately a simple, direct heuristic, independent
of the FOOOF/specparam aperiodic fit in `qc_aperiodic.py` (Part B), so the two
checks corroborate rather than duplicate each other.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from .qc_config import QcConfig
from .qc_reconstruct import SessionSignal

_GRID_SHAPE = (6, 3)  # 18 channels


@dataclass
class ChannelPsd:
    """One channel's Welch PSD plus the flatness heuristic.

    Attributes
    ----------
    channel : str
    freqs, power : np.ndarray
        Welch PSD (uV^2/Hz), from `f=0` up to Nyquist.
    low_high_ratio_db : float
        `10*log10(mean_power(low_band) / mean_power(high_band))`.
    flat_flag : bool
        True iff `low_high_ratio_db` is below `qc.psd.flat_ratio_db_threshold`.
    """
    channel: str
    freqs: np.ndarray
    power: np.ndarray
    low_high_ratio_db: float
    flat_flag: bool


def welch_psd(data_1d: np.ndarray, sfreq: float, qc: QcConfig) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of one channel's continuous signal.

    Parameters
    ----------
    data_1d : np.ndarray
        Shape `(n_samples,)`, microvolts.
    sfreq : float
    qc : QcConfig

    Returns
    -------
    freqs, power : np.ndarray
    """
    nperseg = int(round(qc.psd.welch_segment_s * sfreq))
    nperseg = min(nperseg, data_1d.shape[-1])
    noverlap = int(round(nperseg * qc.psd.welch_overlap_frac))
    freqs, power = welch(data_1d, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    return freqs, power


def _band_mean_power(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return float("nan")
    return float(np.mean(power[mask]))


def flatness_check(freqs: np.ndarray, power: np.ndarray, qc: QcConfig) -> tuple[float, bool]:
    """Low-band-vs-high-band mean-power ratio (dB) and the flat-PSD flag.

    Parameters
    ----------
    freqs, power : np.ndarray
        As returned by `welch_psd`.
    qc : QcConfig

    Returns
    -------
    ratio_db : float
    flat_flag : bool
    """
    lo1, hi1 = qc.psd.flat_low_band_hz
    lo2, hi2 = qc.psd.flat_high_band_hz
    low = _band_mean_power(freqs, power, lo1, hi1)
    high = _band_mean_power(freqs, power, lo2, hi2)
    if not (np.isfinite(low) and np.isfinite(high)) or high <= 0 or low <= 0:
        return float("nan"), True  # can't establish any decline -> flag, loudly
    ratio_db = float(10 * np.log10(low / high))
    return ratio_db, ratio_db < qc.psd.flat_ratio_db_threshold


def compute_channel_psds(session: SessionSignal, qc: QcConfig) -> list[ChannelPsd]:
    """`welch_psd` + `flatness_check` for every channel of one session."""
    results = []
    for ch_idx, ch in enumerate(session.ch_names):
        freqs, power = welch_psd(session.data_uV[ch_idx], session.sfreq, qc)
        ratio_db, flat = flatness_check(freqs, power, qc)
        results.append(ChannelPsd(ch, freqs, power, ratio_db, flat))
    return results


def _grid_fig(title: str):
    fig, axes = plt.subplots(*_GRID_SHAPE, figsize=(13, 16))
    fig.suptitle(title, fontsize=13)
    return fig, axes.flat


def plot_psd_grid(session: SessionSignal, channel_psds: list[ChannelPsd],
                  qc: QcConfig, out_path) -> None:
    """18-channel PSD grid (log-log axes) for one session.

    Parameters
    ----------
    session : SessionSignal
        The preprocessed session (used for the title only; PSDs are passed in
        so they are computed exactly once and shared with the results CSV).
    channel_psds : list of ChannelPsd
    qc : QcConfig
    out_path : Path
    """
    fig, axes = _grid_fig(f"{session.key}: preprocessed continuous PSD (Welch, log-log)")
    for ax, cp in zip(axes, channel_psds):
        f, p = cp.freqs[1:], cp.power[1:]  # drop DC bin (0 Hz undefined on log axis)
        ax.loglog(f, p, color="C0", lw=1.0)
        flag_note = " [FLAT]" if cp.flat_flag else ""
        ax.set_title(f"{cp.channel}{flag_note}", fontsize=9,
                    color="crimson" if cp.flat_flag else "black")
        ax.set_xlabel("Hz", fontsize=8)
        ax.set_ylabel(r"$\mu V^2$/Hz", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=qc.plots.dpi)
    plt.close(fig)


def reference_band(all_clean_channel_psds: list[ChannelPsd]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool many clean channels' PSDs into a "what normal EEG looks like" band.

    Parameters
    ----------
    all_clean_channel_psds : list of ChannelPsd
        Every channel, from every clean reference session/patient, sharing
        the same frequency grid (same `sfreq`/`welch_segment_s`).

    Returns
    -------
    freqs, p10, median, p90 : np.ndarray
        The 10th/50th/90th percentile power across all pooled channels, at
        each frequency bin.
    """
    freqs = all_clean_channel_psds[0].freqs
    stacked = np.stack([cp.power for cp in all_clean_channel_psds])
    p10, median, p90 = np.percentile(stacked, [10, 50, 90], axis=0)
    return freqs, p10, median, p90


def plot_patient_vs_reference(patient_id: str,
                              condition_psds: dict[str, list[ChannelPsd]],
                              condition_reference: dict[str, tuple],
                              condition_stats: dict[str, str],
                              qc: QcConfig, out_path) -> None:
    """One evidence figure per patient: their 18 channels vs a normal-EEG
    reference band, one panel per condition (ON/OFF).

    Parameters
    ----------
    patient_id : str
    condition_psds : dict
        `{"ON": [ChannelPsd, ...], "OFF": [ChannelPsd, ...]}` for this patient.
    condition_reference : dict
        `{"ON": (freqs, p10, median, p90), "OFF": (...)}` from `reference_band`,
        computed over clean reference patients for that condition.
    condition_stats : dict
        `{"ON": "<annotation text>", "OFF": "<annotation text>"}`.
    qc : QcConfig
    out_path : Path
    """
    conditions = [c for c in ("ON", "OFF") if c in condition_psds]
    fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 6))
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conditions):
        freqs, p10, median, p90 = condition_reference[cond]
        ax.fill_between(freqs[1:], p10[1:], p90[1:], color="0.75", alpha=0.6,
                        label="normal EEG range (10th-90th pct, clean reference patients)")
        ax.loglog(freqs[1:], median[1:], color="0.35", lw=1.5, ls="--", label="clean reference median")
        for cp in condition_psds[cond]:
            ax.loglog(cp.freqs[1:], cp.power[1:], lw=0.8, alpha=0.8)
        ax.set_title(f"{patient_id} -- {cond}", fontsize=12)
        ax.set_xlabel("Hz")
        ax.set_ylabel(r"$\mu V^2$/Hz")
        ax.text(0.03, 0.05, condition_stats.get(cond, ""), transform=ax.transAxes,
               fontsize=9, va="bottom", ha="left",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{patient_id}: all 18 channels (colored) vs normal-EEG reference band (grey)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=qc.plots.dpi)
    plt.close(fig)


def plot_before_after_grid(raw: SessionSignal, processed: SessionSignal,
                           raw_psds: list[ChannelPsd], processed_psds: list[ChannelPsd],
                           qc: QcConfig, out_path) -> None:
    """18-channel raw-vs-processed PSD overlay (log-log axes) for one session.

    Parameters
    ----------
    raw, processed : SessionSignal
    raw_psds, processed_psds : list of ChannelPsd
        Same channel order as `processed.ch_names` (both sides use the
        canonical `cfg.channels.keep` order).
    qc : QcConfig
    out_path : Path
    """
    fig, axes = _grid_fig(f"{processed.key}: raw (pre-preprocessing) vs processed PSD")
    for ax, rp, pp in zip(axes, raw_psds, processed_psds):
        ax.loglog(rp.freqs[1:], rp.power[1:], color="0.6", lw=1.0, label="raw")
        ax.loglog(pp.freqs[1:], pp.power[1:], color="C0", lw=1.2, label="processed")
        ax.set_title(pp.channel, fontsize=9)
        ax.set_xlabel("Hz", fontsize=8)
        ax.set_ylabel(r"$\mu V^2$/Hz", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=qc.plots.dpi)
    plt.close(fig)
