"""QC Part B - aperiodic (1/f) slope via `specparam` (the current package name
for what was published as FOOOF; `fooof` is not installable/maintained under
that name any more -- see qc_signal_quality's smoke-test notes). Fits each
channel's Welch PSD (computed once in `qc_psd.py` and passed in here, not
recomputed), separating the aperiodic background component from periodic
oscillatory peaks.

Flags a channel/session when the aperiodic exponent is near zero (no real 1/f
decline -- the numeric counterpart of a flat PSD, Part A) or when the model
fit quality (R^2) is poor. Beta-band (13-30 Hz) peak presence is reported as a
Parkinsonian-motor sanity reference (levodopa-related beta suppression is a
well-established finding) -- not something this module forces a result toward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from specparam import Bands, SpectralModel

from .qc_config import QcConfig
from .qc_psd import ChannelPsd


@dataclass
class Peak:
    """One detected periodic peak.

    Attributes
    ----------
    center_freq, power, bandwidth : float
    """
    center_freq: float
    power: float
    bandwidth: float


@dataclass
class AperiodicFit:
    """One channel/session's specparam fit result.

    Attributes
    ----------
    channel : str
    exponent, offset, r_squared : float
    peaks : list of Peak
    beta_peak_present : bool
    beta_cf, beta_pw, beta_bw : float
        NaN if no peak fell in the configured beta band.
    exponent_near_zero, poor_fit : bool
        The two Part-B flag criteria (see module docstring).
    fit_error : str or None
        Non-None iff the specparam fit raised; all other fields are NaN/empty
        in that case (loud failure, not a silent skip -- caller must surface it).
    """
    channel: str
    exponent: float
    offset: float
    r_squared: float
    peaks: list[Peak] = field(default_factory=list)
    beta_peak_present: bool = False
    beta_cf: float = float("nan")
    beta_pw: float = float("nan")
    beta_bw: float = float("nan")
    exponent_near_zero: bool = False
    poor_fit: bool = False
    fit_error: str | None = None

    @property
    def flagged(self) -> bool:
        return self.fit_error is not None or self.exponent_near_zero or self.poor_fit

    @property
    def flag_reason(self) -> str:
        if self.fit_error is not None:
            return f"fit error: {self.fit_error}"
        reasons = []
        if self.exponent_near_zero:
            reasons.append(f"exponent near zero ({self.exponent:.3f})")
        if self.poor_fit:
            reasons.append(f"poor fit (R^2={self.r_squared:.3f})")
        return "; ".join(reasons) if reasons else "ok"


def fit_channel(channel_psd: ChannelPsd, qc: QcConfig) -> AperiodicFit:
    """Fit one channel's Welch PSD with specparam.

    Parameters
    ----------
    channel_psd : ChannelPsd
        From `qc_psd.compute_channel_psds` -- reuses that PSD, does not
        recompute Welch.
    qc : QcConfig

    Returns
    -------
    AperiodicFit
    """
    a = qc.aperiodic
    try:
        model = SpectralModel(
            aperiodic_mode="fixed", periodic_mode="gaussian",
            peak_width_limits=tuple(a.peak_width_limits_hz),
            max_n_peaks=a.max_n_peaks, min_peak_height=a.min_peak_height,
            peak_threshold=a.peak_threshold, verbose=False,
        )
        # power at f=0 is not physically meaningful on a log-power model; the
        # fit range already starts at 1 Hz so this is dropped implicitly, but
        # guard against a leading exact-zero frequency raising in specparam.
        freqs, power = channel_psd.freqs, channel_psd.power
        model.fit(freqs, power, freq_range=list(a.fit_freq_range_hz))

        bands = Bands({"beta": tuple(a.beta_band_hz)})
        row = model.to_df(bands=bands)
        exponent = float(row["exponent"])
        offset = float(row["offset"])
        r_squared = float(row["gof_rsquared"])
        beta_cf, beta_pw, beta_bw = float(row["beta_cf"]), float(row["beta_pw"]), float(row["beta_bw"])

        peak_params = np.atleast_2d(model.get_params("peak"))
        peaks = ([] if peak_params.size == 0 else
                [Peak(float(cf), float(pw), float(bw)) for cf, pw, bw in peak_params])

        exponent_near_zero = abs(exponent) < a.exponent_near_zero_threshold
        poor_fit = (not np.isfinite(r_squared)) or (r_squared < a.r2_poor_threshold)

        return AperiodicFit(
            channel=channel_psd.channel, exponent=exponent, offset=offset,
            r_squared=r_squared, peaks=peaks,
            beta_peak_present=bool(np.isfinite(beta_cf)),
            beta_cf=beta_cf, beta_pw=beta_pw, beta_bw=beta_bw,
            exponent_near_zero=exponent_near_zero, poor_fit=poor_fit,
        )
    except Exception as e:  # noqa: BLE001 - deliberately broad: a bad fit on
        # one channel must not abort the whole cohort audit; report it loudly
        # (fit_error) instead, per the brief's "defensive, loud failures".
        return AperiodicFit(
            channel=channel_psd.channel, exponent=float("nan"), offset=float("nan"),
            r_squared=float("nan"), fit_error=str(e),
        )


def peaks_to_str(peaks: list[Peak]) -> str:
    """Serialize detected peaks as `"cf/power/bandwidth; ..."` for a CSV cell."""
    return "; ".join(f"{p.center_freq:.2f}/{p.power:.3f}/{p.bandwidth:.2f}" for p in peaks)


def find_persistent_low_exponent_patients(aperiodic_df: pd.DataFrame, qc: QcConfig) -> pd.DataFrame:
    """Patients with a weak aperiodic exponent in a substantial fraction of
    channels in BOTH the ON and the OFF session.

    A single noisy session (e.g. movement/muscle artifact) can drag down one
    condition's fit quality without a real 1/f deficit; a persistent, both-
    conditions pattern is a much stronger signal of a genuine whole-cap
    contact/impedance problem specific to that patient (a hardware issue
    doesn't clear up between the ON and OFF visits).

    Parameters
    ----------
    aperiodic_df : pd.DataFrame
        Full `aperiodic_fit_results.csv` contents (columns include
        `patient_id`, `condition` ["ON"/"OFF"], `exponent`).
    qc : QcConfig

    Returns
    -------
    pd.DataFrame
        Indexed by `patient_id`, columns `frac_low_exponent_ON` and
        `frac_low_exponent_OFF` (fraction of channels with
        `exponent < qc.aperiodic.flat_candidate_exponent_threshold`),
        filtered to patients where BOTH fractions are
        `>= qc.aperiodic.flat_candidate_min_fraction`. Sorted by the minimum
        of the two fractions, descending (most persistent first).
    """
    thr = qc.aperiodic.flat_candidate_exponent_threshold
    min_frac = qc.aperiodic.flat_candidate_min_fraction

    low = aperiodic_df.assign(low_exp=aperiodic_df["exponent"] < thr)
    frac = low.groupby(["patient_id", "condition"])["low_exp"].mean().unstack("condition")
    for cond in ("ON", "OFF"):
        if cond not in frac.columns:
            frac[cond] = 0.0
    frac = frac.rename(columns={"ON": "frac_low_exponent_ON", "OFF": "frac_low_exponent_OFF"})

    both = frac[(frac["frac_low_exponent_ON"] >= min_frac) & (frac["frac_low_exponent_OFF"] >= min_frac)]
    both = both.assign(_min=both[["frac_low_exponent_ON", "frac_low_exponent_OFF"]].min(axis=1))
    return both.sort_values("_min", ascending=False).drop(columns="_min")
