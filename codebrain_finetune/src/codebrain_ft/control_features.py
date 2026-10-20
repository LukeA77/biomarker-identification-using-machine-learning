"""Per-window control features that SURVIVE input normalisation.

WHY THIS MODULE EXISTS
----------------------
Experiment 12c measured the within-patient task-split leak with AMPLITUDE
controls (``amp1``, ``crude3``, ``chanamp``). The supervisor's proposed fix is
``chan_z`` — per-window, per-channel z-scoring. Under ``chan_z`` those controls
are **constant by construction**: z-scoring forces every channel to mean 0 and
std 1 (then a single shared rescale), so ``chanamp`` reads ~0.500 whether or not
the recording-identity leak is gone. The headline control stops being evidence
precisely when it is needed.

The blocks defined here describe a channel's spectral **shape** rather than its
scale, so ``chan_z`` cannot trivially erase them:

* ``relbp``     — per-channel RELATIVE band power (each channel's five band
  powers normalised to sum to 1 within that channel);
* ``aperiodic`` — per-channel 1/f exponent + offset. This dataset contains **no
  impedance data anywhere** (the QC audit checked raw-CSV metadata/columns, EDF
  channels/annotations and the ``.dsi`` byte stream across 31-33 files each and
  found none — ``qc_signal_quality/qc_summary_report.md`` §C), so the aperiodic
  exponent is the only available proxy for contact quality.

None of these can carry a neural levodopa signature that the amplitude controls
could not; they are the same kind of negative control, chosen so that
normalisation does not silently zero them out.

NORMALISATION IS NOT REIMPLEMENTED HERE. Windows are drawn through the real
:class:`~.data.NormalizedWindowDataset` wrapping the real ``WindowDataset``, so
a control sees byte-identical input to what the model arms see, including the
``target_scale`` rescale to the cohort-median window std.

Every threshold, band edge and Welch parameter comes from
``codebrain_finetune/config/controls.yaml``; the aperiodic fit settings are
inherited from ``config/qc.yaml`` so the per-window fit is the same model as the
cohort fit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

__all__ = [
    "ControlsConfig", "load_controls_config",
    "FEATURE_BLOCKS", "CONTROL_DESCRIPTIONS",
    "compute_feature_blocks", "load_or_compute_features", "chanamp_features",
    "control_matrix", "control_target",
]

# Blocks stored in the feature cache. `recid` is NOT a block: it is a different
# TARGET evaluated on a configured union of these blocks (see `control_target`).
FEATURE_BLOCKS = ("amp1", "crude3", "chanamp", "relbp", "aperiodic")

CONTROL_DESCRIPTIONS = {
    "amp1": "mean|x| only (1 feature)",
    "crude3": "mean|x|, std, max|x| (3 features)",
    "chanamp": "per-channel mean|x| (18 — electrode-contact fingerprint)",
    "relbp": "per-channel RELATIVE band power, 5 bands (90 — scale-free spectral shape)",
    "aperiodic": "per-channel 1/f exponent + offset (36 — contact-quality proxy)",
    "recid": "which of this patient's TWO RECORDINGS is this window from (target, not feature)",
}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class ControlsConfig:
    """Typed view of ``config/controls.yaml`` (fail-loud on missing keys)."""

    input_norm: dict
    psd: dict
    bands: dict
    aperiodic: dict
    recid: dict
    classifier: dict
    transfer_test: dict
    cache: dict
    output: dict
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def band_names(self) -> tuple[str, ...]:
        return tuple(self.bands)

    @property
    def schemes(self) -> list[str]:
        return list(self.input_norm["schemes"])


def load_controls_config(root: Path | None = None) -> ControlsConfig:
    """Load ``config/controls.yaml``, merging the inherited ``config/qc.yaml`` keys.

    The aperiodic settings are deliberately NOT duplicated between the two files:
    ``qc.yaml`` remains the single source of truth for what a 1/f fit means in
    this project, and ``controls.yaml`` names the keys it inherits.
    """
    from livinglab_codebrain.config import run_root

    root = root or run_root()
    path = root / "config" / "controls.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"controls config not found: {path}")
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = ("input_norm", "psd", "bands", "aperiodic", "recid",
                "classifier", "transfer_test", "cache", "output")
    for key in required:
        if key not in d:
            raise KeyError(f"controls.yaml: missing required top-level key '{key}'")

    ap = dict(d["aperiodic"])
    inherit_rel = ap.pop("inherit_from", None)
    inherit_keys = ap.pop("inherit_keys", [])
    if inherit_rel:
        # project root = the directory ABOVE codebrain_finetune/
        qc_path = root.parent / inherit_rel
        if not qc_path.is_file():
            raise FileNotFoundError(
                f"controls.yaml aperiodic.inherit_from points at {qc_path}, which does not "
                f"exist. The per-window 1/f fit must use the same settings as the cohort fit; "
                f"refusing to silently substitute defaults.")
        qc_ap = yaml.safe_load(qc_path.read_text(encoding="utf-8"))["aperiodic"]
        for k in inherit_keys:
            if k not in qc_ap:
                raise KeyError(f"{qc_path}: aperiodic.{k} not found (named in controls.yaml "
                               f"aperiodic.inherit_keys)")
            ap[k] = qc_ap[k]
        ap["_inherited_from"] = str(inherit_rel)

    return ControlsConfig(
        input_norm=d["input_norm"], psd=d["psd"], bands=d["bands"], aperiodic=ap,
        recid=d["recid"], classifier=d["classifier"], transfer_test=d["transfer_test"],
        cache=d["cache"], output=d["output"], raw=d,
    )


# ---------------------------------------------------------------------------
# per-window feature computation
# ---------------------------------------------------------------------------
def _welch_psd(sig: np.ndarray, sfreq: float, psd_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel Welch PSD of a ``(C, T)`` window."""
    from scipy.signal import welch

    nperseg = int(round(float(psd_cfg["welch_segment_s"]) * sfreq))
    noverlap = int(round(nperseg * float(psd_cfg["welch_overlap_frac"])))
    if nperseg > sig.shape[-1]:
        raise ValueError(
            f"welch_segment_s={psd_cfg['welch_segment_s']} needs {nperseg} samples but the "
            f"window is only {sig.shape[-1]}; lower it in controls.yaml.")
    freqs, power = welch(sig, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                         detrend=psd_cfg.get("detrend", "constant"), axis=-1)
    return freqs, power


def _relative_band_power(freqs: np.ndarray, power: np.ndarray, bands: dict) -> np.ndarray:
    """``(C, n_bands)`` band powers normalised to sum to 1 WITHIN each channel.

    Scale-free by construction: multiplying a channel by any constant leaves
    this unchanged, which is exactly why ``chan_z`` cannot remove it.
    """
    cols = []
    for lo, hi in bands.values():
        m = (freqs >= float(lo)) & (freqs < float(hi))
        if not m.any():
            raise ValueError(f"band [{lo}, {hi}) Hz contains no PSD bin at this resolution")
        cols.append(power[..., m].sum(axis=-1))
    bp = np.stack(cols, axis=-1)                       # (C, n_bands)
    total = bp.sum(axis=-1, keepdims=True)
    return bp / np.clip(total, 1e-30, None)


def _aperiodic_loglog(freqs: np.ndarray, power: np.ndarray, ap_cfg: dict) -> np.ndarray:
    """Fast fallback: iterative peak-masked log-log least squares, ``(C, 2)``.

    The same idea as specparam's initial aperiodic estimate — fit a line in
    log-log, drop the points that sit above it (the oscillatory peaks), refit —
    but without the full periodic model. Used for smoke tests; the reported runs
    use ``estimator: specparam``.
    """
    lo, hi = ap_cfg["fit_freq_range_hz"]
    m = (freqs >= float(lo)) & (freqs <= float(hi))
    lf = np.log10(freqs[m])
    lp = np.log10(np.clip(power[..., m], 1e-30, None))
    out = np.empty((lp.shape[0], 2), dtype=np.float64)
    for c in range(lp.shape[0]):
        keep = np.ones(lf.shape, dtype=bool)
        slope = intercept = 0.0
        for _ in range(2):
            slope, intercept = np.polyfit(lf[keep], lp[c][keep], 1)
            resid = lp[c] - (slope * lf + intercept)
            keep = resid <= 0.0                        # drop points above the fit (peaks)
            if keep.sum() < 4:
                keep = np.ones(lf.shape, dtype=bool)
                break
        out[c] = (-slope, intercept)                   # (exponent, offset), specparam sign
    return out


def _aperiodic_specparam(freqs: np.ndarray, power: np.ndarray, ap_cfg: dict) -> np.ndarray:
    """``(C, 2)`` = (exponent, offset) per channel, same model as ``qc_aperiodic.py``.

    A failed fit yields NaN for that channel rather than a number: imputation is
    done later, per (patient, channel), and is recorded. Substituting 0.0 here
    would invent a value the data does not contain.
    """
    from specparam import SpectralModel

    out = np.full((power.shape[0], 2), np.nan, dtype=np.float64)
    for c in range(power.shape[0]):
        try:
            model = SpectralModel(
                aperiodic_mode="fixed", periodic_mode="gaussian",
                peak_width_limits=tuple(ap_cfg["peak_width_limits_hz"]),
                max_n_peaks=int(ap_cfg["max_n_peaks"]),
                min_peak_height=float(ap_cfg["min_peak_height"]),
                peak_threshold=float(ap_cfg["peak_threshold"]), verbose=False,
            )
            model.fit(freqs, power[c], freq_range=list(ap_cfg["fit_freq_range_hz"]))
            row = model.to_df()
            out[c] = (float(row["exponent"]), float(row["offset"]))
        except Exception:                              # noqa: BLE001 — one bad channel
            pass                                       # must not abort the cohort
    return out


def _blocks_for_chunk(rows: pd.DataFrame, cache_dir: Path, window_spec, scheme: str,
                      target_scale: float | None, psd_cfg: dict, bands: dict,
                      ap_cfg: dict, wanted: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Compute the requested feature blocks for one chunk of windows (one worker)."""
    import warnings

    from livinglab_codebrain.dataset import WindowDataset
    from .data import NormalizedWindowDataset

    ds = WindowDataset(rows, cache_dir, window_spec)
    if scheme != "none":
        if target_scale is None:
            raise ValueError("target_scale is required when scheme != 'none'")
        ds = NormalizedWindowDataset(ds, scheme, target_scale)

    sfreq = float(window_spec.expected_sfreq)
    estimator = ap_cfg.get("estimator", "specparam")
    n, n_bands, n_ch = len(rows), len(bands), window_spec.n_channels
    shapes = {"amp1": 1, "crude3": 3, "chanamp": n_ch,
              "relbp": n_ch * n_bands, "aperiodic": n_ch * 2}
    out = {b: np.empty((n, shapes[b])) for b in wanted}
    need_psd = bool({"relbp", "aperiodic"} & set(wanted))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            x = ds[i][0].numpy()                       # (C, P, S), post-normalisation
            sig = x.reshape(x.shape[0], -1).astype(np.float64)
            absX = np.abs(sig)
            if "amp1" in out:
                out["amp1"][i] = absX.mean()
            if "crude3" in out:
                out["crude3"][i] = (absX.mean(), sig.std(), absX.max())
            if "chanamp" in out:
                out["chanamp"][i] = absX.mean(axis=1)
            if need_psd:
                freqs, power = _welch_psd(sig, sfreq, psd_cfg)
                if "relbp" in out:
                    out["relbp"][i] = _relative_band_power(freqs, power, bands).ravel()
                if "aperiodic" in out:
                    ap = (_aperiodic_specparam(freqs, power, ap_cfg) if estimator == "specparam"
                          else _aperiodic_loglog(freqs, power, ap_cfg))
                    out["aperiodic"][i] = ap.ravel()
    return out


def compute_feature_blocks(frame: pd.DataFrame, data_cfg, cc: ControlsConfig, *,
                           scheme: str, norm_scale: float | None,
                           blocks: tuple[str, ...] = FEATURE_BLOCKS,
                           n_jobs: int | None = None) -> dict[str, np.ndarray]:
    """Requested feature blocks for every window in ``frame``, row-aligned to it."""
    from joblib import Parallel, delayed

    unknown = [b for b in blocks if b not in FEATURE_BLOCKS]
    if unknown:
        raise ValueError(f"unknown feature block(s) {unknown}; known: {list(FEATURE_BLOCKS)}")
    blocks = tuple(blocks)
    n_jobs = int(n_jobs if n_jobs is not None else cc.aperiodic.get("n_jobs", 1))
    frame = frame.reset_index(drop=True)
    n = len(frame)
    n_jobs = max(1, min(n_jobs, n))
    bounds = np.linspace(0, n, n_jobs + 1).astype(int)
    chunks = [frame.iloc[a:b].reset_index(drop=True) for a, b in zip(bounds[:-1], bounds[1:])
              if b > a]
    log.info("Computing control features %s for %d windows | norm=%s | %d chunk(s) x %d job(s) "
             "| aperiodic estimator=%s", list(blocks), n, scheme, len(chunks), n_jobs,
             cc.aperiodic.get("estimator"))
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_blocks_for_chunk)(
            chunk, data_cfg.paths.cache_dir, data_cfg.window, scheme, norm_scale,
            cc.psd, cc.bands, cc.aperiodic, blocks)
        for chunk in chunks)
    return {b: np.concatenate([p[b] for p in parts], axis=0) for b in blocks}


def _impute_aperiodic(blocks: dict[str, np.ndarray], frame: pd.DataFrame,
                      how: str) -> tuple[dict[str, np.ndarray], dict]:
    """Fill failed 1/f fits with the (patient, column) median; report how many."""
    A = blocks["aperiodic"]
    bad = ~np.isfinite(A)
    report = {"n_nonfinite_cells": int(bad.sum()), "n_cells": int(A.size),
              "n_windows_with_any": int(bad.any(axis=1).sum()), "method": how}
    if not bad.any():
        return blocks, report
    if how != "median":
        raise ValueError(f"aperiodic.impute must be 'median', got {how!r}")
    pids = frame["patient_id"].to_numpy()
    for p in np.unique(pids):
        m = pids == p
        sub = A[m]
        med = np.nanmedian(np.where(np.isfinite(sub), sub, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)     # a wholly-failed column -> 0
        A[m] = np.where(np.isfinite(sub), sub, med)
    log.warning("Aperiodic fit failed for %d/%d cells (%d windows); imputed with the "
                "per-patient per-column median.", report["n_nonfinite_cells"],
                report["n_cells"], report["n_windows_with_any"])
    blocks["aperiodic"] = A
    return blocks, report


def load_or_compute_features(frame: pd.DataFrame, data_cfg, cc: ControlsConfig, *,
                             scheme: str, norm_scale: float | None, run_root: Path,
                             force: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """Cached feature blocks for ``frame`` under ``scheme`` (skip-if-exists).

    The cache stores the key columns it was built from and refuses to be reused
    if they no longer match the frame — a stale cache would silently attach the
    wrong features to the wrong windows.
    """
    import hashlib

    cache_dir = run_root / cc.cache["dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_cols = list(cc.cache["key_columns"])
    key = np.asarray(["|".join(str(v) for v in row)
                      for row in frame[key_cols].itertuples(index=False)])
    # CONTENT-ADDRESSED, not just scheme-named. A smoke run over 3 patients and
    # the full 27-patient run produce different files instead of one silently
    # overwriting the other — and the estimator is in the digest, so a
    # `loglog_robust` cache can never be mistaken for a `specparam` one.
    digest = hashlib.sha256(
        ("\n".join(key.tolist()) + f"|{scheme}|{cc.aperiodic.get('estimator')}"
         f"|{norm_scale}|{sorted(cc.bands.items())}|{sorted(cc.psd.items())}"
        ).encode()).hexdigest()[:12]
    path = cache_dir / f"features_{scheme}_{digest}.npz"

    if path.is_file() and not force:
        # Close the handle before any write: on Windows an open NpzFile keeps a
        # lock that makes the tmp+replace below fail with PermissionError.
        with np.load(path, allow_pickle=False) as z:
            blocks = {b: z[b] for b in FEATURE_BLOCKS}
        log.info("Feature cache HIT %s (%d windows)", path.name, len(key))
        return blocks, {"cache": "hit", "path": str(path),
                        "aperiodic_impute": {"note": "applied before caching"}}

    blocks = compute_feature_blocks(frame, data_cfg, cc, scheme=scheme, norm_scale=norm_scale)
    blocks, imp = _impute_aperiodic(blocks, frame, cc.aperiodic.get("impute", "median"))
    tmp = path.with_suffix(".npz.tmp")
    # Write through a file handle: np.savez_compressed appends ".npz" to a *path*
    # that does not already end in it, which would silently defeat tmp+replace.
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, _key=key, **blocks)
    tmp.replace(path)
    log.info("Wrote feature cache %s", path)
    return blocks, {"cache": "miss", "path": str(path), "aperiodic_impute": imp}


def chanamp_features(frame: pd.DataFrame, data_cfg, cc: ControlsConfig, *,
                     scheme: str, norm_scale: float | None,
                     n_jobs: int | None = None) -> np.ndarray:
    """``(N, 18)`` per-channel mean|x| under ``scheme`` — the amplitude control.

    Thin wrapper so the pooled/LOPO drivers get the amplitude control through the
    SAME normalised windows the model arms see, instead of each reimplementing a
    raw-pickle read that silently ignores ``--input-norm``.
    """
    return compute_feature_blocks(frame, data_cfg, cc, scheme=scheme, norm_scale=norm_scale,
                                  blocks=("chanamp",), n_jobs=n_jobs)["chanamp"]


# ---------------------------------------------------------------------------
# control -> (X, y)
# ---------------------------------------------------------------------------
def control_matrix(blocks: dict[str, np.ndarray], control: str, cc: ControlsConfig) -> np.ndarray:
    """Feature matrix for a named control (``recid`` = the configured union)."""
    if control in FEATURE_BLOCKS:
        return blocks[control]
    if control == "recid":
        names = list(cc.recid["feature_blocks"])
        unknown = [n for n in names if n not in FEATURE_BLOCKS]
        if unknown:
            raise ValueError(f"controls.yaml recid.feature_blocks names unknown block(s) {unknown}")
        return np.concatenate([blocks[n] for n in names], axis=1)
    raise ValueError(f"unknown control {control!r}")


def control_target(frame: pd.DataFrame, control: str, cc: ControlsConfig) -> np.ndarray:
    """Label vector for a control.

    Every control except ``recid`` predicts the ON/OFF condition. ``recid``
    predicts RECORDING IDENTITY — and within a patient that is, by construction,
    the same variable, because the patient has exactly one ON recording and one
    OFF recording. That identity is not a defect of the control, it is the
    finding it exists to state: on this design "decode levodopa state" and
    "decode which recording" are the same classification problem, so no
    task-split ON/OFF number can be more than an alias of recording identity.
    The two are computed separately anyway, and asserted equal per patient, so
    the claim is checked rather than assumed.
    """
    if control != "recid":
        return frame["y_cond"].to_numpy().astype(int)
    target = cc.recid.get("target", "recording_id")
    if target != "recording_id":
        raise ValueError(f"controls.yaml recid.target must be 'recording_id', got {target!r}")
    sess = frame["session"].to_numpy().astype(str)
    pids = frame["patient_id"].to_numpy().astype(str)
    y = np.empty(len(frame), dtype=int)
    for p in np.unique(pids):
        m = pids == p
        uniq = sorted(np.unique(sess[m]))
        if len(uniq) != 2:
            raise ValueError(f"{p}: {len(uniq)} recordings ({uniq}); recid assumes exactly 2")
        y[m] = np.searchsorted(uniq, sess[m])
    return y
