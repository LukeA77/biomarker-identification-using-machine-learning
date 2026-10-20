"""Narrow-band spectral profile — the decisive tremor-vs-cortex test.

The Tier-1 result (relative band power, bal-acc 0.643/0.657, p=0.005) collapses to
chance when EITHER the theta band (4-8 Hz) OR the lateral temporal/frontal
electrodes (F7/F8/T3/T4) are removed. Two explanations remain, and they license
very different claims:

  (a) TREMOR / EMG ARTIFACT — Parkinsonian rest tremor is 4-6 Hz and OFF state
      means more tremor. Its electrical signature on the scalp is strongest at
      lateral/temporal sites. Predicts a SHARP, NARROW peak at 4-6 Hz, and
      per-patient peaks at that patient's own tremor frequency.
  (b) CORTICAL SLOWING — genuine dopaminergic theta increase, a documented PD
      finding. Predicts a BROAD elevation across the whole 4-8 Hz band, without a
      sharp narrow peak.

This script extracts the full 0.5 Hz-resolution PSD once (cached), then:
  1. profiles the within-subject OFF-ON relative-power difference per 0.5 Hz bin,
  2. classifies from each narrow band alone (LOPO, session-level),
  3. reports each patient's individual peak-difference frequency,
  4. contrasts lateral (F7/F8/T3/T4) vs midline/central channels.

A sharp 4-6 Hz peak concentrated on lateral channels = tremor. Answer honestly
either way -- this determines what the dissertation can claim.

Run:  python codebrain_finetune/src/stage4d_spectral_profile.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent
_PROJECT = _RUN_ROOT.parent
sys.path.insert(0, str(_RUN_ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from livinglab_codebrain import load_data_config, CohortManifest, build_lopo_folds  # noqa: E402
from codebrain_ft import load_finetune_config, seed_everything  # noqa: E402
from codebrain_ft.metrics import (  # noqa: E402
    aggregate_session_level, classification_metrics,
    bootstrap_session_ci, permutation_test_balanced_accuracy,
)
from scipy.signal import welch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

log = logging.getLogger("stage4d")

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
            "C3", "Cz", "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]
LATERAL = ["F7", "F8", "T3", "T4"]          # muscle/tremor-prone
MIDLINE = ["Fz", "Cz", "F3", "F4", "C3", "C4"]
_NPERSEG, _NOVERLAP = 400, 200               # 0.5 Hz resolution
FMIN, FMAX = 0.5, 45.0
PSD_NPZ = _RUN_ROOT / "results" / "psd_profile.npz"


def _window_psd(args):
    cache_dir, rel_path = args
    with open(Path(cache_dir) / rel_path, "rb") as fh:
        x = np.asarray(pickle.load(fh)["X"], dtype=np.float64)
    f, p = welch(x, fs=200.0, nperseg=_NPERSEG, noverlap=_NOVERLAP, axis=-1)
    m = (f >= FMIN) & (f <= FMAX)
    p = p[:, m]
    return (p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-20)).astype(np.float32)


def extract(manifest, data_cfg, workers):
    df = manifest.subset(None, nonoverlap_only=False).copy()
    jobs = [(str(data_cfg.paths.cache_dir), p) for p in df["path"]]
    log.info("Extracting 0.5 Hz relative PSD for %d windows...", len(jobs))
    t0 = time.time(); out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_window_psd, jobs, chunksize=64)):
            out.append(r)
            if (i + 1) % 10000 == 0:
                log.info("  %d/%d (%.0fs)", i + 1, len(jobs), time.time() - t0)
    P = np.stack(out)
    log.info("PSD tensor %s in %.0fs", P.shape, time.time() - t0)
    return P, df


def lopo(F, y, pids, conds, no, folds, fc, tag=None):
    pr, tr_, pa = [], [], []
    for f in folds:
        t = pids != f.test_patient; e = (pids == f.test_patient) & no
        s = StandardScaler().fit(F[t])
        c = LogisticRegression(C=fc.probe.C, max_iter=fc.probe.max_iter,
                               class_weight="balanced").fit(s.transform(F[t]), y[t])
        p = c.predict_proba(s.transform(F[e]))[:, 1]
        ss = aggregate_session_level(pids[e], conds[e], p, y[e])
        pr += ss.prob.tolist(); tr_ += ss.y_true.tolist(); pa += ss.patients.tolist()
    pr, tr_, pa = np.array(pr), np.array(tr_), np.array(pa)
    a = classification_metrics(tr_, pr, threshold=fc.eval.threshold)
    b = bootstrap_session_ci(tr_, pr, pa, n_resamples=fc.eval.bootstrap.n_resamples,
                             ci=fc.eval.bootstrap.ci, seed=fc.seed.base)
    pm = permutation_test_balanced_accuracy(
        tr_, pr, pa, n_permutations=fc.eval.permutation.n_permutations,
        seed_base=fc.seed.base + fc.seed.permutation_offset)
    n = sum(pr[(pa == p) & (tr_ == 1)][0] > pr[(pa == p) & (tr_ == 0)][0] for p in np.unique(pa))
    if tag:
        log.info("  %-30s bal-acc %.3f CI [%.3f,%.3f] AUC %-6s p %-7.4f ord %2d/35", tag,
                 a.balanced_accuracy, b["ci_low"] or np.nan, b["ci_high"] or np.nan,
                 f"{a.auc:.3f}" if a.auc is not None else "n/a", pm["p_value"], n)
    return {"tag": tag, "aggregate": a.as_dict(), "bootstrap": b, "permutation": pm,
            "ordering_correct": int(n)}


def main() -> int:
    logging.basicConfig(level="INFO", datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dc = load_data_config(); fc = load_finetune_config()
    seed_everything(fc.seed.base, deterministic=fc.seed.deterministic)
    mf = CohortManifest.from_config(dc); folds = build_lopo_folds(mf)
    df = mf.subset(None, nonoverlap_only=False).copy()

    freqs = np.arange(FMIN, FMAX + 1e-9, 0.5)
    if PSD_NPZ.exists():
        log.info("Reusing %s", PSD_NPZ.name)
        P = np.load(PSD_NPZ)["P"]
    else:
        P, df = extract(mf, dc, 8)
        PSD_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PSD_NPZ, P=P)
    freqs = freqs[:P.shape[-1]]

    y = df["y_cond"].to_numpy().astype(int)
    pids = df["patient_id"].to_numpy().astype(str)
    conds = df["condition"].to_numpy().astype(str)
    no = df["is_nonoverlap_subset"].to_numpy().astype(bool)
    pats = np.unique(pids)

    # ---- 1. within-subject OFF-ON profile per 0.5 Hz bin -------------------
    log.info("=" * 88)
    log.info("WITHIN-SUBJECT OFF-ON relative-power profile (2-14 Hz), lateral vs midline")
    lat = [CHANNELS.index(c) for c in LATERAL]
    mid = [CHANNELS.index(c) for c in MIDLINE]
    prof = {"all": [], "lateral": [], "midline": []}
    per_pat_diff = np.zeros((len(pats), len(freqs)))
    for pi, p in enumerate(pats):
        mA = (pids == p) & (conds == "A"); mC = (pids == p) & (conds == "C")
        per_pat_diff[pi] = P[mC].mean(axis=(0, 1)) - P[mA].mean(axis=(0, 1))
    band = (freqs >= 2) & (freqs <= 14)
    log.info("  %-7s %-12s %-12s %-12s %s", "Hz", "all-ch d", "lateral d", "midline d", "n OFF>ON")
    for fi in np.where(band)[0]:
        dall, dlat, dmid = [], [], []
        for p in pats:
            mA = (pids == p) & (conds == "A"); mC = (pids == p) & (conds == "C")
            dall.append(P[mC][:, :, fi].mean() - P[mA][:, :, fi].mean())
            dlat.append(P[mC][:, lat, fi].mean() - P[mA][:, lat, fi].mean())
            dmid.append(P[mC][:, mid, fi].mean() - P[mA][:, mid, fi].mean())
        prof["all"].append(float(np.mean(dall))); prof["lateral"].append(float(np.mean(dlat)))
        prof["midline"].append(float(np.mean(dmid)))
        log.info("  %-7.1f %+.6f    %+.6f    %+.6f    %2d/35", freqs[fi],
                 np.mean(dall), np.mean(dlat), np.mean(dmid), int(np.sum(np.array(dall) > 0)))

    # ---- 2. classification from each narrow band alone ---------------------
    log.info("=" * 88)
    log.info("LOPO from NARROW bands alone (relative power, all 18 channels)")
    narrow = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 14), (4, 8)]
    narrow_res = []
    for lo, hi in narrow:
        m = (freqs >= lo) & (freqs < hi)
        F = P[:, :, m].reshape(len(P), -1)
        narrow_res.append(lopo(F, y, pids, conds, no, folds, fc, tag=f"{lo}-{hi} Hz"))

    # ---- 3. per-patient peak difference frequency -------------------------
    log.info("=" * 88)
    tremor_band = (freqs >= 3.5) & (freqs <= 7.5)
    peaks = [float(freqs[tremor_band][np.argmax(per_pat_diff[pi][tremor_band])]) for pi in range(len(pats))]
    peaks = np.array(peaks)
    log.info("PER-PATIENT peak OFF-ON frequency within 3.5-7.5 Hz:")
    log.info("  median %.1f Hz | IQR %.1f-%.1f | spread(sd) %.2f Hz",
             np.median(peaks), np.percentile(peaks, 25), np.percentile(peaks, 75), peaks.std())
    hist = {f"{f:.1f}": int((peaks == f).sum()) for f in np.unique(peaks)}
    log.info("  distribution: %s", hist)

    out = {"freqs": freqs.tolist(), "profile_2_14Hz": prof,
           "narrow_band_lopo": narrow_res,
           "per_patient_peak_hz": peaks.tolist(),
           "peak_median": float(np.median(peaks)), "peak_sd": float(peaks.std())}
    (_RUN_ROOT / "results" / "stage4d_spectral_profile.json").write_text(json.dumps(out, indent=2, default=str))
    log.info("Saved -> results/stage4d_spectral_profile.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
