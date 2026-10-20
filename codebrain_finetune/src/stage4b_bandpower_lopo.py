"""Tier-1 diagnostic — classical spectral-biomarker LOPO baseline.

**The question this answers:** is there ANY decodable ON/OFF levodopa signal in
this cohort, or is the near-chance CodeBrain result telling us the signal is not
recoverable from ambulatory EEG at N=35?

CodeBrain (frozen probe 0.443, LoRA 0.557, full-FT 0.443) cannot distinguish
"the foundation model's representation is wrong for this task" from "there is
nothing to decode." A classical band-power classifier settles it, because
beta-band (13-30 Hz) cortical power is *the* established levodopa-responsive EEG
marker: levodopa suppresses pathological beta. If beta decodes and CodeBrain
doesn't, that is a real finding about the model. If neither decodes, that is a
real finding about the data — and it is the honest floor for the whole thesis.

Method (deliberately mirrors ``stage4_probe_lopo.py`` so the numbers are
comparable): Welch PSD per 10 s window -> per-channel band powers -> LOPO
logistic regression in feature space -> aggregate the held-out patient's
non-overlap windows to session level -> pooled metrics + patient-cluster
bootstrap CI + within-subject permutation test.

Feature sets (ALL pre-declared before looking at any result, ALL reported --
no cherry-picking):
  * ``logabs``  log10 absolute band power, 18 ch x 5 bands = 90 feats  [PRIMARY]
  * ``rel``     relative band power (band / total), 90 feats
  * ``beta``    beta band only (log-abs + relative), 36 feats -- the a priori marker

Each is additionally run with ``--subject-norm``, which z-scores every feature
*within each subject using both of that subject's sessions* (unsupervised -- no
labels touched). This directly tests the mechanism measured in the LoRA result:
between-subject spread in predicted P(OFF) was 2.1x the within-subject ON/OFF
effect, so subject identity, not medication state, dominates the representation.
Subject-norm is the cheap feature-space analogue of Euclidean alignment. It is
*transductive* (uses both of a test patient's sessions at prediction time) and
MUST be reported as such -- it answers "which of this patient's two recordings is
OFF", not "is this single recording OFF".

Run:  python codebrain_finetune/src/stage4b_bandpower_lopo.py            # all variants
      python codebrain_finetune/src/stage4b_bandpower_lopo.py --features logabs
"""
from __future__ import annotations

import argparse
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
from codebrain_ft.tracking import git_sha, env_versions  # noqa: E402

from scipy.signal import welch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

log = logging.getLogger("stage4b")

EXPECTED_PATIENTS, EXPECTED_WINDOWS = 35, 37174

# Bands. Upper edge 45 Hz stays clear of the 50 Hz mains notch; lower edge 0.5 Hz
# clears the 0.3 Hz highpass. Beta is the a priori levodopa-responsive marker.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES = list(BANDS)
TOTAL_BAND = (0.5, 45.0)
_NPERSEG, _NOVERLAP = 400, 200   # 2 s segments @ 200 Hz -> 0.5 Hz resolution, 9 segments


def _window_features(args) -> np.ndarray:
    """Band powers for one cached window -> (n_ch, n_bands, 2) [log-abs, relative].

    Reads the RAW microvolt array (no /100 model normalisation) so the powers are
    physically interpretable.
    """
    cache_dir, rel_path = args
    with open(Path(cache_dir) / rel_path, "rb") as fh:
        x = pickle.load(fh)["X"]           # (18, 2000) float, microvolts
    x = np.asarray(x, dtype=np.float64)
    freqs, psd = welch(x, fs=200.0, nperseg=_NPERSEG, noverlap=_NOVERLAP, axis=-1)
    tot_mask = (freqs >= TOTAL_BAND[0]) & (freqs < TOTAL_BAND[1])
    total = np.trapezoid(psd[:, tot_mask], freqs[tot_mask], axis=-1)
    total = np.maximum(total, 1e-20)
    out = np.empty((psd.shape[0], len(BANDS), 2), dtype=np.float32)
    for bi, (lo, hi) in enumerate(BANDS.values()):
        m = (freqs >= lo) & (freqs < hi)
        p = np.trapezoid(psd[:, m], freqs[m], axis=-1)
        p = np.maximum(p, 1e-20)
        out[:, bi, 0] = np.log10(p)        # log absolute power
        out[:, bi, 1] = p / total          # relative power
    return out


def extract_all(manifest, data_cfg, workers: int) -> tuple[np.ndarray, "pd.DataFrame"]:
    """Band-power features for every window in the cohort -> (N, ch, band, 2)."""
    df = manifest.subset(None, nonoverlap_only=False).copy()
    cache = str(data_cfg.paths.cache_dir)
    jobs = [(cache, p) for p in df["path"].tolist()]
    log.info("Extracting band powers for %d windows (%d workers)...", len(jobs), workers)
    t0 = time.time()
    feats = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(ex.map(_window_features, jobs, chunksize=64)):
            feats.append(f)
            if (i + 1) % 5000 == 0:
                log.info("  %d/%d windows (%.0fs)", i + 1, len(jobs), time.time() - t0)
    X = np.stack(feats)
    log.info("Features %s in %.0fs", X.shape, time.time() - t0)
    return X, df


def build_feature_matrix(X: np.ndarray, kind: str) -> np.ndarray:
    """Flatten the (N, ch, band, 2) tensor into a design matrix for one feature set."""
    n = X.shape[0]
    if kind == "logabs":
        return X[:, :, :, 0].reshape(n, -1)
    if kind == "rel":
        return X[:, :, :, 1].reshape(n, -1)
    if kind == "beta":
        bi = BAND_NAMES.index("beta")
        return np.concatenate([X[:, :, bi, 0], X[:, :, bi, 1]], axis=1)
    raise ValueError(f"unknown feature set {kind!r}")


def subject_normalise(F: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """Z-score every feature within each subject (both sessions pooled, no labels).

    Removes the between-subject offset that dominates this representation while
    preserving each subject's within-subject ON/OFF contrast. Transductive.
    """
    out = F.copy()
    for p in np.unique(pids):
        m = pids == p
        mu = out[m].mean(axis=0)
        sd = out[m].std(axis=0)
        sd[sd < 1e-12] = 1.0
        out[m] = (out[m] - mu) / sd
    return out


def run_lopo(F, y, pids, conds, nonoverlap, folds, ft_cfg, *, C, tag):
    """LOPO logistic regression in feature space -> pooled session-level metrics."""
    all_prob, all_true, all_pat = [], [], []
    per_fold = []
    for fold in folds:
        tr = pids != fold.test_patient
        te = (pids == fold.test_patient) & nonoverlap
        scaler = StandardScaler().fit(F[tr])
        clf = LogisticRegression(C=C, max_iter=ft_cfg.probe.max_iter, class_weight="balanced")
        clf.fit(scaler.transform(F[tr]), y[tr])
        prob_te = clf.predict_proba(scaler.transform(F[te]))[:, 1]
        sess = aggregate_session_level(pids[te], conds[te], prob_te, y[te])
        per_fold.append({
            "fold": fold.index, "test_patient": fold.test_patient,
            "sessions": sess.sessions.tolist(),
            "prob": [round(float(p), 4) for p in sess.prob],
            "y_true": sess.y_true.tolist(),
        })
        all_prob += sess.prob.tolist(); all_true += sess.y_true.tolist()
        all_pat += sess.patients.tolist()

    all_prob = np.asarray(all_prob); all_true = np.asarray(all_true); all_pat = np.asarray(all_pat)
    agg = classification_metrics(all_true, all_prob, threshold=ft_cfg.eval.threshold)
    boot = bootstrap_session_ci(all_true, all_prob, all_pat,
                                n_resamples=ft_cfg.eval.bootstrap.n_resamples,
                                ci=ft_cfg.eval.bootstrap.ci, seed=ft_cfg.seed.base)
    perm = permutation_test_balanced_accuracy(
        all_true, all_prob, all_pat,
        n_permutations=ft_cfg.eval.permutation.n_permutations,
        seed_base=ft_cfg.seed.base + ft_cfg.seed.permutation_offset)

    # within-subject ordering (the paired question), reported alongside
    on = {r["test_patient"]: r["prob"][r["y_true"].index(0)] for r in per_fold}
    off = {r["test_patient"]: r["prob"][r["y_true"].index(1)] for r in per_fold}
    n_ord = sum(off[p] > on[p] for p in on)

    log.info("[%-22s] bal-acc %.3f CI [%.3f, %.3f] | AUC %s | perm p %.3f | ordering %d/%d",
             tag, agg.balanced_accuracy,
             boot["ci_low"] if boot["ci_low"] is not None else float("nan"),
             boot["ci_high"] if boot["ci_high"] is not None else float("nan"),
             f"{agg.auc:.3f}" if agg.auc is not None else "n/a",
             perm["p_value"], n_ord, len(on))
    return {
        "tag": tag, "aggregate": agg.as_dict(), "bootstrap": boot, "permutation": perm,
        "ordering_correct": int(n_ord), "n_patients": len(on), "per_fold": per_fold,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier-1 classical band-power LOPO baseline")
    ap.add_argument("--features", nargs="+", default=["logabs", "rel", "beta"],
                    choices=["logabs", "rel", "beta"])
    ap.add_argument("--C", type=float, default=None, help="logreg C (default: probe.C)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-npz", default=None,
                    help="path to cache/reuse the extracted band-power tensor")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data_cfg = load_data_config()
    ft_cfg = load_finetune_config()
    C = args.C if args.C is not None else ft_cfg.probe.C
    seed_everything(ft_cfg.seed.base, deterministic=ft_cfg.seed.deterministic)

    manifest = CohortManifest.from_config(data_cfg)
    n_pat, n_win = len(manifest.patients), manifest.n_windows
    log.info("Cohort: %d patients, %d windows, balance=%s", n_pat, n_win, manifest.label_balance())
    if n_pat != EXPECTED_PATIENTS or n_win != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch — STOP. got (%d,%d) expected (%d,%d)",
                  n_pat, n_win, EXPECTED_PATIENTS, EXPECTED_WINDOWS)
        return 1

    npz = Path(args.cache_npz) if args.cache_npz else (_RUN_ROOT / "results" / "bandpower_feats.npz")
    if npz.exists():
        log.info("Reusing cached band-power tensor %s", npz)
        z = np.load(npz, allow_pickle=True)
        X = z["X"]
        df = manifest.subset(None, nonoverlap_only=False).copy()
        if len(df) != len(X):
            log.error("Cached features (%d) disagree with manifest (%d) — delete %s and re-run.",
                      len(X), len(df), npz)
            return 1
    else:
        X, df = extract_all(manifest, data_cfg, args.workers)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, X=X)
        log.info("Cached band-power tensor -> %s", npz)

    y = df["y_cond"].to_numpy().astype(int)
    pids = df["patient_id"].to_numpy().astype(str)
    conds = df["condition"].to_numpy().astype(str)
    nonoverlap = df["is_nonoverlap_subset"].to_numpy().astype(bool)
    folds = build_lopo_folds(manifest)

    results = []
    for kind in args.features:
        F = build_feature_matrix(X, kind)
        results.append(run_lopo(F, y, pids, conds, nonoverlap, folds, ft_cfg,
                                C=C, tag=f"{kind}"))
        Fn = subject_normalise(F, pids)
        results.append(run_lopo(Fn, y, pids, conds, nonoverlap, folds, ft_cfg,
                                C=C, tag=f"{kind}+subjnorm"))

    out = {
        "run_tag": "stage4b_bandpower_lopo", "arm": "bandpower",
        "git_sha": git_sha(_PROJECT), "env": env_versions(),
        "bands": BANDS, "welch": {"nperseg": _NPERSEG, "noverlap": _NOVERLAP},
        "C": C, "results": results,
        "note": ("subjnorm variants are TRANSDUCTIVE (z-score within subject using "
                 "both sessions, unsupervised); they answer the paired question."),
    }
    dest = _RUN_ROOT / "results" / "stage4b_bandpower_lopo.json"
    dest.write_text(json.dumps(out, indent=2, default=str))

    log.info("=" * 78)
    log.info("TIER-1 CLASSICAL BAND-POWER BASELINE — pooled over 35 folds / 70 sessions")
    log.info("  %-24s %-9s %-18s %-7s %-8s %s", "variant", "bal-acc", "95% CI", "AUC", "perm p", "ordering")
    for r in results:
        a, b, p = r["aggregate"], r["bootstrap"], r["permutation"]
        log.info("  %-24s %-9.3f [%.3f, %.3f]   %-7s %-8.3f %d/%d",
                 r["tag"], a["balanced_accuracy"],
                 b["ci_low"] or float("nan"), b["ci_high"] or float("nan"),
                 f"{a['auc']:.3f}" if a["auc"] is not None else "n/a",
                 p["p_value"], r["ordering_correct"], r["n_patients"])
    log.info("  reference: CodeBrain LoRA 0.557 [0.457,0.643] AUC 0.541 p 0.194 | frozen 0.443")
    log.info("=" * 78)
    log.info("Saved -> %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
