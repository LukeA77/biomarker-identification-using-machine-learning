"""Mechanism check for the Tier-1 relative-band-power result (bal-acc 0.643, p=0.005).

This is NOT a search for a better number -- it is a confound audit. Relative band
power significantly decodes ON/OFF where CodeBrain does not, so before that claim
goes anywhere we need to know WHICH band and WHICH channels carry it, because the
most likely benign-looking-but-serious confound is movement/tremor artifact:

  * Parkinsonian rest tremor is 4-6 Hz -- squarely inside the theta band (4-8 Hz),
    and its harmonic lands in alpha. OFF state means more tremor.
  * Gross movement and EMG contaminate delta (<4 Hz) and gamma (>30 Hz).

So a classifier keyed on relative delta/theta/gamma may be detecting *the motor
symptom as an artifact in the EEG* rather than a cortical signature of medication
state. That is a materially different (and much weaker) claim than "cortical
spectral state decodes levodopa status", and it must be settled explicitly.

Per-band ablation (each band's 18 channels alone) + effect direction + topography.
Reuses the cached band-power tensor -- no re-extraction.

Run:  python codebrain_finetune/src/stage4c_bandpower_ablation.py
"""
from __future__ import annotations

import json
import logging
import sys
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

log = logging.getLogger("stage4c")
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
            "C3", "Cz", "C4", "T4", "T5", "P3", "P4", "T6", "O1", "O2"]


def lopo(F, y, pids, conds, nonoverlap, folds, ft_cfg, C, tag):
    prob, true, pat = [], [], []
    for fold in folds:
        tr = pids != fold.test_patient
        te = (pids == fold.test_patient) & nonoverlap
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(C=C, max_iter=ft_cfg.probe.max_iter, class_weight="balanced")
        clf.fit(sc.transform(F[tr]), y[tr])
        p = clf.predict_proba(sc.transform(F[te]))[:, 1]
        s = aggregate_session_level(pids[te], conds[te], p, y[te])
        prob += s.prob.tolist(); true += s.y_true.tolist(); pat += s.patients.tolist()
    prob, true, pat = np.array(prob), np.array(true), np.array(pat)
    agg = classification_metrics(true, prob, threshold=ft_cfg.eval.threshold)
    boot = bootstrap_session_ci(true, prob, pat, n_resamples=ft_cfg.eval.bootstrap.n_resamples,
                                ci=ft_cfg.eval.bootstrap.ci, seed=ft_cfg.seed.base)
    perm = permutation_test_balanced_accuracy(
        true, prob, pat, n_permutations=ft_cfg.eval.permutation.n_permutations,
        seed_base=ft_cfg.seed.base + ft_cfg.seed.permutation_offset)
    n_ord = sum(prob[(pat == p) & (true == 1)][0] > prob[(pat == p) & (true == 0)][0]
                for p in np.unique(pat))
    log.info("  %-18s bal-acc %.3f  CI [%.3f, %.3f]  AUC %-6s perm p %-7.3f ordering %2d/35",
             tag, agg.balanced_accuracy, boot["ci_low"] or np.nan, boot["ci_high"] or np.nan,
             f"{agg.auc:.3f}" if agg.auc is not None else "n/a", perm["p_value"], n_ord)
    return {"tag": tag, "aggregate": agg.as_dict(), "bootstrap": boot,
            "permutation": perm, "ordering_correct": int(n_ord)}


def main() -> int:
    logging.basicConfig(level="INFO", datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data_cfg = load_data_config(); ft_cfg = load_finetune_config()
    seed_everything(ft_cfg.seed.base, deterministic=ft_cfg.seed.deterministic)
    manifest = CohortManifest.from_config(data_cfg)
    folds = build_lopo_folds(manifest)
    df = manifest.subset(None, nonoverlap_only=False).copy()

    X = np.load(_RUN_ROOT / "results" / "bandpower_feats.npz")["X"]   # (N, 18, 5, 2)
    y = df["y_cond"].to_numpy().astype(int)
    pids = df["patient_id"].to_numpy().astype(str)
    conds = df["condition"].to_numpy().astype(str)
    nonoverlap = df["is_nonoverlap_subset"].to_numpy().astype(bool)
    C = ft_cfg.probe.C

    log.info("=" * 82)
    log.info("PER-BAND ABLATION of the relative-power result (18 channels per band)")
    results = [lopo(X[:, :, bi, 1], y, pids, conds, nonoverlap, folds, ft_cfg, C, f"rel::{b}")
               for bi, b in enumerate(BAND_NAMES)]

    # ---- effect direction & size, within subject (OFF minus ON), per band ----
    log.info("=" * 82)
    log.info("WITHIN-SUBJECT effect direction (OFF - ON), relative power, mean over channels")
    log.info("  %-8s %-22s %-14s %s", "band", "mean dOFF-ON", "patients OFF>ON", "Cohen's dz")
    direction = {}
    for bi, b in enumerate(BAND_NAMES):
        per_pat = []
        for p in np.unique(pids):
            m = pids == p
            on = X[m & (conds == "A"), :, bi, 1].mean()
            off = X[m & (conds == "C"), :, bi, 1].mean()
            per_pat.append(off - on)
        d = np.array(per_pat)
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
        direction[b] = {"mean_diff": float(d.mean()), "n_pos": int((d > 0).sum()), "dz": float(dz)}
        log.info("  %-8s %+.5f              %2d/35          %+.3f", b, d.mean(), (d > 0).sum(), dz)

    # ---- topography of the strongest band ----
    best = max(results, key=lambda r: r["aggregate"]["balanced_accuracy"])
    bi = BAND_NAMES.index(best["tag"].split("::")[1])
    log.info("=" * 82)
    log.info("TOPOGRAPHY of %s (within-subject OFF-ON per channel, x1000)", best["tag"])
    chan = []
    for ci, ch in enumerate(CHANNELS):
        d = [X[(pids == p) & (conds == "C"), ci, bi, 1].mean() -
             X[(pids == p) & (conds == "A"), ci, bi, 1].mean() for p in np.unique(pids)]
        chan.append((ch, float(np.mean(d) * 1000), int((np.array(d) > 0).sum())))
    for ch, v, npos in sorted(chan, key=lambda t: -abs(t[1])):
        log.info("  %-4s %+7.2f   (%2d/35 patients positive)", ch, v, npos)

    out = {"per_band": results, "direction": direction,
           "topography_band": BAND_NAMES[bi],
           "topography": [{"channel": c, "diff_x1000": v, "n_pos": n} for c, v, n in chan]}
    (_RUN_ROOT / "results" / "stage4c_bandpower_ablation.json").write_text(
        json.dumps(out, indent=2, default=str))
    log.info("=" * 82)
    log.info("Saved -> results/stage4c_bandpower_ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
