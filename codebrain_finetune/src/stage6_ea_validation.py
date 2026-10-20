"""Cheap validation of Euclidean Alignment BEFORE committing to a 15 h LoRA run.

Same prototype-before-scaling discipline used throughout. EA is the last
principled lever (see `codebrain_ft/align.py` for why it is legitimate under this
project's no-hand-crafted-features constraint). Rather than spend 15 h finding out,
this measures EA's effect where it is cheap to measure:

  1. frozen CodeBrain probe, raw vs EA-aligned input          (~4 min)
  2. duration-matched evaluation, raw vs EA                    (removes the
     recording-length confound found in Experiment 10)
  3. the trivial duration-only baseline (0.614) as the floor to beat

Every number is scored against a REFIT-BASED null (labels shuffled within patient,
whole pipeline refit), because Experiment 10 showed the post-hoc permutation test
understates the null.

Run:  python codebrain_finetune/src/stage6_ea_validation.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent
_PROJECT = _RUN_ROOT.parent
sys.path.insert(0, str(_RUN_ROOT / "src"))
sys.path.insert(0, str(_PROJECT / "codebrain_finetune" / "external" / "CodeBrain"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402
from livinglab_codebrain import load_data_config, CohortManifest, build_lopo_folds  # noqa: E402
from livinglab_codebrain.dataset import WindowDataset  # noqa: E402
from livinglab_codebrain.loader import make_dataloader  # noqa: E402
from codebrain_ft import (  # noqa: E402
    load_finetune_config, build_classifier_model, seed_everything,
    cohort_median_scale, load_or_compute_whiteners, EAWindowDataset,
    duration_matched_mask,
)
from codebrain_ft.probe import extract_features  # noqa: E402
from codebrain_ft.metrics import (  # noqa: E402
    aggregate_session_level, classification_metrics, bootstrap_session_ci,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

log = logging.getLogger("stage6")
EXPECTED_PATIENTS, EXPECTED_WINDOWS = 35, 37174
DURATION_ONLY_BASELINE = 0.614      # Experiment 10c — the floor any model must beat
N_NULL = 100


def lopo(F, y, pids, conds, mask, folds, fc):
    prob, true, pat = [], [], []
    for fold in folds:
        tr = (pids != fold.test_patient) & mask
        te = (pids == fold.test_patient) & mask
        if tr.sum() == 0 or te.sum() == 0:
            continue
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(C=fc.probe.C, max_iter=fc.probe.max_iter,
                                 class_weight="balanced").fit(sc.transform(F[tr]), y[tr])
        p = clf.predict_proba(sc.transform(F[te]))[:, 1]
        s = aggregate_session_level(pids[te], conds[te], p, y[te])
        prob += s.prob.tolist(); true += s.y_true.tolist(); pat += s.patients.tolist()
    prob, true, pat = np.array(prob), np.array(true), np.array(pat)
    m = classification_metrics(true, prob, threshold=fc.eval.threshold)
    n_ord = sum(prob[(pat == p) & (true == 1)][0] > prob[(pat == p) & (true == 0)][0]
                for p in np.unique(pat))
    return m, prob, true, pat, int(n_ord)


def refit_null(F, y, pids, conds, mask, folds, fc, n=N_NULL, seed0=7000):
    vals = []
    for rep in range(n):
        rng = np.random.default_rng(seed0 + rep)
        ys = y.copy()
        for p in np.unique(pids):
            if rng.random() < 0.5:
                m = pids == p
                ys[m] = 1 - ys[m]
        m_, *_ = lopo(F, ys, pids, conds, mask, folds, fc)
        vals.append(m_.balanced_accuracy)
    return np.array(vals)


def evaluate(tag, F, y, pids, conds, mask, folds, fc, results):
    m, prob, true, pat, n_ord = lopo(F, y, pids, conds, mask, folds, fc)
    boot = bootstrap_session_ci(true, prob, pat, n_resamples=fc.eval.bootstrap.n_resamples,
                               ci=fc.eval.bootstrap.ci, seed=fc.seed.base)
    null = refit_null(F, y, pids, conds, mask, folds, fc)
    p_emp = (np.sum(null >= m.balanced_accuracy) + 1) / (len(null) + 1)
    log.info("  %-34s bal-acc %.3f  CI [%.3f,%.3f]  AUC %-6s  refit-p %.3f  ord %2d/35",
             tag, m.balanced_accuracy, boot["ci_low"] or np.nan, boot["ci_high"] or np.nan,
             f"{m.auc:.3f}" if m.auc is not None else "n/a", p_emp, n_ord)
    log.info("      null: mean %.3f sd %.3f 95th %.3f", null.mean(), null.std(),
             np.percentile(null, 95))
    results.append({"tag": tag, "bal_acc": m.balanced_accuracy, "auc": m.auc,
                    "ci": [boot["ci_low"], boot["ci_high"]], "refit_p": float(p_emp),
                    "ordering": n_ord, "null_mean": float(null.mean()),
                    "null_sd": float(null.std())})


def main() -> int:
    logging.basicConfig(level="INFO", datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not torch.cuda.is_available():
        log.error("CUDA unavailable — aborting."); return 1
    device = "cuda"
    dc = load_data_config(); fc = load_finetune_config()
    seed_everything(fc.seed.base, deterministic=fc.seed.deterministic)
    mf = CohortManifest.from_config(dc)
    if len(mf.patients) != EXPECTED_PATIENTS or mf.n_windows != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch — STOP."); return 1
    folds = build_lopo_folds(mf)
    df = mf.subset(None, nonoverlap_only=False).copy()

    y = df["y_cond"].to_numpy().astype(int)
    pids = df["patient_id"].to_numpy().astype(str)
    conds = df["condition"].to_numpy().astype(str)
    no = df["is_nonoverlap_subset"].to_numpy().astype(bool)

    target = cohort_median_scale(mf, dc, seed=fc.seed.base)
    whiteners = load_or_compute_whiteners(mf, dc, _RUN_ROOT / "results" / "ea_whiteners.npz")

    # smallest session governs k for duration matching (cohort must not change)
    k = int(min(((pids == p) & (conds == c) & no).sum()
                for p in np.unique(pids) for c in ("A", "C")))
    matched = duration_matched_mask(df, k=k, where="middle", nonoverlap_only=True)
    log.info("Duration matching: k=%d windows/session (middle block), %d sessions",
             k, matched.sum() // k)

    base_ds = WindowDataset(df, dc.paths.cache_dir, dc.window)
    results = []
    feats = {}
    for tag, ds in (("raw", base_ds),
                    ("EA", EAWindowDataset(base_ds, whiteners, target))):
        loader = make_dataloader(ds, dc, shuffle=False, seed=fc.seed.base)
        model = build_classifier_model(dc, fc, mode="frozen", device=device)
        t0 = time.time()
        X, yy, pp, _ = extract_features(model, loader, fc, device=device)
        log.info("[%s] frozen features %s in %.0fs", tag, X.shape, time.time() - t0)
        del model; torch.cuda.empty_cache()
        assert np.array_equal(yy, y) and np.array_equal(pp, pids), "loader order drifted"
        feats[tag] = X

    log.info("=" * 96)
    log.info("FROZEN CodeBrain probe — raw vs Euclidean Alignment (refit-based nulls)")
    for tag in ("raw", "EA"):
        evaluate(f"frozen [{tag}] all windows", feats[tag], y, pids, conds, no, folds, fc, results)
        evaluate(f"frozen [{tag}] duration-matched", feats[tag], y, pids, conds, matched, folds, fc, results)

    log.info("=" * 96)
    log.info("REFERENCE FLOOR: duration-only baseline = %.3f (no EEG at all)", DURATION_ONLY_BASELINE)
    log.info("Any model not clearly above %.3f has not demonstrated it uses the EEG.",
             DURATION_ONLY_BASELINE)
    log.info("=" * 96)
    for r in sorted(results, key=lambda r: -r["bal_acc"]):
        log.info("  %-34s %.3f  refit-p %.3f", r["tag"], r["bal_acc"], r["refit_p"])

    (_RUN_ROOT / "results" / "stage6_ea_validation.json").write_text(
        json.dumps({"k_matched": k, "target_scale": target,
                    "duration_only_baseline": DURATION_ONLY_BASELINE,
                    "results": results}, indent=2, default=str))
    log.info("Saved -> results/stage6_ea_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
