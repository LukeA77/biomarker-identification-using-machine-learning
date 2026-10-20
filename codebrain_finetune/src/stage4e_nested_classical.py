"""Properly nested classical baseline — an UNBIASED estimate of the headline result.

Why this exists. Experiment 5 reported relative band power at bal-acc 0.643 (0.657
without gamma, perm p=0.005). But the choice of *which* feature set to report was
made by looking at LOPO test results across 6 variants. That is selection on the
test set: the 0.657 is optimistically biased, and the honest way to report it is
either "exploratory, 6 variants, Bonferroni p=0.03" or -- better -- to re-run with
the feature-set choice made INSIDE the cross-validation, never touching the held-out
patient.

This script does the latter. Nested LOPO:

  OUTER (35 folds): hold out one patient.
  INNER (5-fold GroupKFold on the 34 training patients): score every candidate
        feature set, pick the winner by inner session-level balanced accuracy.
  REFIT on all 34 training patients using the inner-selected feature set.
  PREDICT the held-out patient. The held-out patient influenced nothing.

The pooled result is then an unbiased estimate of the whole *pipeline*
(feature-set selection included), which is the number that belongs in the
dissertation. It will very likely be LOWER than 0.657 -- that is the point, and it
is not a regression: it is the same pipeline honestly scored.

Also reports how often each candidate was chosen, which shows whether the
selection is stable or noise-driven (the LoRA HP search was noise-driven: all 10
configs scored within 0.557-0.629).

Run:  python codebrain_finetune/src/stage4e_nested_classical.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
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
from codebrain_ft import load_finetune_config, seed_everything, inner_group_split  # noqa: E402
from codebrain_ft.metrics import (  # noqa: E402
    aggregate_session_level, classification_metrics,
    bootstrap_session_ci, permutation_test_balanced_accuracy,
)
from codebrain_ft.tracking import git_sha, env_versions  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

log = logging.getLogger("stage4e")
EXPECTED_PATIENTS, EXPECTED_WINDOWS = 35, 37174
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


def candidate_sets(B: np.ndarray) -> dict[str, np.ndarray]:
    """The feature sets the inner loop chooses between (B is (N, 18, 5, 2))."""
    n = len(B)
    gi = BAND_NAMES.index("gamma")
    bi = BAND_NAMES.index("beta")
    keep_nogamma = [i for i in range(5) if i != gi]
    return {
        "logabs":       B[:, :, :, 0].reshape(n, -1),
        "rel":          B[:, :, :, 1].reshape(n, -1),
        "rel_nogamma":  B[:, :, keep_nogamma, 1].reshape(n, -1),
        "beta":         np.concatenate([B[:, :, bi, 0], B[:, :, bi, 1]], axis=1),
        "logabs+rel":   B.reshape(n, -1),
    }


def _fit_predict(F, y, train_mask, test_mask, fc):
    sc = StandardScaler().fit(F[train_mask])
    clf = LogisticRegression(C=fc.probe.C, max_iter=fc.probe.max_iter,
                             class_weight="balanced").fit(sc.transform(F[train_mask]), y[train_mask])
    return clf.predict_proba(sc.transform(F[test_mask]))[:, 1]


def main() -> int:
    logging.basicConfig(level="INFO", datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dc = load_data_config(); fc = load_finetune_config()
    seed_everything(fc.seed.base, deterministic=fc.seed.deterministic)
    mf = CohortManifest.from_config(dc)
    if len(mf.patients) != EXPECTED_PATIENTS or mf.n_windows != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch — STOP."); return 1
    folds = build_lopo_folds(mf)
    df = mf.subset(None, nonoverlap_only=False).copy()

    B = np.load(_RUN_ROOT / "results" / "bandpower_feats.npz")["X"]
    sets = candidate_sets(B)
    y = df["y_cond"].to_numpy().astype(int)
    pids = df["patient_id"].to_numpy().astype(str)
    conds = df["condition"].to_numpy().astype(str)
    nonoverlap = df["is_nonoverlap_subset"].to_numpy().astype(bool)

    log.info("Nested LOPO over %d candidate feature sets: %s", len(sets), list(sets))

    prob, true, pat, chosen, per_fold = [], [], [], [], []
    for fold in folds:
        train_pats = list(fold.train_patients)
        inner = inner_group_split(train_pats, n_folds=fc.nested_lopo.inner_folds,
                                  seed=fc.seed.base + fold.index)
        # --- inner selection: score each candidate on the inner folds only ---
        scores = {}
        for name, F in sets.items():
            vals = []
            for itr, iva in inner:
                tm = np.isin(pids, itr)
                vm = np.isin(pids, iva) & nonoverlap
                p = _fit_predict(F, y, tm, vm, fc)
                s = aggregate_session_level(pids[vm], conds[vm], p, y[vm])
                vals.append(classification_metrics(s.y_true, s.prob,
                                                   threshold=fc.eval.threshold).balanced_accuracy)
            scores[name] = float(np.nanmean(vals))
        best = max(scores, key=scores.get)
        chosen.append(best)

        # --- refit on all 34 with the inner-selected set, predict held-out ---
        F = sets[best]
        tm = pids != fold.test_patient
        em = (pids == fold.test_patient) & nonoverlap
        p = _fit_predict(F, y, tm, em, fc)
        s = aggregate_session_level(pids[em], conds[em], p, y[em])
        prob += s.prob.tolist(); true += s.y_true.tolist(); pat += s.patients.tolist()
        per_fold.append({"fold": fold.index, "test_patient": fold.test_patient,
                         "selected": best, "inner_scores": scores,
                         "prob": [round(float(v), 4) for v in s.prob],
                         "y_true": s.y_true.tolist()})
        log.info("fold %2d %-9s | selected %-12s (inner %.3f) | probs %s true %s",
                 fold.index, fold.test_patient, best, scores[best],
                 [round(float(v), 3) for v in s.prob], s.y_true.tolist())

    prob, true, pat = np.array(prob), np.array(true), np.array(pat)
    agg = classification_metrics(true, prob, threshold=fc.eval.threshold)
    boot = bootstrap_session_ci(true, prob, pat, n_resamples=fc.eval.bootstrap.n_resamples,
                                ci=fc.eval.bootstrap.ci, seed=fc.seed.base)
    perm = permutation_test_balanced_accuracy(
        true, prob, pat, n_permutations=fc.eval.permutation.n_permutations,
        seed_base=fc.seed.base + fc.seed.permutation_offset)
    n_ord = sum(prob[(pat == p) & (true == 1)][0] > prob[(pat == p) & (true == 0)][0]
                for p in np.unique(pat))

    log.info("=" * 84)
    log.info("NESTED CLASSICAL BASELINE — unbiased, feature set chosen inside CV")
    log.info("  selection counts: %s", dict(Counter(chosen)))
    log.info("  balanced accuracy = %.3f  (95%% CI [%.3f, %.3f])",
             agg.balanced_accuracy, boot["ci_low"], boot["ci_high"])
    log.info("  AUC=%.3f  sens(OFF)=%.3f  spec(ON)=%.3f  macro_f1=%.3f",
             agg.auc, agg.sensitivity, agg.specificity, agg.macro_f1)
    log.info("  permutation p = %.4f (null mean %.3f) | ordering %d/35",
             perm["p_value"], perm["null_mean"], n_ord)
    log.info("  (compare: test-selected 'rel_nogamma' reported 0.657, p=0.005 — biased)")
    log.info("=" * 84)

    out = {"run_tag": "stage4e_nested_classical", "arm": "classical_nested",
           "git_sha": git_sha(_PROJECT), "env": env_versions(),
           "candidates": list(sets), "selection_counts": dict(Counter(chosen)),
           "aggregate": agg.as_dict(), "bootstrap": boot, "permutation": perm,
           "ordering_correct": int(n_ord), "per_fold": per_fold}
    (_RUN_ROOT / "results" / "stage4e_nested_classical.json").write_text(
        json.dumps(out, indent=2, default=str))
    log.info("Saved -> results/stage4e_nested_classical.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
