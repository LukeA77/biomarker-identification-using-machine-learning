"""Validity checks — is the 0.629 result real, or an artifact of the setup?

Not a search for a bigger number. These are the checks that decide whether the
result should be believed at all. Each one is a way the pipeline could be
producing a significant number for the WRONG reason.

  CHECK 1 -- NEGATIVE CONTROL (the definitive leakage test).
      Re-run the ENTIRE pipeline (scaling, fitting, LOPO, session aggregation)
      with the ON/OFF labels randomly swapped within each patient. Every patient
      still has one A and one C session; only which is which is randomised. If
      there is any leakage or any bug that lets information reach the classifier
      through a route other than the true label, this will score ABOVE chance.
      A correct pipeline must return ~0.500 across repeats.

  CHECK 2 -- TEMPORAL CONFOUND.
      Can the condition be predicted from window TIME ALONE (t_cont, position
      within the session) with no EEG whatsoever? If yes, recording-time structure
      is confounded with condition, and any EEG model could exploit it as a
      shortcut rather than learning physiology.

  CHECK 3 -- SESSION-VOLUME CONFOUND.
      Do ON and OFF sessions differ systematically in length? A model can't see
      duration directly, but recording length covaries with drowsiness/fatigue.
      Also re-runs the classifier on DURATION-MATCHED subsets (equal windows per
      session, taken from the same relative position) to test whether the effect
      survives when volume is equalised.

Run:  python codebrain_finetune/src/stage4f_leakage_checks.py
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
    aggregate_session_level, classification_metrics, permutation_test_balanced_accuracy,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

log = logging.getLogger("stage4f")
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


def lopo_balacc(F, y, pids, conds, mask, folds, fc):
    """Full LOPO pipeline -> pooled session balanced accuracy (+ ordering count)."""
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
    return m.balanced_accuracy, (m.auc if m.auc is not None else float("nan")), n_ord, len(true)


def main() -> int:
    logging.basicConfig(level="INFO", datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dc = load_data_config(); fc = load_finetune_config()
    seed_everything(fc.seed.base, deterministic=fc.seed.deterministic)
    mf = CohortManifest.from_config(dc); folds = build_lopo_folds(mf)
    df = mf.subset(None, nonoverlap_only=False).copy()

    B = np.load(_RUN_ROOT / "results" / "bandpower_feats.npz")["X"]
    gi = BAND_NAMES.index("gamma")
    keep = [i for i in range(5) if i != gi]
    F = B[:, :, keep, 1].reshape(len(B), -1)          # rel_nogamma, the headline set

    y = df["y_cond"].to_numpy().astype(int)
    pids = df["patient_id"].to_numpy().astype(str)
    conds = df["condition"].to_numpy().astype(str)
    t_cont = df["t_cont"].to_numpy().astype(float)
    nonoverlap = df["is_nonoverlap_subset"].to_numpy().astype(bool)
    out = {}

    # ---------------- reference ----------------
    ba, auc, nord, n = lopo_balacc(F, y, pids, conds, nonoverlap, folds, fc)
    log.info("=" * 80)
    log.info("REFERENCE (true labels): bal-acc %.3f  AUC %.3f  ordering %d/35", ba, auc, nord)
    out["reference"] = {"bal_acc": ba, "auc": auc, "ordering": int(nord)}

    # ---------------- CHECK 1: negative control ----------------
    log.info("=" * 80)
    log.info("CHECK 1 — NEGATIVE CONTROL: labels randomly swapped within each patient")
    log.info("  (a correct, leak-free pipeline MUST score ~0.500 here)")
    scores = []
    for rep in range(10):
        rng = np.random.default_rng(1000 + rep)
        y_shuf = y.copy()
        flipped = 0
        for p in np.unique(pids):
            if rng.random() < 0.5:                      # swap this patient's two labels
                m = pids == p
                y_shuf[m] = 1 - y_shuf[m]
                flipped += 1
        ba_s, auc_s, nord_s, _ = lopo_balacc(F, y_shuf, pids, conds, nonoverlap, folds, fc)
        scores.append(ba_s)
        log.info("  rep %2d (%2d/35 patients flipped): bal-acc %.3f  AUC %.3f  ordering %d/35",
                 rep, flipped, ba_s, auc_s, nord_s)
    scores = np.array(scores)
    log.info("  --> mean %.3f  sd %.3f  min %.3f  max %.3f",
             scores.mean(), scores.std(), scores.min(), scores.max())
    verdict = "PASS (centred on chance)" if abs(scores.mean() - 0.5) < 0.06 else "FAIL — investigate"
    log.info("  VERDICT: %s", verdict)
    out["negative_control"] = {"scores": scores.tolist(), "mean": float(scores.mean()),
                               "sd": float(scores.std()), "verdict": verdict}

    # ---------------- CHECK 2: temporal confound ----------------
    log.info("=" * 80)
    log.info("CHECK 2 — TEMPORAL CONFOUND: predict condition from window TIME alone (no EEG)")
    T = np.column_stack([t_cont, t_cont ** 2])
    ba_t, auc_t, nord_t, _ = lopo_balacc(T, y, pids, conds, nonoverlap, folds, fc)
    log.info("  time-only classifier: bal-acc %.3f  AUC %.3f  ordering %d/35", ba_t, auc_t, nord_t)
    log.info("  VERDICT: %s", "clean (time carries no condition info)" if ba_t < 0.58
             else "WARNING — recording time is confounded with condition")
    out["time_only"] = {"bal_acc": ba_t, "auc": auc_t}

    # ---------------- CHECK 3: session-volume confound ----------------
    log.info("=" * 80)
    log.info("CHECK 3 — SESSION VOLUME")
    counts = {}
    for p in np.unique(pids):
        for c in ("A", "C"):
            counts[(p, c)] = int(((pids == p) & (conds == c) & nonoverlap).sum())
    a = np.array([counts[(p, "A")] for p in np.unique(pids)])
    cc = np.array([counts[(p, "C")] for p in np.unique(pids)])
    from scipy.stats import wilcoxon
    w = wilcoxon(a, cc)
    log.info("  non-overlap windows/session: ON median %d, OFF median %d", np.median(a), np.median(cc))
    log.info("  paired Wilcoxon ON vs OFF window count: p=%.3f  (%s)", w.pvalue,
             "no systematic length difference" if w.pvalue > 0.05 else "SESSIONS DIFFER IN LENGTH")
    out["volume"] = {"on_median": int(np.median(a)), "off_median": int(np.median(cc)),
                     "wilcoxon_p": float(w.pvalue)}

    # duration-matched subset: equal windows per session, from the same relative position
    log.info("  duration-matched re-run (equal windows per session):")
    for k in (100, 200, 300):
        mask = np.zeros(len(y), dtype=bool)
        ok = True
        for p in np.unique(pids):
            for c in ("A", "C"):
                idx = np.where((pids == p) & (conds == c) & nonoverlap)[0]
                if len(idx) < k:
                    ok = False
                    break
                order = np.argsort(t_cont[idx])
                mask[idx[order[:k]]] = True
        if not ok:
            log.info("    k=%3d: skipped (some session has <%d non-overlap windows)", k, k)
            continue
        ba_m, auc_m, nord_m, nn = lopo_balacc(F, y, pids, conds, mask, folds, fc)
        log.info("    k=%3d windows/session: bal-acc %.3f  AUC %.3f  ordering %d/35 (n=%d sessions)",
                 k, ba_m, auc_m, nord_m, nn)
        out[f"matched_k{k}"] = {"bal_acc": ba_m, "auc": auc_m, "ordering": int(nord_m)}

    (_RUN_ROOT / "results" / "stage4f_leakage_checks.json").write_text(json.dumps(out, indent=2, default=str))
    log.info("=" * 80)
    log.info("Saved -> results/stage4f_leakage_checks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
