"""Follow-up validity checks — the session-DURATION confound, done properly.

Experiment 4f raised two red flags that its own thresholds were too lenient to
catch:

  * a window-TIME-only classifier (no EEG at all) reached bal-acc 0.571;
  * ON and OFF sessions differ significantly in length (ON median 288 vs OFF
    median 239 non-overlap windows, paired Wilcoxon p=0.044);
  * and the decisive duration-matched re-run was SKIPPED because k was set above
    the smallest session (98 windows).

If recording duration differs by condition, then session-level aggregation (the
MEAN prediction over a session's windows) can encode duration, and part of the
0.657 could be driven by how long the recording was rather than by its EEG
content. This script settles it:

  1. Can session DURATION ALONE (one number per session, no EEG) classify ON/OFF?
  2. Does the EEG result survive DURATION MATCHING -- exactly k windows per
     session, so every session contributes identically?
  3. A proper NULL DISTRIBUTION from 100 full-pipeline refits on within-patient
     shuffled labels. The post-hoc permutation test does not refit the
     classifier, so it may understate the null; this measures it directly.

Run:  python codebrain_finetune/src/stage4g_duration_confound.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_RUN_ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from livinglab_codebrain import load_data_config, CohortManifest, build_lopo_folds  # noqa: E402
from codebrain_ft import load_finetune_config, seed_everything  # noqa: E402
from codebrain_ft.metrics import aggregate_session_level, classification_metrics  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

log = logging.getLogger("stage4g")
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


def lopo_balacc(F, y, pids, conds, mask, folds, fc):
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
    return m.balanced_accuracy, (m.auc if m.auc is not None else float("nan")), int(n_ord)


def main() -> int:
    logging.basicConfig(level="INFO", datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dc = load_data_config(); fc = load_finetune_config()
    seed_everything(fc.seed.base, deterministic=fc.seed.deterministic)
    mf = CohortManifest.from_config(dc); folds = build_lopo_folds(mf)
    df = mf.subset(None, nonoverlap_only=False).copy()

    B = np.load(_RUN_ROOT / "results" / "bandpower_feats.npz")["X"]
    gi = BAND_NAMES.index("gamma")
    F = B[:, :, [i for i in range(5) if i != gi], 1].reshape(len(B), -1)

    y = df["y_cond"].to_numpy().astype(int)
    pids = df["patient_id"].to_numpy().astype(str)
    conds = df["condition"].to_numpy().astype(str)
    t_cont = df["t_cont"].to_numpy().astype(float)
    no = df["is_nonoverlap_subset"].to_numpy().astype(bool)
    pats = np.unique(pids)
    out = {}

    # ---- 1. duration alone, at SESSION level -----------------------------
    log.info("=" * 84)
    log.info("CHECK A — can session DURATION ALONE classify ON/OFF? (no EEG whatsoever)")
    dur, lab, who = [], [], []
    for p in pats:
        for c, l in (("A", 0), ("C", 1)):
            m = (pids == p) & (conds == c) & no
            dur.append(m.sum()); lab.append(l); who.append(p)
    dur = np.array(dur, float); lab = np.array(lab); who = np.array(who)
    # LOPO at session level on the single duration feature
    preds = []
    for p in pats:
        tr = who != p; te = who == p
        sc = StandardScaler().fit(dur[tr].reshape(-1, 1))
        clf = LogisticRegression(class_weight="balanced").fit(sc.transform(dur[tr].reshape(-1, 1)), lab[tr])
        preds.append(clf.predict_proba(sc.transform(dur[te].reshape(-1, 1)))[:, 1])
    pr = np.concatenate(preds)
    order = np.concatenate([np.where(who == p)[0] for p in pats])
    md = classification_metrics(lab[order], pr, threshold=0.5)
    n_ord_d = sum(pr[np.where(who[order] == p)[0]][1] > pr[np.where(who[order] == p)[0]][0] for p in pats)
    log.info("  duration-only classifier: bal-acc %.3f  AUC %.3f  ordering %d/35",
             md.balanced_accuracy, md.auc if md.auc else float("nan"), n_ord_d)
    log.info("  ON windows median %d | OFF windows median %d",
             np.median(dur[lab == 0]), np.median(dur[lab == 1]))
    out["duration_only"] = {"bal_acc": md.balanced_accuracy, "auc": md.auc,
                            "ordering": int(n_ord_d)}

    # ---- 2. duration-MATCHED EEG re-run ----------------------------------
    log.info("=" * 84)
    min_sess = int(min(((pids == p) & (conds == c) & no).sum() for p in pats for c in "AC"))
    log.info("CHECK B — DURATION-MATCHED re-run (smallest session has %d non-overlap windows)", min_sess)
    ref_ba, ref_auc, ref_ord = lopo_balacc(F, y, pids, conds, no, folds, fc)
    log.info("  unmatched reference: bal-acc %.3f  AUC %.3f  ordering %d/35", ref_ba, ref_auc, ref_ord)
    out["reference"] = {"bal_acc": ref_ba, "auc": ref_auc, "ordering": ref_ord}
    for k in (50, 75, min_sess):
        for where in ("first", "middle"):
            mask = np.zeros(len(y), bool)
            for p in pats:
                for c in "AC":
                    idx = np.where((pids == p) & (conds == c) & no)[0]
                    idx = idx[np.argsort(t_cont[idx])]
                    sel = idx[:k] if where == "first" else idx[(len(idx) - k) // 2:(len(idx) - k) // 2 + k]
                    mask[sel] = True
            ba, auc, nord = lopo_balacc(F, y, pids, conds, mask, folds, fc)
            log.info("  k=%3d (%-6s): bal-acc %.3f  AUC %.3f  ordering %2d/35", k, where, ba, auc, nord)
            out[f"matched_k{k}_{where}"] = {"bal_acc": ba, "auc": auc, "ordering": nord}

    # ---- 3. proper null distribution (100 full refits) -------------------
    log.info("=" * 84)
    log.info("CHECK C — NULL DISTRIBUTION from 100 full-pipeline refits (labels shuffled within patient)")
    log.info("  (the post-hoc permutation test does NOT refit the classifier, so it may understate this)")
    null = []
    for rep in range(100):
        rng = np.random.default_rng(5000 + rep)
        ys = y.copy()
        for p in pats:
            if rng.random() < 0.5:
                m = pids == p
                ys[m] = 1 - ys[m]
        ba, _, _ = lopo_balacc(F, ys, pids, conds, no, folds, fc)
        null.append(ba)
        if (rep + 1) % 25 == 0:
            log.info("  %d/100 done (running mean %.3f)", rep + 1, np.mean(null))
    null = np.array(null)
    p_emp = (np.sum(null >= ref_ba) + 1) / (len(null) + 1)
    log.info("  null: mean %.3f  sd %.3f  95th pct %.3f  max %.3f",
             null.mean(), null.std(), np.percentile(null, 95), null.max())
    log.info("  observed %.3f  ->  EMPIRICAL p = %.4f  (refit-based)", ref_ba, p_emp)
    log.info("  compare: post-hoc permutation test reported p = 0.005")
    out["null_refit"] = {"mean": float(null.mean()), "sd": float(null.std()),
                         "p95": float(np.percentile(null, 95)), "max": float(null.max()),
                         "observed": ref_ba, "p_empirical": float(p_emp),
                         "null": null.tolist()}

    (_RUN_ROOT / "results" / "stage4g_duration_confound.json").write_text(json.dumps(out, indent=2, default=str))
    log.info("=" * 84)
    log.info("Saved -> results/stage4g_duration_confound.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
