"""Stage 4 — frozen linear-probe baseline across ALL 35 LOPO folds.

The honest floor LoRA must beat: freeze the whole CodeBrain backbone, extract
per-window features ONCE for the entire cohort (cached in RAM), then run the
LOPO loop in feature space — fit a linear classifier on the training patients'
windows, predict the held-out patient's non-overlap windows, aggregate to
session level. Cheap (one forward pass over the cache, then sklearn), so it runs
in minutes and tells us whether the representation carries transferable ON/OFF
signal before we invest in gradient-based fine-tuning.

Run:  python codebrain_finetune/src/stage4_probe_lopo.py [--pooling mean|flatten]
"""
from __future__ import annotations

import argparse
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

from livinglab_codebrain import load_data_config, CohortManifest, build_lopo_folds  # noqa: E402
from livinglab_codebrain.dataset import WindowDataset  # noqa: E402
from livinglab_codebrain.loader import make_dataloader  # noqa: E402
from codebrain_ft import load_finetune_config, build_classifier_model, seed_everything  # noqa: E402
from codebrain_ft.probe import extract_features  # noqa: E402
from codebrain_ft.metrics import (  # noqa: E402
    aggregate_session_level, classification_metrics,
    bootstrap_session_ci, permutation_test_balanced_accuracy,
)
from codebrain_ft.tracking import git_sha, file_sha256, env_versions, save_run  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

log = logging.getLogger("stage4")

EXPECTED_PATIENTS, EXPECTED_WINDOWS = 35, 37174


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4 frozen-probe LOPO baseline")
    ap.add_argument("--pooling", default=None, choices=["mean", "flatten"],
                    help="override probe.pooling from config")
    ap.add_argument("--C", type=float, default=None, help="override logreg C")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    import torch
    if not torch.cuda.is_available():
        log.error("CUDA unavailable — SSSM requires GPU. Aborting.")
        return 1
    device = "cuda"

    data_cfg = load_data_config()
    ft_cfg = load_finetune_config()
    if args.pooling:  # allow a quick pooling sweep without editing yaml
        import dataclasses
        ft_cfg = dataclasses.replace(ft_cfg, probe=dataclasses.replace(ft_cfg.probe, pooling=args.pooling))
    C = args.C if args.C is not None else ft_cfg.probe.C
    seed_everything(ft_cfg.seed.base, deterministic=ft_cfg.seed.deterministic)

    manifest = CohortManifest.from_config(data_cfg)
    n_pat, n_win = len(manifest.patients), manifest.n_windows
    log.info("Cohort: %d patients, %d windows, balance=%s", n_pat, n_win, manifest.label_balance())
    if n_pat != EXPECTED_PATIENTS or n_win != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch — STOP. got (%d,%d) expected (%d,%d)",
                  n_pat, n_win, EXPECTED_PATIENTS, EXPECTED_WINDOWS)
        return 1

    # ---- extract features ONCE for the entire cohort ----
    log.info("Extracting frozen features for all %d windows (pooling=%s)...", n_win, ft_cfg.probe.pooling)
    model = build_classifier_model(data_cfg, ft_cfg, mode="frozen", device=device)
    full_df = manifest.subset(None, nonoverlap_only=False).copy()
    ds = WindowDataset(full_df, data_cfg.paths.cache_dir, data_cfg.window)
    loader = make_dataloader(ds, data_cfg, shuffle=False, seed=ft_cfg.seed.base)
    t0 = time.time()
    X, y, pids, conds = extract_features(model, loader, ft_cfg, device=device)
    log.info("Features %s extracted in %.0fs", X.shape, time.time() - t0)
    del model
    torch.cuda.empty_cache()

    # WindowDataset does not shuffle here, so row i of X aligns with full_df row i.
    nonoverlap = full_df["is_nonoverlap_subset"].to_numpy().astype(bool)
    pids = np.asarray(pids); conds = np.asarray(conds); y = np.asarray(y)

    # ---- LOPO in feature space ----
    folds = build_lopo_folds(manifest)
    all_prob, all_true, all_pat, all_sess = [], [], [], []
    per_fold = []
    for fold in folds:
        tr = ~np.isin(pids, [fold.test_patient])
        te = (pids == fold.test_patient) & nonoverlap
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=ft_cfg.probe.max_iter, class_weight="balanced")
        clf.fit(scaler.transform(X[tr]), y[tr])
        prob_te = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        sess = aggregate_session_level(pids[te], conds[te], prob_te, y[te])
        fm = classification_metrics(sess.y_true, sess.prob, threshold=ft_cfg.eval.threshold)
        per_fold.append({
            "fold": fold.index, "test_patient": fold.test_patient,
            "n_test_windows": int(te.sum()),
            "sessions": sess.sessions.tolist(), "prob": sess.prob.tolist(),
            "y_true": sess.y_true.tolist(),
        })
        all_prob.extend(sess.prob.tolist()); all_true.extend(sess.y_true.tolist())
        all_pat.extend(sess.patients.tolist()); all_sess.extend(sess.sessions.tolist())
        log.info("fold %2d %-9s | sessions=%s probs=%s true=%s",
                 fold.index, fold.test_patient, sess.sessions.tolist(),
                 [round(p, 3) for p in sess.prob.tolist()], sess.y_true.tolist())

    all_prob = np.asarray(all_prob); all_true = np.asarray(all_true); all_pat = np.asarray(all_pat)

    # ---- pooled session-level metrics across the whole cohort (70 points) ----
    agg = classification_metrics(all_true, all_prob, threshold=ft_cfg.eval.threshold)
    boot = bootstrap_session_ci(all_true, all_prob, all_pat,
                                n_resamples=ft_cfg.eval.bootstrap.n_resamples,
                                ci=ft_cfg.eval.bootstrap.ci, seed=ft_cfg.seed.base)
    perm = permutation_test_balanced_accuracy(
        all_true, all_prob, all_pat,
        n_permutations=ft_cfg.eval.permutation.n_permutations,
        seed_base=ft_cfg.seed.base + ft_cfg.seed.permutation_offset)

    log.info("=" * 70)
    log.info("STAGE 4 FROZEN PROBE — pooled over 35 folds / %d sessions", len(all_true))
    log.info("  balanced_accuracy = %.3f  (bootstrap %.0f%% CI [%.3f, %.3f])",
             agg.balanced_accuracy, boot["ci"] * 100, boot["ci_low"], boot["ci_high"])
    log.info("  AUC=%s  sens(OFF)=%s  spec(ON)=%s  macro_f1=%.3f  acc=%.3f",
             f"{agg.auc:.3f}" if agg.auc is not None else "n/a",
             f"{agg.sensitivity:.3f}" if agg.sensitivity is not None else "n/a",
             f"{agg.specificity:.3f}" if agg.specificity is not None else "n/a",
             agg.macro_f1, agg.accuracy)
    log.info("  permutation p-value = %.4f (null mean %.3f)", perm["p_value"], perm["null_mean"])
    log.info("=" * 70)

    record = {
        "run_tag": f"stage4_probe_lopo_{ft_cfg.probe.pooling}", "arm": "probe_lopo",
        "git_sha": git_sha(_PROJECT), "checkpoint_sha256": file_sha256(data_cfg.paths.checkpoint),
        "env": env_versions(), "pooling": ft_cfg.probe.pooling, "C": C,
        "n_sessions": len(all_true),
        "aggregate": agg.as_dict(), "bootstrap": boot, "permutation": perm,
        "per_fold": per_fold,
    }
    save_run(record, ft_cfg.tracking.results_dir, ft_cfg.tracking.master_csv,
             csv_fields=["run_tag", "arm", "git_sha"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
