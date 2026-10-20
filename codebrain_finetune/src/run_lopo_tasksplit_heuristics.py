"""No-EEG heuristic baselines for the LOPO + task-split design (N = 27).

WHY THIS EXISTS
----------------
``run_lopo_tasksplit.py`` reports six arms (frozen probe, LoRA, full fine-tune,
channel amplitude, relative band power, aperiodic exponent) on the 27-patient
task-split cohort. The two no-EEG heuristic floors used elsewhere in the
dissertation (window-count / "duration-only", and elapsed-time) were computed
only on the 35-patient primary LOPO cohort (``stage4f_leakage_checks.py``,
``stage4g_duration_confound.py``) with no task-split restriction — confirmed by
reading both scripts, which hardcode ``/35`` in their own log strings and call
``build_lopo_folds`` over the full cohort with no task masking. Comparing those
N = 35 numbers against the N = 27 task-split arms would not be a fair floor, so
this script recomputes both heuristics on the SAME 27-patient, task-split,
LOPO fold structure that the six reported arms use, so all eight numbers are
directly comparable.

Two heuristics, each restricted to the ``eligible_patients`` task-split cohort
and pooled over both directions (108 sessions, matching the six-arm design):

  WINDOW-COUNT (session-level, no EEG whatsoever)
      Feature = the number of non-overlap (eval) windows in a session's test-task
      span. One number per session. LOPO logistic regression over patients.

  ELAPSED-TIME (window-level, no EEG whatsoever)
      Feature = a window's elapsed time within its own recording (``t_cont``,
      plus its square), the same feature ``stage4f_leakage_checks.py`` CHECK 2
      used on the N = 35 cohort. LOPO logistic regression at window level,
      aggregated to session level exactly as the frozen-probe arm is.

Neither heuristic touches the EEG signal or the CodeBrain backbone. Both reuse
the identical train/test fold construction as ``run_lopo_tasksplit.py``'s
``_fold_frames`` (train on OTHER patients' train-task windows, test on the
held-out patient's test-task windows) and the identical aggregation/statistics
(``codebrain_ft.metrics``, patient-cluster bootstrap, 2,000 resamples) so the
output is a drop-in comparison against ``lopo_tasksplit_none_summary.json``.

This script only READS the existing task-split sidecar and preflight files and
WRITES one new file (``results/lopo_tasksplit_none/heuristics_summary.json``).
It does not modify any existing results file, script, or run.

Run:  python codebrain_finetune/src/run_lopo_tasksplit_heuristics.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent          # codebrain_finetune/
sys.path.insert(0, str(_RUN_ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from codebrain_ft.ft_config import load_finetune_config  # noqa: E402
from codebrain_ft.seeding import fold_seed  # noqa: E402
from codebrain_ft.task_windows import load_tasksplit_config, load_sidecar  # noqa: E402
from codebrain_ft.tasksplit import select_task_frame, direction_pairs  # noqa: E402
from codebrain_ft.metrics import (  # noqa: E402
    aggregate_session_level, classification_metrics, bootstrap_session_ci,
    permutation_test_balanced_accuracy,
)
from codebrain_ft.tracking import git_sha, env_versions  # noqa: E402

log = logging.getLogger("lopo_tasksplit_heuristics")

_PROJECT = _RUN_ROOT.parent
HEURISTICS = ("window_count", "elapsed_time")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _fold_frames(sidecar: pd.DataFrame, held_out: str, train_patients: list[str],
                 train_task: str, test_task: str, ts_cfg):
    """(train, test) frames, identical construction to run_lopo_tasksplit.py's
    ``_fold_frames``: other patients' train-task windows -> held-out patient's
    test-task windows. Re-implemented here (not imported) to keep this script
    free of that module's heavier, torch-importing dependency chain.
    """
    proto = ts_cfg.protocol
    tr = sidecar[(sidecar["patient_id"].isin(train_patients)) &
                 (sidecar["task_group"] == train_task)]
    if bool(proto["train_nonoverlap_only"]):
        tr = tr[tr["is_nonoverlap_subset"]]
    tr = tr.reset_index(drop=True)
    te = select_task_frame(sidecar, held_out, test_task,
                           nonoverlap_only=bool(proto["eval_nonoverlap_only"]))
    if held_out in set(tr["patient_id"]):
        raise RuntimeError(f"fold {held_out}: held-out patient leaked into the training frame")
    overlap = set(tr["path"]) & set(te["path"])
    if overlap:
        raise RuntimeError(f"fold {held_out}: {len(overlap)} windows in both train and test")
    if len(tr) == 0:
        raise ValueError(f"fold {held_out}: no training windows for task {train_task}")
    return tr, te


def _window_count_session_scores(sidecar: pd.DataFrame, eligible: list[str], test_task: str):
    """One row per (patient, condition): number of eval windows of ``test_task``.

    No EEG, no per-fold feature extraction needed — the feature is a single
    count that already exists at session granularity. LOPO logistic regression
    over this table is the direct task-split analogue of
    ``stage4g_duration_confound.py``'s duration-only check.
    """
    ev = sidecar[(sidecar["patient_id"].isin(eligible)) &
                 (sidecar["task_group"] == test_task) &
                 (sidecar["is_nonoverlap_subset"])]
    tbl = (ev.groupby(["patient_id", "condition"])
             .agg(count=("path", "size"), y_cond=("y_cond", "first"))
             .reset_index())
    missing = set(eligible) - set(tbl["patient_id"])
    if missing:
        raise RuntimeError(f"window_count/{test_task}: no eval windows for {sorted(missing)}")
    counts_per_patient = tbl.groupby("patient_id").size()
    bad = counts_per_patient[counts_per_patient != 2]
    if len(bad):
        raise RuntimeError(f"window_count/{test_task}: patient(s) without exactly 2 sessions: "
                           f"{bad.to_dict()}")

    X = tbl["count"].to_numpy(dtype=float).reshape(-1, 1)
    y = tbl["y_cond"].to_numpy().astype(int)
    pid = tbl["patient_id"].to_numpy().astype(str)

    prob = np.full(len(tbl), np.nan)
    for i, held_out in enumerate(eligible):
        tr = pid != held_out
        te = pid == held_out
        seed = fold_seed(1, i)
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=seed).fit(sc.transform(X[tr]), y[tr])
        prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    assert np.isfinite(prob).all()
    return pid, tbl["condition"].to_numpy().astype(str), prob, y


def _elapsed_time_session_scores(sidecar: pd.DataFrame, eligible: list[str], ts_cfg,
                                 train_task: str, test_task: str):
    """LOPO over folds, feature = [t_cont, t_cont**2] of each window.

    Identical fold construction to the six reported arms (train on other
    patients' train-task windows, test on the held-out patient's test-task
    windows); the only difference is the feature matrix, which here is a
    2-column function of window elapsed time, carrying no EEG whatsoever. This
    is the task-split analogue of ``stage4f_leakage_checks.py`` CHECK 2.
    """
    all_prob, all_true, all_pid, all_cond = [], [], [], []
    for i, held_out in enumerate(eligible):
        train_patients = [p for p in eligible if p != held_out]
        tr, te = _fold_frames(sidecar, held_out, train_patients, train_task, test_task, ts_cfg)
        Xtr = np.column_stack([tr["t_cont"].to_numpy(dtype=float),
                               tr["t_cont"].to_numpy(dtype=float) ** 2])
        Xte = np.column_stack([te["t_cont"].to_numpy(dtype=float),
                               te["t_cont"].to_numpy(dtype=float) ** 2])
        ytr = tr["y_cond"].to_numpy().astype(int)
        yte = te["y_cond"].to_numpy().astype(int)
        seed = fold_seed(1, i)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=seed).fit(sc.transform(Xtr), ytr)
        prob = clf.predict_proba(sc.transform(Xte))[:, 1]

        sess = aggregate_session_level(te["patient_id"].to_numpy(), te["condition"].to_numpy(),
                                       prob, yte, agg="mean_prob")
        all_prob.append(sess.prob); all_true.append(sess.y_true)
        all_pid.append(sess.patients); all_cond.append([s.split("|")[1] for s in sess.sessions])

    return (np.concatenate(all_pid), np.concatenate(all_cond),
            np.concatenate(all_prob), np.concatenate(all_true))


def _summarise(name: str, direction: str, pid, cond, prob, y_true, ft_cfg) -> dict:
    agg = classification_metrics(y_true, prob, threshold=ft_cfg.eval.threshold)
    boot = bootstrap_session_ci(y_true, prob, pid, n_resamples=ft_cfg.eval.bootstrap.n_resamples,
                                ci=ft_cfg.eval.bootstrap.ci, seed=ft_cfg.seed.base)
    perm = permutation_test_balanced_accuracy(
        y_true, prob, pid, n_permutations=ft_cfg.eval.permutation.n_permutations,
        seed_base=ft_cfg.seed.base + ft_cfg.seed.permutation_offset)
    return {
        "arm": name, "direction": direction, "n_sessions": int(len(y_true)),
        "aggregate": agg.as_dict(), "bootstrap": boot, "permutation": perm,
    }


def main() -> int:
    logging.basicConfig(level="INFO", datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ts_cfg = load_tasksplit_config(_RUN_ROOT)
    ft_cfg = load_finetune_config()

    pf = json.loads((_RUN_ROOT / "results" / ts_cfg.output["results_subdir"] /
                     "preflight_summary.json").read_text())
    eligible = list(pf["eligible_patients"])
    log.info("Eligible patients: %d (task-split cohort, from preflight_summary.json)", len(eligible))

    sidecar = load_sidecar(_RUN_ROOT / ts_cfg.output["sidecar"])
    sidecar = sidecar[sidecar["patient_id"].isin(eligible)].reset_index(drop=True)
    pairs = direction_pairs(ts_cfg.group_names, ts_cfg.protocol["directions"])

    arms: dict[str, dict] = {}

    # ---------------- window-count heuristic ----------------
    per_direction = {}
    for train_task, test_task in pairs:
        direction = f"{train_task}_to_{test_task}"
        pid, cond, prob, y_true = _window_count_session_scores(sidecar, eligible, test_task)
        s = _summarise("window_count", direction, pid, cond, prob, y_true, ft_cfg)
        arms[f"window_count|{direction}"] = s
        per_direction[direction] = (pid, cond, prob, y_true)
        log.info("window_count  %-14s | n=%3d | bal-acc %.3f [%.3f, %.3f]", direction,
                 s["n_sessions"], s["aggregate"]["balanced_accuracy"],
                 s["bootstrap"]["ci_low"], s["bootstrap"]["ci_high"])
    pid = np.concatenate([per_direction[f"{a}_to_{b}"][0] for a, b in pairs])
    cond = np.concatenate([per_direction[f"{a}_to_{b}"][1] for a, b in pairs])
    prob = np.concatenate([per_direction[f"{a}_to_{b}"][2] for a, b in pairs])
    y_true = np.concatenate([per_direction[f"{a}_to_{b}"][3] for a, b in pairs])
    s = _summarise("window_count", "both_directions_pooled", pid, cond, prob, y_true, ft_cfg)
    arms["window_count|both_directions_pooled"] = s
    log.info("window_count  %-14s | n=%3d | bal-acc %.3f [%.3f, %.3f]", "pooled",
             s["n_sessions"], s["aggregate"]["balanced_accuracy"],
             s["bootstrap"]["ci_low"], s["bootstrap"]["ci_high"])

    # ---------------- elapsed-time heuristic ----------------
    per_direction = {}
    for train_task, test_task in pairs:
        direction = f"{train_task}_to_{test_task}"
        pid, cond, prob, y_true = _elapsed_time_session_scores(sidecar, eligible, ts_cfg,
                                                                train_task, test_task)
        s = _summarise("elapsed_time", direction, pid, cond, prob, y_true, ft_cfg)
        arms[f"elapsed_time|{direction}"] = s
        per_direction[direction] = (pid, cond, prob, y_true)
        log.info("elapsed_time  %-14s | n=%3d | bal-acc %.3f [%.3f, %.3f]", direction,
                 s["n_sessions"], s["aggregate"]["balanced_accuracy"],
                 s["bootstrap"]["ci_low"], s["bootstrap"]["ci_high"])
    pid = np.concatenate([per_direction[f"{a}_to_{b}"][0] for a, b in pairs])
    cond = np.concatenate([per_direction[f"{a}_to_{b}"][1] for a, b in pairs])
    prob = np.concatenate([per_direction[f"{a}_to_{b}"][2] for a, b in pairs])
    y_true = np.concatenate([per_direction[f"{a}_to_{b}"][3] for a, b in pairs])
    s = _summarise("elapsed_time", "both_directions_pooled", pid, cond, prob, y_true, ft_cfg)
    arms["elapsed_time|both_directions_pooled"] = s
    log.info("elapsed_time  %-14s | n=%3d | bal-acc %.3f [%.3f, %.3f]", "pooled",
             s["n_sessions"], s["aggregate"]["balanced_accuracy"],
             s["bootstrap"]["ci_low"], s["bootstrap"]["ci_high"])

    out = {
        "run_tag": "lopo_tasksplit_heuristics", "input_norm": "none (no EEG features used)",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("Task-split-restricted, N=27 recomputation of the two no-EEG heuristic "
                 "floors reported at N=35 in stage4f_leakage_checks.json (elapsed_time, there "
                 "'time_only') and stage4g_duration_confound.json (window_count, there "
                 "'duration_only'). Same LOPO+task-split fold structure and statistics "
                 "(patient-cluster bootstrap, 2000 resamples) as lopo_tasksplit_none_summary.json, "
                 "so directly comparable to those six arms."),
        "arms": arms,
        "eligible_patients": eligible,
        "git_sha": git_sha(_PROJECT), "env": env_versions(),
    }
    out_path = _RUN_ROOT / "results" / "lopo_tasksplit_none" / "heuristics_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=_json_default), encoding="utf-8")
    log.info("Saved -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
