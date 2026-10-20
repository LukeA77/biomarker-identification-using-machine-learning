"""Window-count distribution per (patient, condition, task) — duration-confound audit.

Motivation (supervisor meeting, 2026-08-06): in the pooled task-split design the
ON and OFF sessions of a patient contribute DIFFERENT numbers of windows, because
the same ADL task takes a different amount of time in the two medication states.
Recording length alone already scores 0.614 bal-acc at session level with no EEG
(EXPERIMENTS.md 10c-10d), so an unmatched design leaves a non-neural route to the
label open. The agreed fix is to force a FIXED number of windows per session, but
the level at which the count is fixed (global / per task / per patient / per
patient x task) has to be chosen from the actual distribution rather than guessed.

This script only MEASURES. It does not modify the cache, the manifest, the
sidecar, or any experiment config; nothing here changes a result. It reads the
read-only task sidecar written by ``derive_task_windows.py`` and reports:

  1. the per-cell window counts (cell = patient x condition x task group), at both
     the TRAIN convention (all windows, 5 s stride, overlapping) and the EVAL
     convention (``is_nonoverlap_subset``, 10 s stride) that the task-split
     drivers actually use;
  2. how asymmetric ON vs OFF is within each patient x task, and whether that
     asymmetry is systematic enough to carry the label (sign test + the
     leave-one-out "count alone" classifier that formalises the supervisor's
     worry);
  3. what a fixed-K rule costs at each candidate level: the largest feasible K,
     how many windows survive, and how many patients would be dropped if K were
     pushed above the per-cell minimum.

Usage:
    python codebrain_finetune/src/analyze_window_counts.py
    python codebrain_finetune/src/analyze_window_counts.py --report-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent

SIDECAR = _RUN_ROOT / "results" / "within_patient_tasksplit" / "task_windows.parquet"
PREFLIGHT = _RUN_ROOT / "results" / "within_patient_tasksplit" / "preflight_summary.json"
OUT_DIR = _RUN_ROOT / "results" / "window_counts"

# Verified label map (livinglab_codebrain/dataset.py): condition A = ON = 0,
# condition C = OFF = 1.
COND_NAME = {"A": "ON", "C": "OFF"}
WINDOW_S = 10.0          # window length
EVAL_STRIDE_S = 10.0     # non-overlap subset stride
TASKS = ["tea", "toast"]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def load_cells() -> pd.DataFrame:
    """One row per (patient, condition, task) with train- and eval-convention counts."""
    sc = pd.read_parquet(SIDECAR)
    eligible = list(json.loads(PREFLIGHT.read_text(encoding="utf-8"))["eligible_patients"])
    sc = sc[sc["patient_id"].isin(eligible)].copy()

    n_all = sc.groupby(["patient_id", "condition", "task_group"]).size().rename("n_all")
    n_eval = (sc[sc["is_nonoverlap_subset"]]
              .groupby(["patient_id", "condition", "task_group"]).size().rename("n_eval"))
    # span of the task inside the recording, from the window timestamps
    span = sc.groupby(["patient_id", "condition", "task_group"])["t_cont"].agg(["min", "max"])

    cells = (pd.concat([n_all, n_eval], axis=1)
             .join(span)
             .reset_index()
             .rename(columns={"min": "t_first", "max": "t_last"}))
    cells["n_eval"] = cells["n_eval"].fillna(0).astype(int)
    cells["state"] = cells["condition"].map(COND_NAME)
    # Task-covered time implied by the non-overlap subset. This is the honest
    # "how long did this task take" proxy: the eval windows tile the task span
    # without overlap, so K windows == K * 10 s of retained recording.
    cells["covered_s"] = cells["n_eval"] * EVAL_STRIDE_S
    return cells.sort_values(["patient_id", "task_group", "condition"]).reset_index(drop=True)


def pair_table(cells: pd.DataFrame, col: str) -> pd.DataFrame:
    """Wide ON/OFF table per (patient, task) with the asymmetry columns."""
    w = cells.pivot_table(index=["patient_id", "task_group"], columns="state",
                          values=col, aggfunc="first").reset_index()
    w["diff_off_minus_on"] = w["OFF"] - w["ON"]
    w["ratio_off_over_on"] = w["OFF"] / w["ON"]
    w["min_pair"] = w[["ON", "OFF"]].min(axis=1)
    w["imbalance_frac"] = (w["OFF"] - w["ON"]).abs() / w[["ON", "OFF"]].max(axis=1)
    return w


# ---------------------------------------------------------------------------
# is the count itself a label?  (the supervisor's concern, quantified)
# ---------------------------------------------------------------------------
def count_alone_classifier(pairs: pd.DataFrame) -> dict:
    """How well does 'which session has more windows' alone predict OFF?

    Two readouts:
      * ordering  — per patient x task, is OFF's count > ON's count? A sign test
        against 50/50 says whether the asymmetry has a consistent DIRECTION,
        which is the only way a count can be a usable cue at all.
      * loo_threshold — a leave-one-patient-out classifier on the single feature
        log(count), trained on the other patients. This is the pooled analogue:
        it asks whether an absolute count threshold learned from other patients
        transfers, which is exactly what a pooled model could exploit.
    """
    out: dict = {}
    from scipy import stats as sps

    for task in TASKS:
        sub = pairs[pairs["task_group"] == task]
        n = len(sub)
        n_off_longer = int((sub["diff_off_minus_on"] > 0).sum())
        n_tie = int((sub["diff_off_minus_on"] == 0).sum())
        # exact two-sided binomial sign test, ties excluded
        n_eff = n - n_tie
        p_sign = float(sps.binomtest(n_off_longer, n_eff, 0.5).pvalue) if n_eff else float("nan")

        # within-patient ordering rule: predict OFF = the longer session.
        # balanced accuracy of that rule over the 2n sessions.
        ba_within = (n_off_longer + 0.5 * n_tie) / n if n else float("nan")

        # leave-one-patient-out absolute-threshold rule on log(count)
        long_rows = []
        for _, r in sub.iterrows():
            long_rows.append((r["patient_id"], 0, r["ON"]))
            long_rows.append((r["patient_id"], 1, r["OFF"]))
        L = pd.DataFrame(long_rows, columns=["patient_id", "y", "count"])
        correct, total = 0, 0
        per_class = {0: [0, 0], 1: [0, 0]}
        for pid in L["patient_id"].unique():
            tr = L[L["patient_id"] != pid]
            te = L[L["patient_id"] == pid]
            # best single threshold on the training patients, maximising bal-acc
            cand = np.unique(tr["count"].to_numpy())
            best_t, best_s, best_sign = None, -1.0, 1
            for t in cand:
                for sign in (1, -1):
                    pred = ((tr["count"].to_numpy() - t) * sign >= 0).astype(int)
                    ytr = tr["y"].to_numpy()
                    s = 0.5 * (((pred == 1) & (ytr == 1)).sum() / max((ytr == 1).sum(), 1)
                               + ((pred == 0) & (ytr == 0)).sum() / max((ytr == 0).sum(), 1))
                    if s > best_s:
                        best_t, best_s, best_sign = t, s, sign
            pred = ((te["count"].to_numpy() - best_t) * best_sign >= 0).astype(int)
            for p, y in zip(pred, te["y"].to_numpy()):
                per_class[int(y)][1] += 1
                if p == y:
                    per_class[int(y)][0] += 1
                    correct += 1
                total += 1
        ba_loo = 0.5 * (per_class[0][0] / max(per_class[0][1], 1)
                        + per_class[1][0] / max(per_class[1][1], 1))

        out[task] = {
            "n_patients": n,
            "n_off_longer": n_off_longer,
            "n_on_longer": int((sub["diff_off_minus_on"] < 0).sum()),
            "n_tie": n_tie,
            "sign_test_p": p_sign,
            "within_patient_longer_is_off_bal_acc": float(ba_within),
            "loo_absolute_threshold_bal_acc": float(ba_loo),
            "median_diff_off_minus_on": float(sub["diff_off_minus_on"].median()),
            "median_ratio_off_over_on": float(sub["ratio_off_over_on"].median()),
            "median_imbalance_frac": float(sub["imbalance_frac"].median()),
        }
    return out


# ---------------------------------------------------------------------------
# what does a fixed-K rule cost, at each level?
# ---------------------------------------------------------------------------
def matching_options(cells: pd.DataFrame, col: str = "n_eval") -> dict:
    """Feasible K and retained-window budget for each level a K could be fixed at.

    Levels, from strictest (most data thrown away, most confound removed) to
    loosest:
      global          — one K for every cell in the study.
      per_task        — K_tea, K_toast; a task's own typical length is respected.
      per_patient     — K_p, one number for all four of a patient's cells.
      per_patient_task— K_(p,task) = min(ON, OFF) for that patient and task.
                        The minimal intervention: it equalises exactly the ON/OFF
                        pair that carries the confound and nothing else.
    """
    piv = cells.pivot_table(index=["patient_id", "task_group"], columns="state",
                            values=col, aggfunc="first")
    piv["min_pair"] = piv[["ON", "OFF"]].min(axis=1)
    total = int(cells[col].sum())

    opts: dict = {}

    # --- global -----------------------------------------------------------
    k_glob = int(cells[col].min())
    opts["global"] = {
        "k": {"all": k_glob},
        "n_cells": int(len(cells)),
        "kept": k_glob * len(cells),
        "retained_frac": k_glob * len(cells) / total,
        "binding_cell": cells.loc[cells[col].idxmin(),
                                  ["patient_id", "condition", "task_group"]].to_dict(),
    }

    # --- per task ---------------------------------------------------------
    kt = {t: int(cells.loc[cells.task_group == t, col].min()) for t in TASKS}
    kept = sum(kt[t] * int((cells.task_group == t).sum()) for t in TASKS)
    opts["per_task"] = {"k": kt, "kept": kept, "retained_frac": kept / total}

    # --- per patient ------------------------------------------------------
    kp = cells.groupby("patient_id")[col].min().astype(int)
    kept = int((kp * 4).sum())
    opts["per_patient"] = {
        "k": {p: int(v) for p, v in kp.items()},
        "k_min": int(kp.min()), "k_median": float(kp.median()), "k_max": int(kp.max()),
        "kept": kept, "retained_frac": kept / total,
    }

    # --- per patient x task ----------------------------------------------
    kpt = piv["min_pair"].astype(int)
    kept = int((kpt * 2).sum())
    opts["per_patient_task"] = {
        "k": {f"{p}|{t}": int(v) for (p, t), v in kpt.items()},
        "k_min": int(kpt.min()), "k_median": float(kpt.median()), "k_max": int(kpt.max()),
        "kept": kept, "retained_frac": kept / total,
        "by_task": {t: {"k_min": int(kpt.xs(t, level="task_group").min()),
                        "k_median": float(kpt.xs(t, level="task_group").median()),
                        "k_max": int(kpt.xs(t, level="task_group").max())}
                    for t in TASKS},
    }

    opts["_total_windows"] = total
    return opts


def k_sweep(cells: pd.DataFrame, col: str = "n_eval") -> pd.DataFrame:
    """For a GLOBAL or PER-TASK fixed K: how many patients survive at each K.

    A patient survives K only if ALL FOUR of its cells (ON/OFF x tea/toast, or
    the two cells of the task in question) have at least K windows -- dropping
    one condition of a pair would reintroduce exactly the confound being closed.
    """
    rows = []
    piv = cells.pivot_table(index=["patient_id", "task_group"], columns="state",
                            values=col, aggfunc="first")
    piv["min_pair"] = piv[["ON", "OFF"]].min(axis=1)
    per_patient_min = piv.groupby("patient_id")["min_pair"].min()
    n_pat = per_patient_min.size

    for k in range(5, int(cells[col].max()) + 1):
        surv_both = int((per_patient_min >= k).sum())
        row = {"k": k,
               "patients_surviving_both_tasks": surv_both,
               "frac_patients_both_tasks": surv_both / n_pat,
               "windows_kept_both_tasks": surv_both * 4 * k}
        for t in TASKS:
            m = piv.xs(t, level="task_group")["min_pair"]
            s = int((m >= k).sum())
            row[f"patients_surviving_{t}"] = s
            row[f"windows_kept_{t}"] = s * 2 * k
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def describe(series: pd.Series) -> dict:
    q = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(series.size), "mean": float(series.mean()), "sd": float(series.std()),
            "min": float(series.min()), "p05": float(q.loc[0.05]), "p25": float(q.loc[0.25]),
            "median": float(q.loc[0.5]), "p75": float(q.loc[0.75]), "p95": float(q.loc[0.95]),
            "max": float(series.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells()
    pairs_eval = pair_table(cells, "n_eval")
    pairs_all = pair_table(cells, "n_all")

    summary = {
        "source_sidecar": str(SIDECAR.relative_to(_RUN_ROOT)),
        "n_patients": int(cells["patient_id"].nunique()),
        "n_cells": int(len(cells)),
        "n_windows_all_stride5": int(cells["n_all"].sum()),
        "n_windows_eval_nonoverlap": int(cells["n_eval"].sum()),
        "distribution_eval": {
            "overall": describe(cells["n_eval"]),
            "by_task": {t: describe(cells.loc[cells.task_group == t, "n_eval"]) for t in TASKS},
            "by_task_state": {f"{t}|{s}": describe(
                cells.loc[(cells.task_group == t) & (cells.state == s), "n_eval"])
                for t in TASKS for s in ("ON", "OFF")},
        },
        "distribution_all": {
            "overall": describe(cells["n_all"]),
            "by_task": {t: describe(cells.loc[cells.task_group == t, "n_all"]) for t in TASKS},
        },
        "asymmetry_eval": {
            t: {"diff_off_minus_on": describe(pairs_eval.loc[pairs_eval.task_group == t,
                                                             "diff_off_minus_on"]),
                "ratio_off_over_on": describe(pairs_eval.loc[pairs_eval.task_group == t,
                                                             "ratio_off_over_on"]),
                "imbalance_frac": describe(pairs_eval.loc[pairs_eval.task_group == t,
                                                          "imbalance_frac"])}
            for t in TASKS},
        "count_alone_classifier_eval": count_alone_classifier(pairs_eval),
        "matching_options_eval": matching_options(cells, "n_eval"),
        "matching_options_all": matching_options(cells, "n_all"),
    }

    sweep = k_sweep(cells, "n_eval")

    cells.to_csv(out_dir / "window_counts_cells.csv", index=False)
    pairs_eval.to_csv(out_dir / "window_counts_pairs_eval.csv", index=False)
    pairs_all.to_csv(out_dir / "window_counts_pairs_all.csv", index=False)
    sweep.to_csv(out_dir / "window_counts_k_sweep.csv", index=False)
    (out_dir / "window_count_distribution.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    # ---- console read-out ------------------------------------------------
    print(f"{summary['n_patients']} patients x 2 conditions x 2 tasks = {summary['n_cells']} cells")
    print(f"eval (non-overlap) windows: {summary['n_windows_eval_nonoverlap']}   "
          f"train-convention (5 s stride): {summary['n_windows_all_stride5']}\n")

    print("EVAL-CONVENTION WINDOWS PER CELL")
    hdr = f"{'cell':<12}{'n':>4}{'min':>6}{'p25':>7}{'med':>7}{'p75':>7}{'max':>7}{'mean':>8}{'sd':>7}"
    print(hdr)
    for key, d in summary["distribution_eval"]["by_task_state"].items():
        print(f"{key:<12}{d['n']:>4}{d['min']:>6.0f}{d['p25']:>7.1f}{d['median']:>7.1f}"
              f"{d['p75']:>7.1f}{d['max']:>7.0f}{d['mean']:>8.1f}{d['sd']:>7.1f}")
    print()

    print("ON vs OFF ASYMMETRY (eval windows, per patient x task)")
    for t in TASKS:
        c = summary["count_alone_classifier_eval"][t]
        a = summary["asymmetry_eval"][t]
        print(f"  {t}: OFF longer in {c['n_off_longer']}/{c['n_patients']} patients "
              f"(sign p={c['sign_test_p']:.3f}); median |imbalance| "
              f"{a['imbalance_frac']['median']*100:.1f}% of the longer session")
        print(f"       'longer session = OFF' rule bal-acc {c['within_patient_longer_is_off_bal_acc']:.3f} | "
              f"LOO absolute-count threshold bal-acc {c['loo_absolute_threshold_bal_acc']:.3f}")
    print()

    print("FIXED-K OPTIONS (eval windows)")
    mo = summary["matching_options_eval"]
    tot = mo["_total_windows"]
    print(f"  unmatched total: {tot} eval windows")
    g = mo["global"]
    print(f"  global           K={g['k']['all']:<4} kept {g['kept']:>6} "
          f"({g['retained_frac']*100:5.1f}%)  binding cell {g['binding_cell']}")
    pt = mo["per_task"]
    print(f"  per_task         K={pt['k']}  kept {pt['kept']:>6} ({pt['retained_frac']*100:5.1f}%)")
    pp = mo["per_patient"]
    print(f"  per_patient      K in [{pp['k_min']}, {pp['k_max']}] med {pp['k_median']:.0f}   "
          f"kept {pp['kept']:>6} ({pp['retained_frac']*100:5.1f}%)")
    ppt = mo["per_patient_task"]
    print(f"  per_patient_task K in [{ppt['k_min']}, {ppt['k_max']}] med {ppt['k_median']:.0f}   "
          f"kept {ppt['kept']:>6} ({ppt['retained_frac']*100:5.1f}%)")
    for t in TASKS:
        b = ppt["by_task"][t]
        print(f"       {t}: K in [{b['k_min']}, {b['k_max']}] median {b['k_median']:.0f}")
    print()

    print("GLOBAL/PER-TASK K SWEEP — patients keeping ALL FOUR cells at >= K")
    print(f"{'K':>4}{'patients(both tasks)':>22}{'windows':>10}"
          f"{'tea pats':>10}{'toast pats':>12}")
    for k in [10, 15, 20, 25, 26, 30, 35, 40, 45, 50, 60]:
        r = sweep[sweep.k == k]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        print(f"{k:>4}{r['patients_surviving_both_tasks']:>22.0f}"
              f"{r['windows_kept_both_tasks']:>10.0f}"
              f"{r['patients_surviving_tea']:>10.0f}{r['patients_surviving_toast']:>12.0f}")

    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
