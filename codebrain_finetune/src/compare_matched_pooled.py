"""Matched vs unmatched pooled task-split — the side-by-side table.

Pairs every (direction, arm) cell of ``pooled_tasksplit_chanz`` (unmatched) with
``pooled_tasksplit_chanz_matched`` (duration-matched, level=patient_task,
where=uniform) and reports the change in patient-mean balanced accuracy with its
bootstrap CI.

Reading the delta honestly: matching removes ~10% of the windows as well as the
count asymmetry, so a small drop is expected from reduced training data alone.
The CIs are patient-level bootstraps on the SAME 27 patients, so they are paired
in the design but the intervals printed here are marginal — treat an overlap as
"no detectable change", not as a formal paired test.

Usage:
    python codebrain_finetune/src/compare_matched_pooled.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_RUN_ROOT = Path(__file__).resolve().parent.parent
UNMATCHED = _RUN_ROOT / "results" / "pooled_tasksplit_chanz"
MATCHED = _RUN_ROOT / "results" / "pooled_tasksplit_chanz_matched"
ARMS = ["frozen_probe", "lora", "full_ft", "chanamp"]
DIRECTIONS = ["toast_to_tea", "tea_to_toast"]


def _load(d: Path, direction: str, arm: str) -> dict | None:
    p = d / f"{direction}_{arm}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for direction in DIRECTIONS:
        for arm in ARMS:
            u, m = _load(UNMATCHED, direction, arm), _load(MATCHED, direction, arm)
            if u is None or m is None:
                continue
            up, mp = u["pooled_patient_level"], m["pooled_patient_level"]
            rows.append({
                "direction": direction, "arm": arm,
                "unmatched_BA": up["mean"], "u_lo": up["ci_low"], "u_hi": up["ci_high"],
                "matched_BA": mp["mean"], "m_lo": mp["ci_low"], "m_hi": mp["ci_high"],
                "delta": mp["mean"] - up["mean"],
                "u_train": u["n_train_windows"], "m_train": m["n_train_windows"],
                "u_test": u["n_test_windows"], "m_test": m["n_test_windows"],
                "u_window_BA": u["window"]["balanced_accuracy"],
                "m_window_BA": m["window"]["balanced_accuracy"],
                "u_session_BA": u["session"]["balanced_accuracy"],
                "m_session_BA": m["session"]["balanced_accuracy"],
                "u_ordering": u["ordering"].get("frac_correct"),
                "m_ordering": m["ordering"].get("frac_correct"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No paired results found — has the matched run finished?")
        return 1

    out = _RUN_ROOT / "results" / "pooled_matched_vs_unmatched.csv"
    df.to_csv(out, index=False)

    print("POOLED TASK-SPLIT (chan_z): duration-matched vs unmatched")
    print("patient-mean balanced accuracy, 27 patients, 95% cluster-bootstrap CI\n")
    print(f"{'direction':<14}{'arm':<14}{'unmatched':>22}{'matched':>22}{'delta':>9}"
          f"{'train win':>20}")
    for _, r in df.iterrows():
        print(f"{r['direction']:<14}{r['arm']:<14}"
              f"{r['unmatched_BA']:>8.3f} [{r['u_lo']:.3f},{r['u_hi']:.3f}]"
              f"{r['matched_BA']:>8.3f} [{r['m_lo']:.3f},{r['m_hi']:.3f}]"
              f"{r['delta']:>+9.3f}"
              f"{int(r['u_train']):>10,}->{int(r['m_train']):>8,}")

    print("\nSession-level and ordering:")
    print(f"{'direction':<14}{'arm':<14}{'session BA':>22}{'ordering frac':>26}")
    for _, r in df.iterrows():
        uo = f"{r['u_ordering']:.3f}" if r["u_ordering"] is not None else "  n/a"
        mo = f"{r['m_ordering']:.3f}" if r["m_ordering"] is not None else "  n/a"
        print(f"{r['direction']:<14}{r['arm']:<14}"
              f"{r['u_session_BA']:>10.3f} ->{r['m_session_BA']:>9.3f}"
              f"{uo:>16} ->{mo:>9}")

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
