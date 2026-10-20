"""PRE-FLIGHT + derivation for the within-patient task-split experiment.

Run this BEFORE ``run_within_patient_tasksplit.py``. It does two things:

  1. Recovers a TASK label for every window of the locked 35-patient cohort by
     re-deriving each session's task-clock -> recording-clock alignment from the
     drive's annotation CSVs, and writes a read-only *sidecar* table
     (``results/within_patient_tasksplit/task_windows.parquet``). The frozen
     window cache, ``manifest.parquet`` and ``config/dataloader.yaml`` are never
     modified — the production cache has ``task_id == "continuous"`` for every
     window, so there is no task label in it to split on.

  2. Applies the CONFOUND PRE-FLIGHT the experiment depends on, fail-loud:
     both task groups (tea, toast) must be recoverable in BOTH the ON and the
     OFF session of a patient, with disjoint spans and enough eval windows.
     If a task type only ever appears in one session type, "task" silently
     re-encodes "session" and the split reintroduces the very temporal /
     duration confound the design is meant to avoid (EXPERIMENTS.md §10c-10d).

Exit codes: 0 = at least one patient is eligible (experiment may proceed);
2 = the confound gate rules out the design cohort-wide (STOP).

Usage:
    python codebrain_finetune/src/derive_task_windows.py
    python codebrain_finetune/src/derive_task_windows.py --drive C:/Data/LivingLab_PD
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent                 # codebrain_finetune/
_PROJECT = _RUN_ROOT.parent                     # Luke_Dissertation/
sys.path.insert(0, str(_RUN_ROOT / "src"))
sys.path.insert(0, str(_PROJECT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from livinglab_codebrain import load_data_config, CohortManifest  # noqa: E402
from codebrain_ft.task_windows import (  # noqa: E402
    load_tasksplit_config, derive_task_windows,
)

log = logging.getLogger("derive_task_windows")

DEFAULT_DRIVE = Path(r"C:\Data\LivingLab_PD")

# Locked-cohort invariants (same live check the LOPO driver makes).
EXPECTED_PATIENTS = 35
EXPECTED_WINDOWS = 37174


def _write_report(path: Path, deriv, ts_cfg, drive_root: Path) -> None:
    groups = ts_cfg.group_names
    g0, g1 = groups
    L = []
    L.append("# Within-patient task-split — PRE-FLIGHT report\n\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} | drive `{drive_root}`\n\n")
    L.append(f"Task groups: **{g0}** = {list(ts_cfg.task_groups[g0])}, "
             f"**{g1}** = {list(ts_cfg.task_groups[g1])}\n\n")

    n_elig = len(deriv.eligible_patients)
    n_tot = len(deriv.cohort_patients)
    L.append(f"## Verdict: **{n_elig} / {n_tot}** cohort patients are eligible\n\n")

    L.append("### The confound question the supervisor asked\n\n")
    L.append("The design is only valid if each task type occurs in BOTH the ON and the OFF\n"
             "session — otherwise task type is confounded with medication state and the split\n"
             "just relabels the temporal/duration confound. Two levels were checked:\n\n")
    L.append("* **Protocol level** (was the task administered at all — the union of fitting and\n"
             "  non-fitting rows in the session's annotation CSV);\n")
    L.append("* **Recoverable level** (does the task actually land inside the recording once the\n"
             "  two clocks are reconciled, and does it contain eval windows).\n\n")

    L.append("### Per-session derivation\n\n")
    L.append(f"| session | shift (s) | tasks fit | {g0} win/eval | {g1} win/eval | status |\n")
    L.append("|---|---|---|---|---|---|\n")
    for s in sorted(deriv.sessions, key=lambda s: (int(s.patient.split("-")[1]), s.condition)):
        # A session with no hard problem can still be missing a task group entirely
        # (the eligibility gate is applied per PATIENT, across both sessions), so say
        # so here rather than printing a bare "OK" next to a 0/0 count.
        empty = [g for g in groups if s.group_eval_window_counts.get(g, 0) == 0]
        if s.problems:
            status = "; ".join(s.problems)
        elif empty:
            status = f"no recoverable {'/'.join(empty)} windows"
        else:
            status = "OK"
        L.append(f"| {s.session} | {s.best_shift_s:.2f} | {len(s.fitting_tasks)}/{s.n_task_rows} | "
                 f"{s.group_window_counts.get(g0, 0)}/{s.group_eval_window_counts.get(g0, 0)} | "
                 f"{s.group_window_counts.get(g1, 0)}/{s.group_eval_window_counts.get(g1, 0)} | "
                 f"{status} |\n")

    L.append("\n### Eligible patients\n\n")
    L.append(", ".join(f"`{p}`" for p in deriv.eligible_patients) or "_none_")
    L.append("\n\n### Ineligible patients (every one listed with its reason — no silent drops)\n\n")
    inelig = [v for v in deriv.verdicts if not v.eligible]
    if not inelig:
        L.append("_none_\n")
    else:
        L.append("| patient | reason(s) |\n|---|---|\n")
        for v in inelig:
            L.append(f"| {v.patient} | " + "<br>".join(v.reasons) + " |\n")

    L.append("\n### Labelled-window totals\n\n")
    sc = deriv.sidecar
    if len(sc):
        L.append("| task group | windows | eval (non-overlap) windows | ON | OFF |\n|---|---|---|---|---|\n")
        for g in groups:
            sub = sc[sc.task_group == g]
            L.append(f"| {g} | {len(sub)} | {int(sub.is_nonoverlap_subset.sum())} | "
                     f"{int((sub.condition == 'A').sum())} | {int((sub.condition == 'C').sum())} |\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(L), encoding="utf-8")
    log.info("Wrote pre-flight report -> %s", path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive task labels + run the task-split pre-flight")
    ap.add_argument("--drive", type=Path, default=DEFAULT_DRIVE,
                    help="read-only LivingLab_PD drive root")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    log_txt = _RUN_ROOT / "logs" / f"tasksplit_preflight_{time.strftime('%Y%m%d')}.log"
    log_txt.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_txt, encoding="utf-8")],
    )

    if not args.drive.is_dir():
        log.error("Drive root not found: %s", args.drive)
        return 1

    data_cfg = load_data_config()
    ts_cfg = load_tasksplit_config()
    manifest = CohortManifest.from_config(data_cfg)

    n_pat, n_win = len(manifest.patients), manifest.n_windows
    log.info("Cohort: %d patients, %d windows, balance=%s", n_pat, n_win, manifest.label_balance())
    if n_pat != EXPECTED_PATIENTS or n_win != EXPECTED_WINDOWS:
        log.error("Cohort invariant mismatch: got (%d, %d), expected (%d, %d). STOP.",
                  n_pat, n_win, EXPECTED_PATIENTS, EXPECTED_WINDOWS)
        return 1

    # Sanity: the production cache genuinely has no task labels (this is WHY the
    # sidecar exists). If a future cache rebuild adds them, use them instead.
    task_ids = set(manifest.frame["task_id"].unique()) if "task_id" in manifest.frame else set()
    log.info("Manifest task_id values: %s", sorted(task_ids))
    if task_ids and task_ids != {"continuous"}:
        log.warning("Manifest carries REAL task_id values %s — the sidecar derivation may be "
                    "redundant; check before trusting it.", sorted(task_ids))

    deriv = derive_task_windows(
        manifest.frame, manifest.patients, args.drive, _PROJECT, ts_cfg,
        sfreq=data_cfg.window.expected_sfreq,
    )

    out_dir = _RUN_ROOT / "results" / ts_cfg.output["results_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = _RUN_ROOT / ts_cfg.output["sidecar"]
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    deriv.sidecar.to_parquet(sidecar_path, index=False)
    log.info("Wrote sidecar (%d labelled windows) -> %s", len(deriv.sidecar), sidecar_path)

    summary = deriv.summary()
    (out_dir / "preflight_summary.json").write_text(json.dumps(
        {**summary,
         "sessions": [
             {"session": s.session, "shift_s": s.best_shift_s,
              "raw_duration_s": s.raw_duration_s, "window_extent_s": s.window_extent_s,
              "fitting_tasks": s.fitting_tasks, "non_fitting_tasks": s.non_fitting_tasks,
              "group_window_counts": s.group_window_counts,
              "group_eval_window_counts": s.group_eval_window_counts,
              "group_spans": {g: [list(x) for x in v] for g, v in s.group_spans.items()},
              "crosscheck": s.crosscheck, "problems": s.problems,
              "csv": s.csv_path, "annotation": s.annotation_path}
             for s in deriv.sessions],
         "verdicts": [{"patient": v.patient, "eligible": v.eligible, "reasons": v.reasons,
                       "per_session": v.per_session} for v in deriv.verdicts],
         }, indent=2, default=str), encoding="utf-8")

    _write_report(_RUN_ROOT / ts_cfg.output["preflight_report"], deriv, ts_cfg, args.drive)

    log.info("=" * 74)
    log.info("PRE-FLIGHT: %d / %d patients eligible", summary["n_eligible_patients"],
             summary["n_cohort_patients"])
    for v in deriv.verdicts:
        if not v.eligible:
            log.info("  INELIGIBLE %-9s | %s", v.patient, " ; ".join(v.reasons))
    log.info("=" * 74)

    if summary["n_eligible_patients"] == 0:
        log.error("STOP: no patient has both task groups recoverable in BOTH sessions. "
                  "Task type is confounded with ON/OFF for this cohort — the within-patient "
                  "task-split design is NOT valid here. Do not proceed.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
