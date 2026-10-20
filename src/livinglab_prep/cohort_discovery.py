"""Cohort discovery for the shared LivingLab_PD drive (post-pilot cohort audit).

Layout differs from the pilot's flat `paths.raw_dir` AND from this module's
first draft. Reconciled against the real drive on 2026-07-06:

  <root>/<patient>-<cond>/
      eeg/<task-subfolder>/ ... _raw.edf   <- living-lab EEG recording
      annotation/ ... .csv                 <- task-timestamp file (Tasks,Start,End)

Two hard-won facts about the real drive:

  1. The `_raw.csv` sitting NEXT TO the EDF is the DSI-Streamer raw SIGNAL export
     (per-sample voltages, `# Mains_Frequency` header), NOT a task file. The task
     timestamps live under a sibling `annotation/` folder. We pair the EDF with an
     annotation CSV whose header actually carries {Tasks, Start, End}, verified by
     reading the header (filenames vary: `*_annot.csv`, `*_<date>.csv`, and some
     patients also have a 2-column `time offset` file that must be rejected).

  2. Filenames are wildly inconsistent across patients: stripped condition
     (`sub-34_living-lab_raw.edf`), spaces (`sub-42-a-living lab_raw.edf`),
     underscore ids (`sub_51a_raw.edf`), glued condition (`sub-48a_..._raw.edf`),
     `_0001` restart fragments (`sub-134-c_0001_raw.edf`), and folders holding a
     second non-living-lab EDF (sub-42-c has a UPDRS recording alongside). So we
     glob `*_raw.edf` in the task subfolder(s), drop any whose name matches another
     task's keyword, and when >1 remain pick the largest (fullest session) with a
     loud warning.

The task subfolder was renamed partway through the study (`livinglab` ->
`living-lab-tasks`); both are checked. `eeg/` also holds unrelated task subfolders
(Blocks, catsndogs, Motor, ...) which are never matched here.

Restricted to an explicit patient allowlist (not a blind rglob) because the shared
drive also hosts a different sub-study on DSI-only patients that must never be
swept in by accident -- see the known data issues and the 2026-07-06
cohort audit.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .discovery import Recording

TASK_SUBFOLDERS = ("livinglab", "living-lab-tasks")

# Keywords that identify a NON-living-lab EDF that happens to sit in the same
# task subfolder (e.g. a stray UPDRS recording). Matched as lowercased substrings
# of the EDF filename; a normal living-lab name contains none of these.
_OTHER_TASK_KW = (
    "updrs", "block", "motor", "manipulation", "rotation", "rearrange",
    "memory", "delayed", "immediate", "pal", "paired", "associate",
    "catsndog", "cats-and-dog", "catsanddog", "resting",
    "-ec-", "-eo-", "_ec_", "_eo_",
)

# The task-annotation CSV must carry exactly these columns in its header.
_TASK_COLUMNS = {"Tasks", "Start", "End"}


def _find_edf(root: Path, patient: str, cond: str) -> tuple[Path | None, str]:
    """Locate the living-lab EDF for one recording. Returns (path, note)."""
    eeg = root / f"{patient}-{cond}" / "eeg"
    candidates: list[Path] = []
    for sub in TASK_SUBFOLDERS:
        d = eeg / sub
        if d.is_dir():
            candidates += sorted(d.glob("*_raw.edf"))
    if not candidates:
        return None, "no living-lab EDF under eeg/{livinglab,living-lab-tasks}"

    # Drop EDFs whose name belongs to another task recorded in the same folder.
    filtered = [c for c in candidates
                if not any(kw in c.name.lower() for kw in _OTHER_TASK_KW)]
    pool = filtered or candidates
    if len(pool) == 1:
        return pool[0], "ok"
    # >1 living-lab EDF (typically a `_0001` restart fragment): take the largest.
    pool = sorted(pool, key=lambda p: p.stat().st_size, reverse=True)
    names = ", ".join(p.name for p in pool)
    return pool[0], f"AMBIGUOUS: {len(pool)} living-lab EDFs [{names}] -> picked largest '{pool[0].name}'"


def _has_task_header(csv_path: Path) -> bool:
    """True iff the CSV's header row carries {Tasks, Start, End}."""
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh), [])
    except OSError:
        return False
    return _TASK_COLUMNS.issubset({h.strip() for h in header})


def _find_annotation(root: Path, patient: str, cond: str) -> tuple[Path | None, str]:
    """Locate the task-annotation CSV for one recording. Returns (path, note)."""
    d = root / f"{patient}-{cond}" / "annotation"
    if not d.is_dir():
        return None, "no annotation/ folder"
    all_csv = sorted(d.glob("*.csv"))
    valid = [c for c in all_csv if _has_task_header(c)]
    if not valid:
        had = ", ".join(c.name for c in all_csv) or "none"
        return None, f"no annotation CSV with a {sorted(_TASK_COLUMNS)} header (found: {had})"
    if len(valid) == 1:
        return valid[0], "ok"
    names = ", ".join(p.name for p in valid)
    return valid[0], f"AMBIGUOUS: {len(valid)} task CSVs [{names}] -> picked first '{valid[0].name}'"


def _find_recording(root: Path, patient: str, cond: str) -> tuple[Recording | None, str]:
    edf, edf_note = _find_edf(root, patient, cond)
    ann, ann_note = _find_annotation(root, patient, cond)
    if edf is None or ann is None:
        return None, f"edf: {edf_note} | annotation: {ann_note}"
    note = "; ".join(n for n in (edf_note, ann_note) if n != "ok") or "ok"
    return Recording(patient, cond.upper(), edf, ann), note


def discover_cohort_recordings(
    root: Path, patient_ids: list[str], conditions: tuple[str, ...] = ("a", "c")
) -> tuple[list[Recording], list[str]]:
    """Find (living-lab EDF, task-annotation CSV) pairs for an explicit allowlist.

    Returns (recordings, missing): `missing` lists "<patient>-<cond>: <reason>"
    keys for which no complete pair was found -- reported, never silently
    skipped, since absence can mean DSI-only (different sub-study) or a genuine
    gap (see sub-70/71/130 in the cohort audit). Ambiguous but resolvable cases
    (multiple EDFs / task CSVs) are kept and their resolution is printed.
    """
    recordings: list[Recording] = []
    missing: list[str] = []
    for patient in patient_ids:
        for cond in conditions:
            rec, note = _find_recording(root, patient, cond)
            key = f"{patient}-{cond}"
            if rec is None:
                missing.append(f"{key}: {note}")
            else:
                if note != "ok":
                    print(f"[cohort_discovery] {key}: {note}")
                recordings.append(rec)
    return recordings, missing
