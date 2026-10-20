"""CSV-only recording discovery for the continuous (non-task-locked) cohort run.

ADDITIVE, new module -- does not modify cohort_discovery.py. That module pairs
each recording with BOTH a living-lab EDF and a task-annotation CSV (needed for
Stage 5 clock alignment). This build skips Stage 0 preflight and Stage 5
alignment entirely (continuous windowing has no task segments to align), so the
ONLY input file needed per session is the raw-signal CSV sitting next to the EDF
under `eeg/<livinglab|living-lab-tasks>/` -- never the EDF, never the
`annotation/` task-timestamp CSV.

Layout (same drive as cohort_discovery.py, see that module's docstring):
    <root>/<patient>-<cond>/eeg/<livinglab|living-lab-tasks>/*.csv

The raw-signal CSV is identified by CONTENT, not filename (filenames vary
wildly across sessions -- see cohort_discovery.py's docstring and
RAW_CSV_TRIGGER_AUDIT_REPORT.md): its first line starts with `#` and contains
`Mains_Frequency` (case-insensitive). Files with "accel" or "updrs" in the name
are excluded before sniffing -- "accel" is the separate accelerometer export;
"updrs" is a distinct clinical motor-exam recording that happens to share the
same DSI header format as the living-lab continuous recording (confirmed for
sub-42-c: `sub-42-c_UPDRS_raw.csv` is a different task, not a duplicate/retake
of `sub-42-c_raw.csv` -- see logs/pre_finetuning_audit_20260721.md).

Ambiguity (0 or >1 raw-signal CSV found) is reported, never silently resolved
-- for the pre-vetted cohort allowlist this should not happen; if it does, it
means the pre-vetting was wrong for that session and needs a human look. Two
sessions with genuine same-task ambiguity (not resolvable by a filename-pattern
exclusion) have an explicit, documented manual pick in `MANUAL_FILE_OVERRIDES`
below -- confirmed against the drive 2026-07-22, not guessed:

  * sub-70-c:  two candidates -- `sub-70-c_raw.csv` (117 MB, correctly named
    for this session) vs `sub-70-a-living-lab_raw.csv` (224 MB, misnamed for
    condition A but sitting in the C folder -- almost certainly a stray/
    duplicate file left over from data handling). Picked the correctly-named
    file.
  * sub-121-a: two candidates -- `sub-121-a_raw.csv` (79 MB, 452k lines) vs
    `sub-121-a_0001_raw.csv` (30 MB, 172k lines -- the DSI-Streamer restart-
    suffix pattern). Picked the larger file as the complete recording (the
    `_0001` file is presumed a shorter aborted/restarted attempt).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TASK_SUBFOLDERS = ("livinglab", "living-lab-tasks")

# Explicit, human-confirmed file picks for sessions where content-sniffing
# finds >1 genuinely plausible raw-signal CSV and no filename-pattern
# generalization (like the accel/updrs exclusion) can resolve it safely.
# Keyed by "<patient>-<cond>"; value is the exact filename (not full path) to
# use, expected to live directly under one of TASK_SUBFOLDERS. See the module
# docstring for the evidence behind each pick.
MANUAL_FILE_OVERRIDES: dict[str, str] = {
    "sub-70-c": "sub-70-c_raw.csv",
    "sub-121-a": "sub-121-a_raw.csv",
}


@dataclass(frozen=True)
class CsvRecording:
    """Duck-types discovery.Recording's patient_id/condition/key interface.

    Deliberately has NO edf_path: this cohort is processed from the raw-signal
    CSV alone. Any downstream code path that would need edf_path (Stage 0
    preflight, Stage 5 alignment, provenance.write_provenance) is skipped by
    the CSV-continuous run script -- see run_cohort_continuous_csv.py.
    """
    patient_id: str
    condition: str        # 'A' or 'C', upper-case, matching cfg.labels.condition_map keys
    csv_path: Path         # the raw-signal CSV (this is the ONLY input file needed)

    @property
    def key(self) -> str:
        return f"{self.patient_id}-{self.condition.lower()}"


def _looks_like_raw_signal_csv(path: Path) -> bool:
    """Content-sniff: first line starts with '#' and mentions Mains_Frequency.

    "accel" (accelerometer export) and "updrs" (a distinct clinical motor-exam
    recording, not the living-lab task) are excluded by filename before the
    content-sniff even runs -- both otherwise pass the Mains_Frequency check
    because they share the same DSI export header format.
    """
    name = path.name.lower()
    if "accel" in name or "updrs" in name:
        return False
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as fh:
            first_line = fh.readline()
    except OSError:
        return False
    return "mains_frequency" in first_line.lower()


def _find_csv(root: Path, patient: str, cond: str) -> tuple[Path | None, str]:
    """Locate the raw-signal CSV for one session. Returns (path, note)."""
    eeg = root / f"{patient}-{cond}" / "eeg"
    candidates: list[Path] = []
    for sub in TASK_SUBFOLDERS:
        d = eeg / sub
        if d.is_dir():
            candidates += sorted(d.glob("*.csv"))
    if not candidates:
        return None, "no task subfolder (livinglab/living-lab-tasks) with any .csv found"

    key = f"{patient}-{cond}"
    override_name = MANUAL_FILE_OVERRIDES.get(key)
    if override_name is not None:
        picked = [c for c in candidates if c.name == override_name]
        if len(picked) == 1:
            return picked[0], f"manual override -> {override_name}"
        return None, (f"manual override names {override_name!r} but it was not found "
                      f"(exactly once) among [{', '.join(c.name for c in candidates)}]")

    matches = [c for c in candidates if _looks_like_raw_signal_csv(c)]
    if len(matches) == 0:
        names = ", ".join(c.name for c in candidates) or "none"
        return None, (f"no raw-signal CSV found by content-sniff (first-line "
                      f"'Mains_Frequency') among [{names}]")
    if len(matches) > 1:
        names = ", ".join(c.name for c in matches)
        return None, f"AMBIGUOUS: {len(matches)} raw-signal CSVs found by content-sniff [{names}]"
    return matches[0], "ok"


def discover_csv_cohort(
    root: Path, patient_ids: list[str], conditions: tuple[str, ...] = ("a", "c")
) -> tuple[list[CsvRecording], list[str]]:
    """Find the raw-signal CSV for every patient x condition in an explicit allowlist.

    Returns (recordings, missing): `missing` lists "<patient>-<cond>: <reason>"
    keys for which no unambiguous raw-signal CSV was found -- reported, never
    silently skipped or guessed.
    """
    recordings: list[CsvRecording] = []
    missing: list[str] = []
    for patient in patient_ids:
        for cond in conditions:
            csv_path, note = _find_csv(root, patient, cond)
            key = f"{patient}-{cond}"
            if csv_path is None:
                missing.append(f"{key}: {note}")
            else:
                if note != "ok":
                    print(f"[cohort_discovery_csv] {key}: {note}")
                recordings.append(CsvRecording(patient, cond.upper(), csv_path))
    return recordings, missing
