"""Read-only file-type audit for the LivingLab_PD cohort drive.

Per Dr Haar's clarification (2026-07-13): a session folder holding only a
`.dsi` file is NOT missing data. Both the `.edf` and the task-timestamp `.csv`
are derivable from the `.dsi` via a DSI-Streamer re-export, so a `.dsi`-only
("RAW_ONLY") session is recoverable/usable, not excluded. This script audits
file-type presence per session under that corrected understanding and does
NOT decide cohort membership on its own -- see OUTPUT note at the bottom of
this docstring.

ACTUAL DRIVE LAYOUT (discovered, not assumed)
----------------------------------------------
Each session lives in its OWN top-level folder named `sub-<id>-<cond>`, e.g.
`sub-131-a` and `sub-131-c` are two separate folders -- NOT one `sub-131`
folder containing two session subfolders. This differs from a
"sub-XXX/EEG/living-lab-tasks" mental model; the script discovers folders as
they actually exist and flags (does not silently reshape) anything that
doesn't match the observed `sub-<id>-<cond>` pattern.

Within a session folder, the living-lab files live under (case-insensitive):
    eeg/<livinglab|living-lab-tasks>/   <- .edf, .dsi, and a DECOY .csv
    annotation/                        <- the REAL task-timestamp .csv

KNOWN TRAP (see the drive-layout notes): the `.csv` sitting NEXT TO the
`.edf`/`.dsi` under `eeg/<task-subfolder>/` is the DSI-Streamer RAW SIGNAL
export (per-sample voltages), NOT the task-timestamp file. The genuine
task-timestamp CSV lives in the sibling `annotation/` folder. This script
looks for the task-timestamp CSV in `annotation/` and reports the eeg-folder
CSV separately (`raw_signal_csv_count`) so the two are never conflated -- a
naive "does *.csv exist next to the edf" check would misclassify almost every
RAW_ONLY session as EXPORTED, since the decoy signal CSV sits right there.

CLASSIFICATION
--------------
Per-session status (one of):
    EXPORTED          - .edf AND a task-timestamp .csv (in annotation/), no .dsi
    RAW_ONLY          - .dsi present, no .edf and no task-timestamp .csv
    MIXED             - .dsi present AND (.edf or task-timestamp .csv) present
    MISSING           - no .edf/.csv/.dsi found for this session (folder may
                        still hold unrelated content, e.g. a UPDRS recording)
    EMPTY_FOLDER      - the session folder exists but is completely empty
    INCOMPLETE_EXPORT - .edf present XOR task-timestamp .csv present, no .dsi
                        (NOT one of the original 5 categories -- added
                        because forcing this into EXPORTED or MISSING would be
                        a guess; always logged loudly for manual review)

Subject-level overall status (one of):
    COMPLETE_EXPORTED    - both sessions EXPORTED
    COMPLETE_RECOVERABLE - both sessions usable (EXPORTED/RAW_ONLY/MIXED) and
                           of the SAME usable type
    INCONSISTENT         - both sessions usable but of DIFFERENT usable types
                           (e.g. one EXPORTED, one RAW_ONLY -- the sub-70/
                           sub-71 pattern). Not necessarily wrong; flagged for
                           a second look, per the audit spec.
    PARTIAL              - only one session present/usable, the other
                           MISSING/EMPTY_FOLDER/absent entirely
    EMPTY                - neither session has usable data
    NEEDS_REVIEW         - added safety-net category: either session is
                           INCOMPLETE_EXPORT, or the subject's folder name(s)
                           didn't parse cleanly. Always logged loudly.

OUTPUT
------
Prints a summary table + counts by category, and writes the full per-session
table to a CSV. By design, this file-listing audit is NOT treated
as the authoritative cohort roster -- folder contents are not enrollment;
cross-reference the CSV against the enrollment record separately.

Read-only: only Path.iterdir()/Path.is_file()/Path.is_dir() are used. Nothing
under the scanned root is created, moved, deleted, or modified.

Run:
    python cohort_file_type_audit.py --root "C:\\Data\\LivingLab_PD"
    python cohort_file_type_audit.py --root <path> --out my_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

# Task subfolder was renamed partway through the study; both names are checked
# (case-insensitive). Mirrors src/livinglab_prep/cohort_discovery.py.
TASK_SUBFOLDER_NAMES = ("livinglab", "living-lab-tasks")

# Observed top-level folder convention: sub-<id>-<cond>, cond in {a, c}.
# Anchored at both ends so e.g. "sub-42-a-extra" is flagged, not silently matched.
_SESSION_FOLDER_RE = re.compile(r"^(?P<prefix>sub-[^-]+)-(?P<cond>[ac])$", re.IGNORECASE)

VALID_CONDITIONS = ("a", "c")


# --------------------------------------------------------------------------- data model

@dataclass
class SessionInfo:
    """Everything found (or not found) for one subject-condition folder."""
    patient_id: str
    condition: str                     # 'a' or 'c' (lowercase, canonical)
    folder: Path | None                 # None if the folder was never discovered at all
    edf_files: list[Path] = field(default_factory=list)
    dsi_files: list[Path] = field(default_factory=list)
    task_csv_files: list[Path] = field(default_factory=list)   # from annotation/
    raw_signal_csv_files: list[Path] = field(default_factory=list)  # decoy, from eeg/<task>/
    unexpected_files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    status: str = ""

    @property
    def has_edf(self) -> bool:
        return len(self.edf_files) > 0

    @property
    def has_dsi(self) -> bool:
        return len(self.dsi_files) > 0

    @property
    def has_task_csv(self) -> bool:
        return len(self.task_csv_files) > 0

    @property
    def is_usable(self) -> bool:
        """Usable per Dr Haar's clarification: EXPORTED, RAW_ONLY, or MIXED."""
        return self.status in ("EXPORTED", "RAW_ONLY", "MIXED")


@dataclass
class SubjectSummary:
    patient_id: str
    session_a: SessionInfo
    session_c: SessionInfo
    overall_status: str
    overall_note: str = ""


# --------------------------------------------------------------------------- discovery

def discover_top_level_entries(root: Path) -> list[Path]:
    """List every direct child of `root` (files and directories alike).

    Files at the top level (unexpected) are returned too so the caller can
    log them rather than silently ignore them.
    """
    if not root.is_dir():
        raise SystemExit(f"[cohort_file_type_audit] root is not a directory: {root}")
    return sorted(root.iterdir())


def parse_session_folder_name(name: str) -> tuple[str, str] | None:
    """Return (patient_id, condition) if `name` matches 'sub-<id>-<a|c>', else None.

    Deliberately does NOT normalize anything it can't confidently parse --
    callers must log and separately report a None result, per the "flag,
    don't guess" requirement.
    """
    m = _SESSION_FOLDER_RE.match(name)
    if not m:
        return None
    return m.group("prefix"), m.group("cond").lower()


def _find_child_dir_ci(parent: Path, target_name: str, log: list[str]) -> Path | None:
    """Find a child directory of `parent` matching `target_name` case-insensitively.

    Logs (does not raise) if the match required a case difference, since that
    is a naming inconsistency worth surfacing.
    """
    if not parent.is_dir():
        return None
    exact = parent / target_name
    if exact.is_dir():
        return exact
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target_name.lower():
            log.append(f"case-insensitive match: expected '{target_name}', "
                       f"found '{child.name}' under {parent}")
            return child
    return None


def classify_session(patient_id: str, condition: str, session_dir: Path | None) -> SessionInfo:
    """Inspect one session folder and classify its file-type status.

    `session_dir=None` means the folder was never found at all for this
    subject/condition combination (distinct from an EMPTY_FOLDER, which DOES
    exist but holds nothing).
    """
    info = SessionInfo(patient_id=patient_id, condition=condition, folder=session_dir)

    if session_dir is None:
        info.status = "MISSING"
        info.notes.append("no folder found for this subject/condition combination")
        return info

    all_files_recursive = [p for p in session_dir.rglob("*") if p.is_file()]
    if not all_files_recursive:
        info.status = "EMPTY_FOLDER"
        return info

    # ---- eeg/<livinglab|living-lab-tasks>/ : .edf, .dsi, decoy raw-signal .csv ----
    eeg_dir = _find_child_dir_ci(session_dir, "eeg", info.notes)
    task_dirs: list[Path] = []
    if eeg_dir is not None:
        for candidate_name in TASK_SUBFOLDER_NAMES:
            d = _find_child_dir_ci(eeg_dir, candidate_name, info.notes)
            if d is not None:
                task_dirs.append(d)
    else:
        info.notes.append("no eeg/ subfolder found (checked case-insensitively)")

    for task_dir in task_dirs:
        for f in task_dir.iterdir():
            if not f.is_file():
                continue
            suffix = f.suffix.lower()
            if suffix == ".edf":
                info.edf_files.append(f)
            elif suffix == ".dsi":
                info.dsi_files.append(f)
            elif suffix == ".csv":
                # The eeg-folder CSV is the raw DSI signal export, NOT the
                # task-timestamp file -- see module docstring's KNOWN TRAP.
                info.raw_signal_csv_files.append(f)
            else:
                info.unexpected_files.append(f)

    # ---- annotation/ : the REAL task-timestamp .csv ----
    annotation_dir = _find_child_dir_ci(session_dir, "annotation", info.notes)
    if annotation_dir is not None:
        for f in annotation_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() == ".csv":
                info.task_csv_files.append(f)
            else:
                info.unexpected_files.append(f)
    else:
        info.notes.append("no annotation/ subfolder found (checked case-insensitively)")

    # ---- classify ----
    has_edf, has_dsi, has_csv = info.has_edf, info.has_dsi, info.has_task_csv
    if has_dsi and (has_edf or has_csv):
        info.status = "MIXED"
    elif has_edf and has_csv:
        info.status = "EXPORTED"
    elif has_dsi:
        info.status = "RAW_ONLY"
    elif has_edf != has_csv:  # exactly one of the two, no dsi -- ambiguous
        info.status = "INCOMPLETE_EXPORT"
        info.notes.append(
            f"has_edf={has_edf} has_task_csv={has_csv} has_dsi={has_dsi} "
            "-- doesn't fit a clean category, flagged for manual review")
    else:
        # No edf, no dsi, no task csv -- but SOME file existed somewhere under
        # this session folder (else we'd have hit EMPTY_FOLDER above), e.g. a
        # folder holding only a UPDRS recording or camera footage.
        info.status = "MISSING"
        info.notes.append(
            f"folder is non-empty ({len(all_files_recursive)} files total) but "
            "none are living-lab .edf/.dsi/task-csv -- likely holds only "
            "unrelated task data (UPDRS, cameras, etc.)")

    return info


def classify_subject(patient_id: str, session_a: SessionInfo, session_c: SessionInfo) -> tuple[str, str]:
    """Return (overall_status, note) for one subject from its two sessions."""
    a_usable, c_usable = session_a.is_usable, session_c.is_usable
    a_status, c_status = session_a.status, session_c.status

    if a_status == "INCOMPLETE_EXPORT" or c_status == "INCOMPLETE_EXPORT":
        return "NEEDS_REVIEW", "one or both sessions are INCOMPLETE_EXPORT (see per-session notes)"

    if a_status == "EXPORTED" and c_status == "EXPORTED":
        return "COMPLETE_EXPORTED", ""

    if a_usable and c_usable:
        if a_status == c_status:
            return "COMPLETE_RECOVERABLE", ""
        return "INCONSISTENT", f"session A is {a_status}, session C is {c_status}"

    if a_usable != c_usable:
        missing_side = "C" if a_usable else "A"
        return "PARTIAL", f"only session {'A' if a_usable else 'C'} is usable; session {missing_side} is {c_status if a_usable else a_status}"

    return "EMPTY", "neither session has usable data"


# --------------------------------------------------------------------------- orchestration

def audit_cohort(root: Path) -> tuple[list[SubjectSummary], list[str]]:
    """Walk `root`, classify every subject found, and return
    (subject_summaries, unparsed_folder_names)."""
    entries = discover_top_level_entries(root)

    sessions_by_patient: dict[str, dict[str, Path]] = {}
    unparsed: list[str] = []

    for entry in entries:
        if not entry.is_dir():
            unparsed.append(f"{entry.name} (not a directory)")
            continue
        parsed = parse_session_folder_name(entry.name)
        if parsed is None:
            unparsed.append(entry.name)
            continue
        patient_id, cond = parsed
        sessions_by_patient.setdefault(patient_id, {})[cond] = entry

    summaries: list[SubjectSummary] = []
    for patient_id in sorted(sessions_by_patient, key=lambda s: (len(s), s)):
        folders = sessions_by_patient[patient_id]
        session_a = classify_session(patient_id, "a", folders.get("a"))
        session_c = classify_session(patient_id, "c", folders.get("c"))
        overall, note = classify_subject(patient_id, session_a, session_c)
        summaries.append(SubjectSummary(patient_id, session_a, session_c, overall, note))

    return summaries, unparsed


# --------------------------------------------------------------------------- reporting

def _session_file_summary(info: SessionInfo) -> str:
    parts = []
    if info.edf_files:
        parts.append(f"edf={len(info.edf_files)}")
    if info.task_csv_files:
        parts.append(f"task_csv={len(info.task_csv_files)}")
    if info.dsi_files:
        parts.append(f"dsi={len(info.dsi_files)}")
    if info.raw_signal_csv_files:
        parts.append(f"raw_signal_csv={len(info.raw_signal_csv_files)}")
    return ",".join(parts) if parts else "(none)"


def print_summary(summaries: list[SubjectSummary], unparsed: list[str]) -> None:
    print("=== PER-SUBJECT SUMMARY ===")
    hdr = f"{'patient_id':<14}{'session A':<38}{'session C':<38}{'overall':<20}"
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        a_str = f"{s.session_a.status} [{_session_file_summary(s.session_a)}]"
        c_str = f"{s.session_c.status} [{_session_file_summary(s.session_c)}]"
        print(f"{s.patient_id:<14}{a_str:<38}{c_str:<38}{s.overall_status:<20}")

    print("\n=== LOUD FLAGS (non-standard naming, ambiguous classification) ===")
    if unparsed:
        print(f"  {len(unparsed)} top-level entries did NOT match 'sub-<id>-<a|c>' "
              "and were EXCLUDED from the subject table above (not guessed at):")
        for name in unparsed:
            print(f"    {name}")
    else:
        print("  none -- all top-level entries matched the expected folder pattern")

    for s in summaries:
        for label, sess in (("A", s.session_a), ("C", s.session_c)):
            if sess.notes:
                for n in sess.notes:
                    print(f"  [{s.patient_id} session {label}] {n}")
        if s.overall_note:
            print(f"  [{s.patient_id} OVERALL] {s.overall_note}")

    print("\n=== COUNTS BY OVERALL STATUS ===")
    counts: dict[str, int] = {}
    for s in summaries:
        counts[s.overall_status] = counts.get(s.overall_status, 0) + 1
    for status in ("COMPLETE_EXPORTED", "COMPLETE_RECOVERABLE", "PARTIAL",
                   "INCONSISTENT", "EMPTY", "NEEDS_REVIEW"):
        print(f"  {status:<22} {counts.get(status, 0)}")
    other = set(counts) - {"COMPLETE_EXPORTED", "COMPLETE_RECOVERABLE", "PARTIAL",
                           "INCONSISTENT", "EMPTY", "NEEDS_REVIEW"}
    for status in sorted(other):
        print(f"  {status:<22} {counts[status]}  (UNEXPECTED STATUS -- investigate)")

    usable = counts.get("COMPLETE_EXPORTED", 0) + counts.get("COMPLETE_RECOVERABLE", 0)
    print(f"\n  TOTAL usable (COMPLETE_EXPORTED + COMPLETE_RECOVERABLE) = {usable}")
    print(f"  (INCONSISTENT subjects [{counts.get('INCONSISTENT', 0)}] also have usable "
          "data in both sessions, just of different types -- not included in the total "
          "above per your stated formula; consider whether you want to fold them in.)")
    print(f"\n  Total subjects found: {len(summaries)}")
    print("\n  Reminder: this is a file-listing audit, NOT an enrollment record. "
          "Cross-reference the CSV against your enrollment list before treating any "
          "count here as the final cohort.")


def write_csv(summaries: list[SubjectSummary], out_path: Path) -> None:
    fieldnames = [
        "patient_id", "condition", "status", "edf_count", "task_csv_count",
        "dsi_count", "raw_signal_csv_count", "unexpected_file_count", "notes",
        "overall_status", "overall_note",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            for sess in (s.session_a, s.session_c):
                w.writerow({
                    "patient_id": s.patient_id,
                    "condition": sess.condition.upper(),
                    "status": sess.status,
                    "edf_count": len(sess.edf_files),
                    "task_csv_count": len(sess.task_csv_files),
                    "dsi_count": len(sess.dsi_files),
                    "raw_signal_csv_count": len(sess.raw_signal_csv_files),
                    "unexpected_file_count": len(sess.unexpected_files),
                    "notes": " | ".join(sess.notes),
                    "overall_status": s.overall_status,
                    "overall_note": s.overall_note,
                })
    print(f"\n[cohort_file_type_audit] wrote {out_path}")


# --------------------------------------------------------------------------- entry point

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cohort_file_type_audit")
    parser.add_argument("--root", required=True,
                        help="root of the LivingLab_PD cohort drive (read-only)")
    parser.add_argument("--out", default="outputs/audits/cohort_file_type_audit.csv",
                        help="output CSV path (default: outputs/audits/cohort_file_type_audit.csv)")
    args = parser.parse_args(argv)

    root = Path(args.root)
    summaries, unparsed = audit_cohort(root)
    print_summary(summaries, unparsed)
    write_csv(summaries, Path(args.out))


if __name__ == "__main__":
    main()
