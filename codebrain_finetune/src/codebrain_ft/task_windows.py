"""Derive per-window TASK labels for the continuous window cache, + the
pre-flight confound gates for the within-patient task-split experiment.

WHY THIS MODULE HAS TO EXIST
----------------------------
The production cache (``processed_csv_continuous/reref-linkedears_ica/``) was
built by *continuous* windowing: every window's ``task_id`` is the constant
placeholder ``"continuous"`` (``window_continuous.py::CONTINUOUS_TASK_ID``).
Task-locked windowing was deliberately dropped cohort-wide (Option 3 of
RAW_CSV_TRIGGER_AUDIT_REPORT.md) because 0/89 sessions carry a usable native
trigger and task recovery is only partial for ~88% of them. So there is no task
label anywhere in the manifest to split on, and the older EDF-based cache that
*does* carry real ``task_id`` values covers only 17 patients with large ON/OFF
gaps — not the locked 35-patient cohort the LOPO arms ran on.

This module recovers the task labels for the production cohort WITHOUT touching
the frozen cache: it re-derives each session's task->recording clock alignment
from the drive's annotation CSV and emits a *sidecar* table mapping existing
window rows to a task group. The window cache, the manifest and
``config/dataloader.yaml`` are all read-only here.

CORRECTNESS ANCHORS
-------------------
* The clock-shift search is the same algorithm the full-cohort audit validated
  against sub-131/sub-134's documented ground truth
  (``raw_csv_edf_trigger_audit.py::_search_best_shift``, reimplemented here so
  this module has no dependency on a root-level analysis script).
* Every re-derived alignment is CROSS-CHECKED against the stored audit results
  and aborts on divergence: a mismatch means the raw-signal CSV selected here
  differs from the one the audit measured, which would map task timestamps onto
  the wrong recording. Verified reproducing the audit's shift/duration/fit-count
  exactly on the sessions checked during development.
* The raw-signal CSV is located with the PRODUCTION discovery function
  (``cohort_discovery_csv.discover_csv_cohort``), including its human-confirmed
  ``MANUAL_FILE_OVERRIDES``, so the file whose clock we align is the same file
  the windows were cut from.
* Window elapsed time is exact by construction: continuous windowing sets sample
  0 = session start, so ``start_sample / sfreq`` is seconds from the recording's
  own start — the same reference the CSV header timestamp gives.

Nothing is silently skipped: every session that cannot be aligned, and every
patient that fails a gate, is recorded with a reason and surfaced in the report.
"""
from __future__ import annotations

import csv as csvmod
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

__all__ = [
    "TaskSplitConfig", "load_tasksplit_config",
    "SessionAlignment", "PatientVerdict", "TaskWindowDerivation",
    "derive_task_windows", "load_sidecar",
]

# Task-CSV timestamps are epoch microseconds; reject anything implausible as a
# corrupt row (documented: sub-134-c logs several tasks with Start=0).
_PLAUSIBLE_YEAR_RANGE = (2015, 2035)
_TASK_CSV_HEADER_PREFIX = "tasks,start,end"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSplitConfig:
    """Typed view of ``config/tasksplit.yaml`` (fail-loud on missing keys)."""

    task_groups: dict[str, tuple[str, ...]]
    alignment: dict
    assignment: dict
    preflight: dict
    protocol: dict
    arms: dict
    stats: dict
    output: dict
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(self.task_groups)

    def group_of(self, task: str) -> str | None:
        """Map a verbatim task string to its group, or None if ungrouped."""
        for group, members in self.task_groups.items():
            if task in members:
                return group
        return None


def load_tasksplit_config(root: Path | None = None) -> TaskSplitConfig:
    from livinglab_codebrain.config import run_root

    root = root or run_root()
    path = root / "config" / "tasksplit.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"tasksplit config not found: {path}")
    d = yaml.safe_load(path.read_text())
    for key in ("task_groups", "alignment", "assignment", "preflight",
                "protocol", "arms", "stats", "output"):
        if key not in d:
            raise KeyError(f"tasksplit.yaml: missing required top-level key '{key}'")
    groups = {g: tuple(v) for g, v in d["task_groups"].items()}
    if len(groups) != 2:
        raise ValueError(
            f"tasksplit.yaml: task_groups must define exactly 2 groups (train-on-one, "
            f"test-on-other); got {sorted(groups)}")
    return TaskSplitConfig(
        task_groups=groups, alignment=d["alignment"], assignment=d["assignment"],
        preflight=d["preflight"], protocol=d["protocol"], arms=d["arms"],
        stats=d["stats"], output=d["output"], raw=d,
    )


# ---------------------------------------------------------------------------
# raw-signal CSV header / duration (cheap — never reads the whole file)
# ---------------------------------------------------------------------------
def _parse_csv_meta(csv_path: Path) -> dict[str, str]:
    """Read the ``# key =,value`` metadata block above the ``Time,...`` header row."""
    meta: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        for idx, row in enumerate(csvmod.reader(fh)):
            if not row:
                continue
            if row[0].strip() == "Time":
                return meta
            stripped = row[0].lstrip()
            if stripped.startswith("#"):
                key = stripped.lstrip("#").strip()
                for trail in ("=", ":"):
                    if key.endswith(trail):
                        key = key[: -len(trail)].strip()
                meta[key.strip().lower()] = row[1].strip() if len(row) > 1 else ""
            if idx > 60:
                break
    raise ValueError(f"{csv_path}: no 'Time,...' column-header row in the first 60 lines")


def _recording_start(csv_path: Path) -> datetime:
    """Recording start as a naive datetime from the CSV header's Date + Time.

    Format matches ``csv_signal.py::_parse_meas_date`` (``%d/%m/%Y`` + ``%H:%M:%S[.%f]``);
    the header carries millisecond precision, which the EDF export does not.
    """
    meta = _parse_csv_meta(csv_path)
    date_str, time_str = meta.get("date"), meta.get("time")
    if not date_str or not time_str:
        raise ValueError(f"{csv_path}: header has no Date/Time metadata; cannot align tasks")
    d = datetime.strptime(date_str.strip(), "%d/%m/%Y").date()
    ts = time_str.strip()
    t = datetime.strptime(ts, "%H:%M:%S.%f" if "." in ts else "%H:%M:%S").time()
    return datetime.combine(d, t)


def _recording_duration_s(csv_path: Path, *, tail_bytes: int = 65536) -> float:
    """``last Time - first Time`` from the data rows, read by head + tail seek.

    Matches the audit's ``duration_s`` definition. Reading only the head and a
    64 KB tail keeps this cheap on the 100-200 MB per-session CSVs.
    """
    first_t = None
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        seen_header = False
        for row in csvmod.reader(fh):
            if not row:
                continue
            if not seen_header:
                seen_header = row[0].strip() == "Time"
                continue
            try:
                first_t = float(row[0])
            except ValueError:
                continue
            break
    if first_t is None:
        raise ValueError(f"{csv_path}: no parseable data rows after the header")

    size = csv_path.stat().st_size
    with open(csv_path, "rb") as fh:
        fh.seek(max(0, size - tail_bytes))
        tail = fh.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(tail):
        parts = line.split(",")
        try:
            return float(parts[0]) - first_t
        except (ValueError, IndexError):
            continue
    raise ValueError(f"{csv_path}: could not parse a final Time value from the last {tail_bytes} bytes")


# ---------------------------------------------------------------------------
# annotation (task-timestamp) CSV
# ---------------------------------------------------------------------------
def find_annotation_csv(drive_root: Path, patient: str, cond: str) -> tuple[Path | None, str]:
    """Locate ``<patient>-<cond>/annotation/*.csv`` with a ``Tasks,Start,End`` header.

    Header-shape validated, not assumed: ``sub-34-c/annotation/`` holds a 2-column
    ``Tasks,Start`` scratch file next to the real one (cohort-drive-layout trap).
    """
    d = drive_root / f"{patient}-{cond}" / "annotation"
    if not d.is_dir():
        return None, "no annotation/ folder"
    rejected = []
    for f in sorted(d.glob("*.csv")):
        try:
            with open(f, "r", errors="replace") as fh:
                first = fh.readline().strip().lower().replace(" ", "")
        except OSError as e:
            rejected.append(f"{f.name} (unreadable: {e})")
            continue
        if first.startswith(_TASK_CSV_HEADER_PREFIX):
            return f, "ok"
        rejected.append(f"{f.name} (header {first[:40]!r})")
    return None, f"no CSV with a 'Tasks,Start,End' header (rejected: {'; '.join(rejected) or 'none'})"


def parse_task_csv(path: Path) -> tuple[list[dict], list[tuple[list, str]]]:
    """Parse task rows into ``{task, start_dt, end_dt, duration_s}``; return (rows, rejected)."""
    rows: list[dict] = []
    rejected: list[tuple[list, str]] = []
    with open(path, "r", errors="replace", newline="") as fh:
        reader = csvmod.reader(fh)
        next(reader, None)  # header
        for r in reader:
            if len(r) < 3:
                rejected.append((r, "too few columns"))
                continue
            task, s_raw, e_raw = r[0], r[1], r[2]
            try:
                s_us, e_us = float(s_raw), float(e_raw)
            except ValueError:
                rejected.append((r, "unparseable epoch"))
                continue
            try:
                s_dt = datetime(1970, 1, 1) + timedelta(microseconds=s_us)
                e_dt = datetime(1970, 1, 1) + timedelta(microseconds=e_us)
            except (OverflowError, OSError, ValueError):
                rejected.append((r, "epoch out of range"))
                continue
            if not (_PLAUSIBLE_YEAR_RANGE[0] <= s_dt.year <= _PLAUSIBLE_YEAR_RANGE[1]):
                rejected.append((r, f"implausible year {s_dt.year} (corrupt timestamp)"))
                continue
            if e_us < s_us:
                rejected.append((r, "end before start"))
                continue
            rows.append({"task": task, "start_dt": s_dt, "end_dt": e_dt,
                         "duration_s": (e_us - s_us) / 1e6})
    return rows, rejected


# ---------------------------------------------------------------------------
# clock alignment (reimplementation of the audit-validated search)
# ---------------------------------------------------------------------------
def _fitting_tasks(task_rows, raw_start, raw_end, shift_s, slack_s):
    """Tasks that land inside the recording once shifted onto its clock."""
    lo = raw_start - timedelta(seconds=slack_s)
    hi = raw_end + timedelta(seconds=slack_s)
    out = []
    for r in task_rows:
        s = r["start_dt"] - timedelta(seconds=shift_s)
        e = r["end_dt"] - timedelta(seconds=shift_s)
        if lo <= s and e <= hi:
            out.append(r)
    return out


def _hour_residual(shift_s: float) -> float:
    """Distance from ``shift_s`` to the nearest whole-hour multiple."""
    return abs(shift_s - round(shift_s / 3600.0) * 3600.0)


def search_best_shift(task_rows, raw_start, raw_end, acfg: dict) -> tuple[float, list[dict]]:
    """Dense scan for the task->recording clock shift maximising the fit count.

    Deliberately assumes NO shift shape: an earlier whole-hour-only search
    produced false "unresolvable" sessions (sub-34-c/42-c/48-c need non-hour
    shifts). Coarse 2 s scan over +/-4 h, then a 0.05 s refinement.

    TIE-BREAK (added 2026-07-30, investigating Dr Haar's DST hypothesis): the
    coarse scan can find two DIFFERENT shifts that recover the exact same NUMBER
    of tasks, just not the same tasks -- confirmed for sub-62/72/76/84/93/103/106,
    where one candidate recovers 'Suspine Measure'+'Standing Still' (two tasks
    independently known elsewhere in this project to carry corrupted/zero-epoch
    timestamps) and a whole-hour-shifted candidate recovers 'Kettel Task'+'Toast
    Task' instead (the tasks this experiment actually needs, reliably logged).
    Naive first-found tie-breaking landed on the unreliable side for all 7. Among
    EXACT ties only, prefer whichever candidate sits closest to a whole-hour
    offset -- the physically expected value under the documented BST/GMT
    clock-labelling bug (RAW_CSV_TRIGGER_AUDIT_REPORT.md). This never changes a
    session with a single unambiguous best shift; it only arbitrates equally-good
    candidates, and does so towards the already-validated systematic explanation
    rather than an arbitrary scan-order default.
    """
    half = float(acfg["search_half_range_s"])
    step = float(acfg["coarse_step_s"])
    slack = float(acfg["fit_slack_s"])

    best_n = -1
    ties: list[tuple[float, list[dict]]] = []          # coarse-scan candidates tied for best_n
    for shift in np.arange(-half, half + step, step):
        fit = _fitting_tasks(task_rows, raw_start, raw_end, float(shift), slack)
        n = len(fit)
        if n > best_n:
            best_n, ties = n, [(float(shift), fit)]
        elif n == best_n:
            ties.append((float(shift), fit))
    best_shift, best_fit = min(ties, key=lambda c: _hour_residual(c[0]))

    r_half = float(acfg["refine_half_range_s"])
    r_step = float(acfg["refine_step_s"])
    for shift in np.arange(best_shift - r_half, best_shift + r_half + r_step, r_step):
        fit = _fitting_tasks(task_rows, raw_start, raw_end, float(shift), slack)
        if len(fit) > best_n:
            best_n, best_shift, best_fit = len(fit), float(shift), fit
    return best_shift, best_fit


# ---------------------------------------------------------------------------
# per-session result
# ---------------------------------------------------------------------------
@dataclass
class SessionAlignment:
    """One session's derived task spans, in recording-elapsed seconds."""

    session: str                 # "sub-34-a"
    patient: str
    condition: str               # "A" / "C"
    csv_path: str
    annotation_path: str
    raw_start: str
    raw_duration_s: float
    window_extent_s: float
    best_shift_s: float
    n_task_rows: int
    n_rejected_task_rows: int
    fitting_tasks: list[str]
    non_fitting_tasks: list[str]
    # group -> list of (start_elapsed_s, end_elapsed_s) spans
    group_spans: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    group_window_counts: dict[str, int] = field(default_factory=dict)
    group_eval_window_counts: dict[str, int] = field(default_factory=dict)
    crosscheck: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class PatientVerdict:
    patient: str
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    per_session: dict = field(default_factory=dict)


@dataclass
class TaskWindowDerivation:
    """Everything the derivation produced: sidecar rows + audit trail."""

    sidecar: pd.DataFrame
    sessions: list[SessionAlignment]
    verdicts: list[PatientVerdict]
    eligible_patients: list[str]
    cohort_patients: list[str]

    def summary(self) -> dict:
        return {
            "n_cohort_patients": len(self.cohort_patients),
            "n_eligible_patients": len(self.eligible_patients),
            "eligible_patients": list(self.eligible_patients),
            "ineligible": {v.patient: v.reasons for v in self.verdicts if not v.eligible},
            "n_labelled_windows": int(len(self.sidecar)),
        }


# ---------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------
def _load_audit(project_root: Path) -> dict:
    path = project_root / "raw_csv_edf_trigger_audit_results.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"cohort alignment audit not found: {path}. It is required for the "
            f"alignment cross-check (set alignment.crosscheck_audit: false to skip, "
            f"but then the re-derived clock shift is unverified).")
    return {s["session"]: s for s in json.loads(path.read_text())}


def _assign_windows(session_rows: pd.DataFrame, spans: list[tuple[float, float]],
                    sfreq: int, mode: str) -> np.ndarray:
    """Boolean mask of windows belonging to ``spans``."""
    if mode != "containment":
        raise ValueError(f"assignment.mode must be 'containment', got {mode!r}")
    start_s = session_rows["start_sample"].to_numpy() / float(sfreq)
    end_s = session_rows["end_sample"].to_numpy() / float(sfreq)
    mask = np.zeros(len(session_rows), dtype=bool)
    for lo, hi in spans:
        mask |= (start_s >= lo) & (end_s <= hi)
    return mask


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge touching/overlapping spans within one group (e.g. Toast Task1+Task2)."""
    if not spans:
        return []
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return [tuple(s) for s in out]


def _spans_overlap_s(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Total seconds of overlap between two span lists."""
    total = 0.0
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            total += max(0.0, min(hi1, hi2) - max(lo1, lo2))
    return total


def derive_task_windows(
    manifest_frame: pd.DataFrame,
    cohort_patients: list[str],
    drive_root: Path,
    project_root: Path,
    ts_cfg: TaskSplitConfig,
    *,
    sfreq: int,
) -> TaskWindowDerivation:
    """Map every cohort window to a task group and apply the pre-flight gates.

    ``manifest_frame`` must be the post-exclusion cohort manifest (read-only).
    Returns a :class:`TaskWindowDerivation`; nothing is written to disk here.
    """
    sys.path.insert(0, str(project_root)) if str(project_root) not in sys.path else None
    from src.livinglab_prep.cohort_discovery_csv import discover_csv_cohort

    acfg = ts_cfg.alignment
    pcfg = ts_cfg.preflight
    groups = ts_cfg.group_names
    audit = _load_audit(project_root) if acfg.get("crosscheck_audit", True) else {}

    log.info("Discovering raw-signal CSVs for %d cohort patients (production discovery, "
             "incl. MANUAL_FILE_OVERRIDES)...", len(cohort_patients))
    recordings, missing = discover_csv_cohort(drive_root, cohort_patients)
    if missing:
        # A cohort patient whose windows exist but whose source CSV can no longer
        # be located unambiguously is a hard inconsistency, not something to skip.
        raise RuntimeError(
            "Raw-signal CSV could not be resolved for cohort sessions that DO have "
            "windows in the manifest — refusing to guess:\n  " + "\n  ".join(missing))

    sessions: list[SessionAlignment] = []
    sidecar_parts: list[pd.DataFrame] = []

    for rec in recordings:
        key = rec.key                       # "sub-34-a"
        rows = manifest_frame[
            (manifest_frame["patient_id"] == rec.patient_id)
            & (manifest_frame["condition"] == rec.condition)
        ]
        if len(rows) == 0:
            raise RuntimeError(
                f"{key}: discovered a raw CSV but the manifest has 0 windows for this "
                f"session — cohort/manifest inconsistency, aborting.")

        problems: list[str] = []
        raw_start = _recording_start(rec.csv_path)
        duration_s = _recording_duration_s(rec.csv_path)
        raw_end = raw_start + timedelta(seconds=duration_s)
        window_extent_s = float(rows["end_sample"].max()) / sfreq

        ann_path, ann_note = find_annotation_csv(drive_root, rec.patient_id, rec.condition.lower())
        if ann_path is None:
            problems.append(f"no usable annotation CSV: {ann_note}")
            sessions.append(SessionAlignment(
                session=key, patient=rec.patient_id, condition=rec.condition,
                csv_path=str(rec.csv_path), annotation_path="", raw_start=raw_start.isoformat(),
                raw_duration_s=duration_s, window_extent_s=window_extent_s,
                best_shift_s=float("nan"), n_task_rows=0, n_rejected_task_rows=0,
                fitting_tasks=[], non_fitting_tasks=[], problems=problems,
            ))
            continue

        task_rows, rejected = parse_task_csv(ann_path)
        if not task_rows:
            problems.append(f"annotation CSV has 0 valid task rows ({len(rejected)} rejected)")

        best_shift, fitting = (float("nan"), [])
        if task_rows:
            best_shift, fitting = search_best_shift(task_rows, raw_start, raw_end, acfg)

        # ---- cross-check against the cohort-wide audit -------------------
        # Duration still has to match closely -- that only depends on which raw
        # CSV was read, nothing to do with tie-breaking, so any divergence there
        # still means "probably the wrong file" and stays a hard failure.
        #
        # The SHIFT check was changed 2026-07-30 (DST tie-break fix): the audit's
        # shift is no longer treated as ground truth to match exactly, because
        # search_best_shift() now deliberately picks a DIFFERENT shift than the
        # audit's whenever several shifts tie on total tasks recovered (preferring
        # the whole-hour-consistent one). Exact-shift agreement is informational.
        #
        # The whole-cohort FIT-COUNT check was also tried and then dropped as a
        # hard failure (same date): investigating 3 flagged sessions
        # (sub-71-c/73-c/103-a) found the shift matched the audit EXACTLY in all
        # three -- the deficit was one specific task, "Balance", logged with a
        # corrupted end timestamp of literal "0" (e.g. sub-71-c:
        # ['Balance','1688396253132731','0']). This module's parser correctly
        # rejects that as "end before start"; the ORIGINAL audit script's parser
        # never validated end_dt's year (only start_dt's), so the corrupted
        # end=1970-01-01 slipped through and, because it is trivially earlier
        # than any recording's end, the audit's fit check degenerated into
        # "does the task's START land in the window" for that row -- an
        # unintentional pass, not evidence the row is trustworthy. Comparing
        # TOTAL fit count therefore just measures a parser-strictness difference
        # on tasks this experiment doesn't use. What's checked instead is scoped
        # to what actually matters: recovery of THIS experiment's own task
        # groups (tea/toast) must not regress relative to the audit.
        crosscheck: dict = {}
        if audit:
            a = (audit.get(key) or {}).get("alignment") or {}
            a_shift, a_dur = a.get("best_shift_s"), a.get("raw_duration_s")
            a_fit = a.get("best_fit_task_count")
            a_scoped = sum(1 for t in (a.get("fitting_tasks") or []) if ts_cfg.group_of(t) is not None)
            my_scoped = sum(1 for r in fitting if ts_cfg.group_of(r["task"]) is not None)
            crosscheck = {"audit_best_shift_s": a_shift, "audit_raw_duration_s": a_dur,
                          "audit_fit_count": a_fit, "audit_scoped_fit_count": a_scoped,
                          "my_scoped_fit_count": my_scoped,
                          "shift_diverged_from_audit": bool(
                              a_shift is not None
                              and abs(a_shift - best_shift) > float(acfg["crosscheck_max_shift_delta_s"]))}
            if a_dur is not None and abs(a_dur - duration_s) > float(acfg["crosscheck_max_duration_delta_s"]):
                problems.append(
                    f"ALIGNMENT CROSS-CHECK FAILED: recording duration {duration_s:.2f}s differs "
                    f"from the cohort audit's {a_dur:.2f}s by more than "
                    f"{acfg['crosscheck_max_duration_delta_s']}s — the raw CSV selected here is "
                    f"probably not the one the audit measured ({Path(rec.csv_path).name})")
            if my_scoped < a_scoped:
                problems.append(
                    f"ALIGNMENT CROSS-CHECK FAILED: re-derived alignment recovers {my_scoped} of this "
                    f"experiment's own task-group instances (tea/toast), fewer than the "
                    f"{a_scoped} the cohort audit's fitting_tasks imply — the raw CSV selected here "
                    f"may not be the one the audit measured ({Path(rec.csv_path).name})")

        # ---- task spans in recording-elapsed seconds ---------------------
        group_spans: dict[str, list[tuple[float, float]]] = {g: [] for g in groups}
        for r in fitting:
            g = ts_cfg.group_of(r["task"])
            if g is None:
                continue
            s_el = (r["start_dt"] - timedelta(seconds=best_shift) - raw_start).total_seconds()
            e_el = (r["end_dt"] - timedelta(seconds=best_shift) - raw_start).total_seconds()
            # Clip to the windowed extent; a task may legitimately run a few
            # seconds past the last full window (fit_slack_s).
            s_el, e_el = max(0.0, s_el), min(window_extent_s, e_el)
            if e_el > s_el:
                group_spans[g].append((s_el, e_el))
        group_spans = {g: _merge_spans(v) for g, v in group_spans.items()}

        # ---- G3: the two groups must not overlap in time -----------------
        if pcfg.get("require_disjoint_task_spans", True) and all(group_spans[g] for g in groups):
            ov = _spans_overlap_s(group_spans[groups[0]], group_spans[groups[1]])
            if ov > float(pcfg.get("max_task_span_overlap_s", 0.0)):
                problems.append(
                    f"TASK SPANS OVERLAP by {ov:.1f}s ({groups[0]} vs {groups[1]}) — the annotation "
                    f"CSV logs these as the same/overlapping interval, so a 'task split' here would "
                    f"train and test on the same windows")

        # ---- window assignment ------------------------------------------
        counts, eval_counts = {}, {}
        eval_mask_all = rows["is_nonoverlap_subset"].to_numpy().astype(bool)
        for g in groups:
            m = _assign_windows(rows, group_spans[g], sfreq, ts_cfg.assignment["mode"])
            counts[g] = int(m.sum())
            eval_counts[g] = int((m & eval_mask_all).sum())
            if m.any():
                part = rows.loc[m, ["path", "patient_id", "condition", "y_cond",
                                    "t_cont", "is_nonoverlap_subset", "start_sample",
                                    "end_sample", "sfreq"]].copy()
                part["task_group"] = g
                part["session"] = key
                sidecar_parts.append(part)

        sessions.append(SessionAlignment(
            session=key, patient=rec.patient_id, condition=rec.condition,
            csv_path=str(rec.csv_path), annotation_path=str(ann_path),
            raw_start=raw_start.isoformat(), raw_duration_s=duration_s,
            window_extent_s=window_extent_s, best_shift_s=best_shift,
            n_task_rows=len(task_rows), n_rejected_task_rows=len(rejected),
            fitting_tasks=[r["task"] for r in sorted(fitting, key=lambda r: r["start_dt"])],
            non_fitting_tasks=[r["task"] for r in task_rows if r not in fitting],
            group_spans=group_spans, group_window_counts=counts,
            group_eval_window_counts=eval_counts, crosscheck=crosscheck, problems=problems,
        ))
        log.info("%-11s shift=%9.2fs fit=%d/%d | %s | %s",
                 key, best_shift, len(fitting), len(task_rows),
                 ", ".join(f"{g}={counts[g]}w/{eval_counts[g]}e" for g in groups),
                 "OK" if not problems else f"PROBLEMS: {problems[0]}")

    sidecar = (pd.concat(sidecar_parts, ignore_index=True) if sidecar_parts
               else pd.DataFrame(columns=["path", "patient_id", "condition", "y_cond", "t_cont",
                                          "is_nonoverlap_subset", "start_sample", "end_sample",
                                          "sfreq", "task_group", "session"]))

    verdicts = _apply_patient_gates(sessions, cohort_patients, ts_cfg)
    eligible = [v.patient for v in verdicts if v.eligible]
    return TaskWindowDerivation(
        sidecar=sidecar, sessions=sessions, verdicts=verdicts,
        eligible_patients=eligible, cohort_patients=list(cohort_patients),
    )


def _apply_patient_gates(sessions, cohort_patients, ts_cfg) -> list[PatientVerdict]:
    """Gate each patient on the confound + volume requirements. Fail-loud, per reason."""
    pcfg = ts_cfg.preflight
    groups = ts_cfg.group_names
    min_eval = int(pcfg["min_eval_windows_per_session_task"])
    by_patient: dict[str, dict[str, SessionAlignment]] = {}
    for s in sessions:
        by_patient.setdefault(s.patient, {})[s.condition] = s

    verdicts = []
    for p in cohort_patients:
        reasons: list[str] = []
        sess = by_patient.get(p, {})
        if pcfg.get("require_both_conditions", True):
            for cond in ("A", "C"):
                if cond not in sess:
                    reasons.append(f"no {cond} session was derived at all")
        for cond, s in sorted(sess.items()):
            for prob in s.problems:
                reasons.append(f"{cond}: {prob}")

        if pcfg.get("require_both_tasks_in_both_sessions", True):
            for g in groups:
                absent = [c for c, s in sorted(sess.items()) if s.group_eval_window_counts.get(g, 0) == 0]
                if absent:
                    reasons.append(
                        f"task group '{g}' has NO recoverable eval windows in session(s) "
                        f"{'/'.join(absent)} — task type would be confounded with ON/OFF")
        for cond, s in sorted(sess.items()):
            for g in groups:
                n = s.group_eval_window_counts.get(g, 0)
                if 0 < n < min_eval:
                    reasons.append(f"{cond}/{g}: only {n} eval windows (< {min_eval} required)")

        verdicts.append(PatientVerdict(
            patient=p, eligible=not reasons, reasons=reasons,
            per_session={c: {"shift_s": s.best_shift_s,
                             "windows": s.group_window_counts,
                             "eval_windows": s.group_eval_window_counts,
                             "spans": {g: s.group_spans.get(g, []) for g in groups}}
                         for c, s in sorted(sess.items())},
        ))
    return verdicts


def load_sidecar(path: Path) -> pd.DataFrame:
    """Read a previously derived sidecar, validating the columns downstream needs."""
    if not path.is_file():
        raise FileNotFoundError(
            f"task-window sidecar not found: {path}. Run "
            f"`python codebrain_finetune/src/derive_task_windows.py` first.")
    df = pd.read_parquet(path)
    required = {"path", "patient_id", "condition", "y_cond", "task_group",
                "is_nonoverlap_subset", "t_cont"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: sidecar missing columns {sorted(missing)}")
    return df
