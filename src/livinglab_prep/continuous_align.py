"""Trivial always-valid AlignmentResult for continuous (non-task-locked)
windowing (see window_continuous.py).

Stage 5 (align.py) exists to map a task-PC wall-clock timestamp onto an EEG
sample index -- a real problem only because task segments are defined by an
independent clock. Continuous windowing has no such problem: every window's
continuous target is just elapsed samples from session start (label.py's
`continuous_reference: session_start`), which is exact by construction and
needs no cross-clock reconciliation. `label_windows` (label.py, unmodified)
only reads `align.continuous_label_valid` and `align.key` off its `align`
argument (verified by reading label.py); this module builds the dummy
AlignmentResult (align.py's dataclass, unmodified) label_windows needs, with
`continuous_label_valid=True` since the continuous target is always exact here.
"""
from __future__ import annotations

from .align import AlignmentResult

CONTINUOUS_METHOD = "continuous_no_task_locking"


def make_continuous_alignment(key: str) -> AlignmentResult:
    return AlignmentResult(
        key=key,
        method=CONTINUOUS_METHOD,
        offset_s=0.0,
        correction_s=0,
        n_fit=0,
        n_usable=0,
        fit_frac=1.0,
        reconciled=True,
        continuous_label_valid=True,
        note="continuous windowing: no task-CSV alignment needed",
        aligned_tasks=[],
    )
