"""Continuous (non-task-locked) windowing -- counterpart to window.py::make_windows
for recordings with no task segments to respect (the raw-signal-CSV cohort;
see cohort_discovery_csv.py / run_cohort_continuous_csv.py).

Mirrors window.py's cutting logic exactly (same `Window` dataclass, imported
unmodified, not redefined), except the span iterated is the WHOLE recording
`[0, pre.n_samples)` instead of per-task segments, and every window's task_id
is tagged with the constant placeholder `CONTINUOUS_TASK_ID` instead of a real
task id. `is_nonoverlap_subset` uses the same `offset % spw == 0` rule as
window.py, with `offset = start - 0` (session start is sample 0, so this is
just `start % spw == 0`).
"""
from __future__ import annotations

import numpy as np

from .align import AlignmentResult
from .config import Config
from .preprocess import PreprocessedRecording
from .window import Window

CONTINUOUS_TASK_ID = "continuous"


def make_continuous_windows(cfg: Config, pre: PreprocessedRecording,
                            align: AlignmentResult) -> list[Window]:
    sfreq = pre.sfreq
    spw = cfg.window.samples_per_window(sfreq)          # samples per window
    stride = cfg.window.train_stride_s * sfreq          # finest stride
    data = pre.data_uV
    n = pre.n_samples

    windows: list[Window] = []
    idx = 0
    start = 0
    while start + spw <= n:
        end = start + spw
        X = np.ascontiguousarray(data[:, start:end], dtype=np.float32)
        offset = start - 0
        is_nonoverlap = (offset % spw) == 0
        windows.append(Window(
            X=X, task_id=CONTINUOUS_TASK_ID, window_idx=idx,
            start_sample=start, end_sample=end,
            stride_s=cfg.window.train_stride_s,
            is_nonoverlap_subset=is_nonoverlap,
        ))
        idx += 1
        start += stride

    n_sub = sum(w.is_nonoverlap_subset for w in windows)
    print(f"[stage6-continuous] {align.key}: {len(windows)} windows "
          f"({n_sub} non-overlap subset) over whole recording [0,{n}), shape (18, {spw})")
    return windows
