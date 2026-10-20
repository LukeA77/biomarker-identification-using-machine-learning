"""Pipeline orchestrator: runs Stages 1-9 per recording, then reports.

Stages are coupled in memory (the continuous signal flows straight from load to
windowing), so the whole per-recording chain runs in one pass. Stage 0 (preflight)
and Stage 12 (validation) are separate entry points in the CLI. Determinism
(§5.9): recordings are processed in a stable discovery order and windows are
written in sequential order, so identical config+data -> identical manifest.
"""
from __future__ import annotations

import json
import random
import traceback

import numpy as np

from .align import align_recording
from .config import Config
from .discovery import discover_recordings, Recording
from .label import label_windows
from .preprocess import preprocess_recording
from .provenance import write_provenance
from .psd import compute_psd_check
from .reject import reject_windows
from .report import RecordingSummary, write_reports
from .serialize import Serializer
from .window import make_windows


def run_pipeline(cfg: Config, recordings: list[Recording] | None = None) -> list[RecordingSummary]:
    """Run Stages 1-9 over ``recordings`` (default: the pilot's flat-dir
    discovery, `discover_recordings(cfg)`, for backward compatibility).

    Passing an explicit list (e.g. from
    `cohort_discovery.discover_cohort_recordings`) is how the cohort-scale run
    reuses this exact orchestration without touching pilot behavior.
    """
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)

    if recordings is None:
        recordings = discover_recordings(cfg)
    print(f"[pipeline] discovered {len(recordings)} recordings: "
          f"{[r.key for r in recordings]}")
    print(f"[pipeline] reference scheme='{cfg.reref.scheme}' "
          f"(ref_channels={cfg.reref.ref_channels or 'recorded'}) -> {cfg.run_dir}")
    write_provenance(cfg, recordings)

    serializer = Serializer(cfg)
    summaries: list[RecordingSummary] = []
    failed: list[tuple[str, str]] = []

    for rec in recordings:
        print(f"\n[pipeline] === {rec.key} ({rec.condition}) ===")
        try:
            pre = preprocess_recording(cfg, rec)          # Stages 1-3 (+2b ICA)
            psd = compute_psd_check(cfg, pre)             # Stage 4
            align = align_recording(cfg, pre)            # Stage 5
            windows = make_windows(cfg, pre, align)       # Stage 6
            labeled = label_windows(cfg, pre, align, windows)  # Stage 7
            kept, rstats = reject_windows(cfg, rec.key, labeled)  # Stage 8
            n = serializer.write_recording(pre, kept)     # Stage 9
            _write_ica_report(cfg, pre)
        except Exception as e:
            print(f"[pipeline] FAILED {rec.key}: {e}")
            traceback.print_exc()
            failed.append((rec.key, str(e)))
            continue

        summaries.append(RecordingSummary(
            key=rec.key, patient_id=rec.patient_id, condition=rec.condition,
            align=align, reject=rstats, psd=psd, n_windows=n, ica=pre.ica_report))

    serializer.finalize()
    write_reports(cfg, summaries)
    if failed:
        print(f"\n[pipeline] {len(failed)}/{len(recordings)} recordings FAILED "
              f"(logged and skipped, others unaffected): {failed}")
    return summaries


def _write_ica_report(cfg: Config, pre) -> None:
    """Write <reports_dir>/ica_report_<key>.json (component removals, provenance)."""
    if pre.ica_report is None:
        return
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.reports_dir / f"ica_report_{pre.rec.key}.json"
    out.write_text(json.dumps(pre.ica_report.as_dict(), indent=2), encoding="utf-8")
    print(f"[pipeline] wrote {out}")
