"""Run Stages 1-3 + a NEW continuous (non-task-locked) windowing stage over the
38-patient / 76-session raw-signal-CSV cohort on the shared LivingLab_PD drive.

ADDITIVE-ONLY, parallel to (and does not touch) `run_cohort.py`: nothing under
`src/livinglab_prep/`, `config/pipeline.yaml`, or `run_cohort.py` is modified.
This script imports and reuses the existing Stage 1-3/6-9 functions unchanged
(`apply_reference`, `filter_`, `clean_with_ica`, `select_channels`,
`resample_selected`, `extract_uV`, `label_windows`, `reject_windows`,
`Serializer`, `write_reports`, `_write_ica_report`) and only supplies NEW
pieces where the EDF-based pipeline has no CSV-shaped counterpart:

  - cohort_discovery_csv.discover_csv_cohort   (find each session's raw-signal
    CSV by content-sniff, NOT the EDF, NOT the annotation/ task CSV)
  - csv_signal.load_raw_csv                    (STAGE 1: build an mne RawArray
    from the CSV, with channel names in the same "EEG {name}-Pz" convention
    read_raw_edf produces -- see "Why this works" below)
  - continuous_align.make_continuous_alignment (a trivial always-valid
    AlignmentResult; continuous windowing has no cross-clock problem to solve)
  - window_continuous.make_continuous_windows  (STAGE 6 variant: windows the
    WHOLE recording instead of per-task segments)
  - psd_csv.compute_psd_check_csv              (Stage 4 variant: psd.py's
    compute_psd_check needs rec.edf_path, which this cohort's CsvRecording
    does not have -- see psd_csv.py's docstring)

Explicitly SKIPPED (not applicable / not called):
  - Stage 0 preflight (task-duration check; no task locking here)
  - Stage 5 clock alignment (align.py::align_recording; superseded by the
    dummy continuous AlignmentResult)
  - amplitude rejection stays governed by config (reject.enabled: false); this
    script does not touch that setting
  - provenance.write_provenance: reads rec.edf_path (CsvRecording has none);
    this script writes its own lightweight provenance.json instead (see
    `_write_provenance_csv` below) rather than modifying provenance.py

Why this works with zero changes to existing files (the channel-naming trick):
`reference.py`, `ica.py`, and `preprocess.py::select_channels` all look for
channels via `cfg.channels.rename_from_pattern.format(name=std)` (e.g.
"EEG Fp1-Pz") -- the raw EDF label convention. `csv_signal.load_raw_csv`
constructs its RawArray with channel names ALREADY in that convention, so
`apply_reference`, `filter_`, `clean_with_ica`, and `select_channels` all work
on it unmodified: they only ever look up channel names, never the file format.

Output isolation: this run's output tree is rooted at NEW_PROCESSED_DIR_NAME
(`processed_csv_continuous/`, a directory distinct from the existing
`processed/` tree used by the 18-patient EDF cohort), obtained via
`Config.model_copy` (a standard pydantic method, not a config.py edit) so
`Serializer`/`report.write_reports`/`psd_csv` -- which all read
`cfg.run_dir`/`cfg.windows_dir`/`cfg.manifest_path`/`cfg.reports_dir` off the
`Config` instance they're given -- resolve under the new root automatically,
with no risk of colliding with `processed/reref-linkedears_ica/`.

Run:
    python run_cohort_continuous_csv.py --root "C:\\Data\\LivingLab_PD"
    python run_cohort_continuous_csv.py --root <path> --patients sub-131,sub-134   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import mne

from src.livinglab_prep.cohort_discovery_csv import CsvRecording, discover_csv_cohort
from src.livinglab_prep.config import Config, load_config
from src.livinglab_prep.continuous_align import make_continuous_alignment
from src.livinglab_prep.csv_signal import load_raw_csv
from src.livinglab_prep.ica import clean_with_ica
from src.livinglab_prep.label import label_windows
from src.livinglab_prep.pipeline import _write_ica_report
from src.livinglab_prep.preprocess import (
    PreprocessedRecording,
    extract_uV,
    filter_,
    resample_selected,
    select_channels,
)
from src.livinglab_prep.psd_csv import compute_psd_check_csv
from src.livinglab_prep.reference import apply_reference
from src.livinglab_prep.reject import reject_windows
from src.livinglab_prep.report import RecordingSummary, write_reports
from src.livinglab_prep.serialize import Serializer
from src.livinglab_prep.window_continuous import make_continuous_windows

mne.set_log_level("ERROR")

# The 41-patient cohort pre-vetted (read-only audit) to each have exactly ONE
# unambiguous raw-signal CSV per condition, all required channels, 300 Hz, no
# structural corruption.
#
# 2026-07-22: added sub-42, sub-70, sub-121 -- their original "ambiguous file
# selection" exclusion was resolved (see cohort_discovery_csv.py docstring /
# MANUAL_FILE_OVERRIDES and logs/pre_finetuning_audit_20260721.md): sub-42-c's
# ambiguity was just a different task (UPDRS exam) now excluded by filename
# pattern; sub-70-c and sub-121-a had a genuine two-candidate ambiguity
# resolved by an explicit, human-confirmed manual override.
#
# Still excludes sub-78/82/100/111/130 (sub-78: 3-way filename ambiguity,
# confirmed excluded 2026-07-22; sub-82/100: genuine EEG-acquisition
# gap, no data ever recorded; sub-111: incompatible 13-channel montage;
# sub-130: EDF/CSV for one condition only).
PATIENTS = [
    "sub-34", "sub-42", "sub-48", "sub-51", "sub-53", "sub-62", "sub-69", "sub-70",
    "sub-71", "sub-72", "sub-73", "sub-76", "sub-77", "sub-80", "sub-81", "sub-84",
    "sub-90", "sub-92", "sub-93", "sub-94", "sub-96", "sub-97", "sub-98", "sub-99",
    "sub-103", "sub-105", "sub-106", "sub-107", "sub-108", "sub-114", "sub-115",
    "sub-116", "sub-117", "sub-119", "sub-121", "sub-123", "sub-125", "sub-129",
    "sub-131", "sub-134", "sub-135",
]

# New, distinct output root -- see module docstring "Output isolation".
NEW_PROCESSED_DIR_NAME = "processed_csv_continuous"


def _cfg_for_csv_continuous(cfg: Config) -> Config:
    """Config whose run_dir/windows_dir/manifest_path/reports_dir resolve under
    a NEW output root, so this run can never collide with the existing
    18-patient EDF cohort's `processed/reref-linkedears_ica/` output. Built via
    `Config.model_copy` (standard pydantic API) -- config.py itself is
    untouched, and every non-path field (channels/reref/signal/ica/window/
    reject/labels/align config) is identical to `cfg`.
    """
    new_paths = cfg.paths.model_copy(update={"processed_dir": NEW_PROCESSED_DIR_NAME})
    return cfg.model_copy(update={"paths": new_paths})


def preprocess_recording_csv(cfg: Config, rec: CsvRecording,
                             ica_report_dir: Path | None = None) -> PreprocessedRecording:
    """CSV counterpart to preprocess.py::preprocess_recording. Mirrors its
    Stage 1 -> 1a -> 2 -> 2b -> 1b -> 3 order EXACTLY (see that function's
    docstring for the strict-order rationale), swapping only the Stage 1 load
    step (load_raw_csv instead of load_raw).
    """
    raw = load_raw_csv(cfg, rec.csv_path, rec.key)     # Stage 1 (CSV)
    meas_date = raw.info.get("meas_date")               # capture early, same as preprocess_recording
    raw = apply_reference(cfg, raw, rec.key)            # Stage 1a (A1/A2 still present)
    raw = filter_(cfg, raw, rec.key)                    # Stage 2 (full channel set)

    ica_report = None
    if cfg.ica.enabled:
        if ica_report_dir is None:
            ica_report_dir = cfg.reports_dir / "ica_plots"
        raw, ica_report = clean_with_ica(cfg, raw, rec.key,   # Stage 2b -> 18ch
                                         report_dir=ica_report_dir)
    else:
        raw = select_channels(cfg, raw, rec.key)         # Stage 1b -> 18ch

    raw = resample_selected(cfg, raw, rec.key)           # Stage 3
    data = extract_uV(cfg, raw, rec.key)
    return PreprocessedRecording(
        rec=rec, data_uV=data, ch_names=list(raw.ch_names),
        sfreq=cfg.signal.target_sfreq, n_samples=data.shape[1],
        meas_date=meas_date, ica_report=ica_report,
    )


def _write_provenance_csv(cfg: Config, recordings: list[CsvRecording]) -> Path:
    """Lightweight provenance.json for this run (CsvRecording has no edf_path,
    so provenance.py::write_provenance -- which embeds `rec.edf_path` -- is not
    reusable unmodified; see module docstring). New, additive, does not touch
    provenance.py.
    """
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run": "csv_continuous_cohort",
        "reference_scheme": cfg.reref.scheme,
        "reref_enabled": cfg.reref.enabled,
        "ref_channels": list(cfg.reref.ref_channels),
        "git_sha": cfg.git_sha,
        "config_hash": cfg.config_hash,
        "run_dir": str(cfg.run_dir),
        "inputs": [
            {"key": rec.key, "patient_id": rec.patient_id,
             "condition": rec.condition, "csv": str(rec.csv_path)}
            for rec in recordings
        ],
        "config": cfg.model_dump(mode="json"),
    }
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.run_dir / "provenance.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[provenance] wrote {out}")
    return out


def run(cfg: Config, recordings: list[CsvRecording]) -> list[RecordingSummary]:
    _write_provenance_csv(cfg, recordings)
    serializer = Serializer(cfg)
    summaries: list[RecordingSummary] = []
    failed: list[tuple[str, str]] = []

    for i, rec in enumerate(recordings, 1):
        print(f"\n[run_csv_continuous] === ({i}/{len(recordings)}) {rec.key} ({rec.condition}) ===")
        t0 = time.time()
        try:
            pre = preprocess_recording_csv(cfg, rec)              # Stages 1-3 (+2b ICA)
            psd = compute_psd_check_csv(cfg, pre)                 # Stage 4 (CSV variant)
            align = make_continuous_alignment(rec.key)            # dummy Stage 5
            windows = make_continuous_windows(cfg, pre, align)    # Stage 6 (continuous)
            labeled = label_windows(cfg, pre, align, windows)     # Stage 7
            kept, rstats = reject_windows(cfg, rec.key, labeled)  # Stage 8 (no-op: disabled)
            n = serializer.write_recording(pre, kept)             # Stage 9
            _write_ica_report(cfg, pre)
        except Exception as e:
            print(f"[run_csv_continuous] FAILED {rec.key}: {e}")
            traceback.print_exc()
            failed.append((rec.key, str(e)))
            continue

        summaries.append(RecordingSummary(
            key=rec.key, patient_id=rec.patient_id, condition=rec.condition,
            align=align, reject=rstats, psd=psd, n_windows=n, ica=pre.ica_report))
        print(f"[run_csv_continuous] {rec.key}: done in {time.time() - t0:.1f}s, {n} windows")

    manifest_path = serializer.finalize()
    write_reports(cfg, summaries)
    print(f"\n[run_csv_continuous] manifest -> {manifest_path}")
    if failed:
        print(f"\n[run_csv_continuous] {len(failed)}/{len(recordings)} recordings FAILED "
              f"(logged and skipped, others unaffected): {failed}")
    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="run_cohort_continuous_csv")
    parser.add_argument("--root", required=True, help="root of the LivingLab_PD shared drive")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    parser.add_argument("--patients", default=None,
                        help="comma-separated patient-id override (default: the 38-patient cohort)")
    args = parser.parse_args(argv)

    cfg_base = load_config(args.config)
    cfg = _cfg_for_csv_continuous(cfg_base)
    root = Path(args.root)
    patients = args.patients.split(",") if args.patients else PATIENTS

    recordings, missing = discover_csv_cohort(root, patients)
    print(f"[run_csv_continuous] config_hash={cfg.config_hash} git_sha={cfg.git_sha}")
    print(f"[run_csv_continuous] ica.enabled={cfg.ica.enabled} -> run_dir={cfg.run_dir}")
    print(f"[run_csv_continuous] found {len(recordings)}/{2 * len(patients)} recordings under {root}")
    if missing:
        print(f"[run_csv_continuous] MISSING/AMBIGUOUS (STOP AND CONFIRM, not silently skipped): {missing}")

    t_start = time.time()
    summaries = run(cfg, recordings)
    elapsed = time.time() - t_start
    total_windows = sum(s.n_windows for s in summaries)
    print(f"\n[run_csv_continuous] {len(summaries)}/{len(recordings)} recordings succeeded, "
          f"{total_windows} total windows, {elapsed:.1f}s elapsed")


if __name__ == "__main__":
    main()
