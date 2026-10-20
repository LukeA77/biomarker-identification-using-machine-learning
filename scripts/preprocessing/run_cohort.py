"""Run the full Stage 0-9 pipeline (with the fixed ICA step) against the real
18-patient cohort on the shared LivingLab_PD drive.

This is the cohort-scale counterpart to `python -m src.livinglab_prep.cli all`,
which only understands the flat 2-patient pilot layout. Here, discovery comes
from `cohort_discovery.discover_cohort_recordings` (the nested per-patient
`eeg/`+`annotation/` layout) instead, and the exact same `run_preflight` /
`run_pipeline` orchestration is reused unchanged (see
SERVER_HANDOFF_full_dataset_preprocessing.md §2/§4/§6).

Because `cfg.ica.enabled` is true, `cfg.run_dir` resolves to the `_ica`-suffixed
sibling directory (see config.py::Config.run_dir) -- this run can never
overwrite the pilot's existing non-ICA `processed/reref-linkedears/` output.

Run:
    python run_cohort.py --root "C:\\Data\\LivingLab_PD"
    python run_cohort.py --root <path> --patients sub-131,sub-134   # smoke test
    python run_cohort.py --root <path> --skip-preflight              # rerun only Stage 1-9
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.livinglab_prep.cohort_discovery import discover_cohort_recordings
from src.livinglab_prep.config import load_config
from src.livinglab_prep.pipeline import run_pipeline
from src.livinglab_prep.preflight import run_preflight
from src.livinglab_prep.validate import run_validation

import mne

mne.set_log_level("ERROR")

# The 18-patient cohort with COMPLETE ON/OFF EDF+CSV pairs (2026-07-06 audit,
# reconfirmed 2026-07-13). Same allowlist as cohort_reject_sweep.py::COHORT_PATIENTS
# -- excludes incomplete-pair patients (sub-70, sub-71, sub-130) and all DSI-only
# patients (a different sub-study sharing the drive).
COHORT_PATIENTS = [
    "sub-34", "sub-42", "sub-48", "sub-51", "sub-53",
    "sub-111", "sub-114", "sub-115", "sub-116", "sub-117", "sub-119", "sub-121",
    "sub-123", "sub-125", "sub-129", "sub-131", "sub-134", "sub-135",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="run_cohort")
    parser.add_argument("--root", required=True, help="root of the LivingLab_PD shared drive")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    parser.add_argument("--patients", default=None,
                        help="comma-separated patient-id override (default: the 18-patient cohort)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="skip Stage 0 (useful when re-running only Stage 1-9 on a subset)")
    parser.add_argument("--skip-validate", action="store_true",
                        help="skip Stage 12 (useful for a quick smoke test)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(args.root)
    patients = args.patients.split(",") if args.patients else COHORT_PATIENTS

    recordings, missing = discover_cohort_recordings(root, patients)
    print(f"[run_cohort] config_hash={cfg.config_hash} git_sha={cfg.git_sha}")
    print(f"[run_cohort] ica.enabled={cfg.ica.enabled} -> run_dir={cfg.run_dir}")
    print(f"[run_cohort] found {len(recordings)}/{2 * len(patients)} recordings under {root}")
    if missing:
        print(f"[run_cohort] MISSING (no EDF+CSV under either task-folder naming): {missing}")
        print("[run_cohort] STOP AND CONFIRM: a missing pair can mean DSI-only or a "
              "genuine gap -- do not treat this as a silent skip.")

    if not args.skip_preflight:
        run_preflight(cfg, recordings=recordings)

    run_pipeline(cfg, recordings=recordings)

    if not args.skip_validate:
        run_validation(cfg)


if __name__ == "__main__":
    main()
