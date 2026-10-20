"""QC audit: PSD, 1/f (aperiodic) slope, and impedance -- see the task brief
for the full spec. ADDITIVE DIAGNOSTICS ONLY: read-only against
`processed_csv_continuous/` (the production continuous-CSV cohort run,
config/qc.yaml's `source.processed_dir_name`) and the shared drive; writes
only to a new, separate output directory (`config/qc.yaml`'s `output.dir`).
Never modifies, deletes, excludes, or re-runs any preprocessing, and never
reintroduces amplitude rejection or window-exclusion logic.

Runs three checks over every channel of every session in the manifest:
  A. PSD flatness           (qc_psd.py)
  B. Aperiodic (1/f) slope  (qc_aperiodic.py, via `specparam`)
and, for a small representative sample of sessions, renders 18-channel PSD
grids and raw-vs-processed before/after grids. Separately runs an impedance
data-availability inventory (qc_impedance.py) over a handful of sample
sessions.

Run:
    python run_qc_signal_quality.py
    python run_qc_signal_quality.py --qc-config config/qc.yaml --pipeline-config config/pipeline.yaml
"""
from __future__ import annotations

import argparse
import time
import traceback

import pandas as pd

from src.livinglab_prep.config import load_config
from src.livinglab_prep.qc_aperiodic import fit_channel, find_persistent_low_exponent_patients, peaks_to_str
from src.livinglab_prep.qc_config import load_qc_config
from src.livinglab_prep.qc_impedance import run_impedance_inventory
from src.livinglab_prep.qc_psd import compute_channel_psds, plot_before_after_grid, plot_psd_grid
from src.livinglab_prep.qc_reconstruct import (
    cohort_sessions,
    find_raw_csv_paths,
    load_manifest,
    load_raw_session,
    pipeline_cfg_for_qc,
    reconstruct_processed_session,
)
from src.livinglab_prep.qc_report import write_summary_report

_CONDITION_TO_ON_OFF = {0: "ON", 1: "OFF"}


def _condition_label(cfg, condition: str) -> str:
    return _CONDITION_TO_ON_OFF[cfg.labels.condition_map[condition.upper()]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="run_qc_signal_quality")
    parser.add_argument("--qc-config", default=None, help="path to qc.yaml")
    parser.add_argument("--pipeline-config", default=None, help="path to pipeline.yaml")
    parser.add_argument("--patients", default=None,
                        help="comma-separated patient-id override for a smoke test "
                             "(default: every session found in the manifest)")
    args = parser.parse_args(argv)

    pipeline_cfg = load_config(args.pipeline_config)
    qc = load_qc_config(args.qc_config)
    cfg = pipeline_cfg_for_qc(qc, pipeline_cfg)

    print(f"[qc] reading production run at {cfg.run_dir}")
    manifest = load_manifest(cfg.run_dir)
    sessions = cohort_sessions(manifest)
    if args.patients:
        wanted = set(args.patients.split(","))
        sessions = [s for s in sessions if s[0] in wanted]
    print(f"[qc] {len(sessions)} sessions found in manifest")

    out_dir = qc.output_dir
    psd_examples_dir = out_dir / "psd_examples"
    before_after_dir = out_dir / "psd_before_after_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    rep_patients = set(qc.representative_sessions.patients)
    raw_csv_paths = find_raw_csv_paths(qc.drive_root, sorted(rep_patients))

    aperiodic_rows: list[dict] = []
    session_failures: list[tuple[str, str]] = []
    before_after_failures: list[tuple[str, str]] = []

    t_start = time.time()
    for i, (patient_id, condition) in enumerate(sessions, 1):
        key = f"{patient_id}-{condition.lower()}"
        cond_label = _condition_label(cfg, condition)
        print(f"\n[qc] === ({i}/{len(sessions)}) {key} ({cond_label}) ===")
        t0 = time.time()
        try:
            session = reconstruct_processed_session(
                cfg.run_dir, manifest, patient_id, condition, cfg.channels.keep)
            channel_psds = compute_channel_psds(session, qc)

            for cp in channel_psds:
                fit = fit_channel(cp, qc)
                aperiodic_rows.append({
                    "patient_id": patient_id, "condition": cond_label, "channel": cp.channel,
                    "exponent": fit.exponent, "offset": fit.offset, "r_squared": fit.r_squared,
                    "n_peaks": len(fit.peaks), "peaks": peaks_to_str(fit.peaks),
                    "beta_peak_present": fit.beta_peak_present,
                    "beta_cf_hz": fit.beta_cf, "beta_power": fit.beta_pw, "beta_bandwidth_hz": fit.beta_bw,
                    "psd_low_high_ratio_db": cp.low_high_ratio_db, "psd_flat_flag": cp.flat_flag,
                    "exponent_near_zero": fit.exponent_near_zero, "poor_fit": fit.poor_fit,
                    "fit_error": fit.fit_error, "flagged": fit.flagged, "flag_reason": fit.flag_reason,
                })

            if patient_id in rep_patients:
                plot_psd_grid(session, channel_psds, qc, psd_examples_dir / f"{key}.png")
                print(f"[qc] {key}: wrote {psd_examples_dir / (key + '.png')}")

                raw_csv_path = raw_csv_paths.get(key)
                if raw_csv_path is None:
                    before_after_failures.append((key, "raw-signal CSV not found on drive"))
                else:
                    try:
                        raw_session = load_raw_session(cfg, raw_csv_path, key)
                        raw_psds = compute_channel_psds(raw_session, qc)
                        plot_before_after_grid(raw_session, session, raw_psds, channel_psds,
                                              qc, before_after_dir / f"{key}.png")
                        print(f"[qc] {key}: wrote {before_after_dir / (key + '.png')}")
                    except Exception as e:
                        traceback.print_exc()
                        before_after_failures.append((key, str(e)))

        except Exception as e:
            print(f"[qc] FAILED {key}: {e}")
            traceback.print_exc()
            session_failures.append((key, str(e)))
            continue
        print(f"[qc] {key}: done in {time.time() - t0:.1f}s")

    aperiodic_df = pd.DataFrame(aperiodic_rows)
    aperiodic_csv_path = out_dir / "aperiodic_fit_results.csv"
    aperiodic_df.to_csv(aperiodic_csv_path, index=False)
    print(f"\n[qc] wrote {aperiodic_csv_path} ({len(aperiodic_df)} rows)")

    print("\n[qc] running impedance inventory ...")
    impedance_checks = run_impedance_inventory(cfg, qc)
    impedance_df = pd.DataFrame([{
        "patient_id": c.patient_id, "condition": c.condition, "source_type": c.source_type,
        "source_path": c.source_path, "impedance_found": c.impedance_found, "notes": c.notes,
    } for c in impedance_checks])
    impedance_csv_path = out_dir / "impedance_results.csv"
    impedance_df.to_csv(impedance_csv_path, index=False)
    print(f"[qc] wrote {impedance_csv_path} ({len(impedance_df)} rows)")

    print("\n[qc] checking for cross-session persistent low-exponent patients ...")
    persistent_df = find_persistent_low_exponent_patients(aperiodic_df, qc)
    if len(persistent_df):
        flagged_patients = sorted(persistent_df.index)
        print(f"[qc] {len(flagged_patients)} patient(s) flagged: {flagged_patients}")
        flagged_examples_dir = out_dir / "psd_examples_flagged_patients"
        flagged_before_after_dir = out_dir / "psd_before_after_flagged_patients"
        flagged_raw_paths = find_raw_csv_paths(qc.drive_root, flagged_patients)
        for patient_id in flagged_patients:
            for condition in ("A", "C"):
                key = f"{patient_id}-{condition.lower()}"
                try:
                    session = reconstruct_processed_session(
                        cfg.run_dir, manifest, patient_id, condition, cfg.channels.keep)
                    psds = compute_channel_psds(session, qc)
                    plot_psd_grid(session, psds, qc, flagged_examples_dir / f"{key}.png")
                    raw_path = flagged_raw_paths.get(key)
                    if raw_path is not None:
                        raw_session = load_raw_session(cfg, raw_path, key)
                        raw_psds = compute_channel_psds(raw_session, qc)
                        plot_before_after_grid(raw_session, session, raw_psds, psds, qc,
                                              flagged_before_after_dir / f"{key}.png")
                    print(f"[qc] {key}: wrote flagged-patient plot(s)")
                except Exception as e:
                    print(f"[qc] WARNING: could not render flagged-patient plot for {key}: {e}")
    else:
        print("[qc] no patient met the cross-session persistence criterion")

    write_summary_report(qc, aperiodic_df, impedance_checks, session_failures,
                        before_after_failures, out_dir / "qc_summary_report.md",
                        persistent_low_exponent_df=persistent_df)

    elapsed = time.time() - t_start
    print(f"\n[qc] done in {elapsed:.1f}s: {len(sessions) - len(session_failures)}/{len(sessions)} "
          f"sessions processed, {len(aperiodic_df)} channel-session rows, "
          f"{len(session_failures)} session failure(s)")


if __name__ == "__main__":
    main()
