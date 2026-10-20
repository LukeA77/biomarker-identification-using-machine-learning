"""Supervisor-facing evidence plots for the cross-session-persistent
low-exponent patients found in `qc_summary_report.md`'s follow-up section
(see `qc_aperiodic.find_persistent_low_exponent_patients`).

Renders one figure per flagged patient (one panel per available condition):
all 18 of that patient's channels overlaid against a "what normal EEG looks
like" reference band (10th-90th percentile power, pooled across every
channel of `qc.representative_sessions.patients` -- the same clean-patient
pool used elsewhere in this audit). The reference band makes the amplitude
deficit visible without requiring the reader to know what an aperiodic
exponent is.

ADDITIVE DIAGNOSTICS ONLY: read-only against `processed_csv_continuous/` and
the shared drive; writes only under `qc.output_dir/flagged_patient_evidence/`.

Run:
    python run_qc_flagged_patient_evidence.py
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.livinglab_prep.config import load_config
from src.livinglab_prep.qc_aperiodic import find_persistent_low_exponent_patients
from src.livinglab_prep.qc_config import load_qc_config
from src.livinglab_prep.qc_psd import compute_channel_psds, plot_patient_vs_reference, reference_band
from src.livinglab_prep.qc_reconstruct import (
    load_manifest,
    pipeline_cfg_for_qc,
    reconstruct_processed_session,
)

_COND_CODE = {"ON": "A", "OFF": "C"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="run_qc_flagged_patient_evidence")
    parser.add_argument("--qc-config", default=None, help="path to qc.yaml")
    parser.add_argument("--pipeline-config", default=None, help="path to pipeline.yaml")
    args = parser.parse_args(argv)

    pipeline_cfg = load_config(args.pipeline_config)
    qc = load_qc_config(args.qc_config)
    cfg = pipeline_cfg_for_qc(qc, pipeline_cfg)
    manifest = load_manifest(cfg.run_dir)

    aperiodic_path = qc.output_dir / "aperiodic_fit_results.csv"
    if not aperiodic_path.exists():
        raise FileNotFoundError(f"{aperiodic_path} not found -- run run_qc_signal_quality.py first")
    aperiodic_df = pd.read_csv(aperiodic_path)

    flagged = sorted(find_persistent_low_exponent_patients(aperiodic_df, qc).index)
    if not flagged:
        print("[qc-evidence] no cross-session persistent low-exponent patients found; nothing to plot")
        return
    print(f"[qc-evidence] flagged patients: {flagged}")

    ref_patients = list(qc.representative_sessions.patients)
    print(f"[qc-evidence] building 'normal EEG' reference band from: {ref_patients}")
    ref_psds_by_cond: dict[str, list] = {"ON": [], "OFF": []}
    for patient_id in ref_patients:
        for cond_label, cond_code in _COND_CODE.items():
            session = reconstruct_processed_session(
                cfg.run_dir, manifest, patient_id, cond_code, cfg.channels.keep)
            ref_psds_by_cond[cond_label].extend(compute_channel_psds(session, qc))
    condition_reference = {c: reference_band(psds) for c, psds in ref_psds_by_cond.items()
                          if psds}

    out_dir = qc.output_dir / "flagged_patient_evidence"
    for patient_id in flagged:
        condition_psds: dict[str, list] = {}
        condition_stats: dict[str, str] = {}
        for cond_label, cond_code in _COND_CODE.items():
            try:
                session = reconstruct_processed_session(
                    cfg.run_dir, manifest, patient_id, cond_code, cfg.channels.keep)
            except Exception as e:
                print(f"[qc-evidence] {patient_id}-{cond_label}: could not reconstruct ({e}), skipping panel")
                continue
            psds = compute_channel_psds(session, qc)
            condition_psds[cond_label] = psds

            rows = aperiodic_df[(aperiodic_df.patient_id == patient_id)
                               & (aperiodic_df.condition == cond_label)]
            n_weak = int((rows["exponent"] < qc.aperiodic.flat_candidate_exponent_threshold).sum())
            ref_freqs, _, ref_median, _ = condition_reference[cond_label]
            low_band_bin = 1  # first non-DC Welch bin, ~lowest plotted frequency
            condition_stats[cond_label] = (
                f"median offset (log10 power proxy): {rows['offset'].median():.2f}\n"
                f"clean reference median power @ {ref_freqs[low_band_bin]:.2f} Hz: "
                f"{ref_median[low_band_bin]:.3g} uV^2/Hz\n"
                f"mean exponent: {rows['exponent'].mean():.2f}\n"
                f"weak channels (exp<{qc.aperiodic.flat_candidate_exponent_threshold:g}): {n_weak}/18\n"
                f"beta-band peak detected: {100 * rows['beta_peak_present'].mean():.0f}% of channels"
            )

        if condition_psds:
            plot_patient_vs_reference(patient_id, condition_psds, condition_reference,
                                     condition_stats, qc, out_dir / f"{patient_id}_evidence.png")
            print(f"[qc-evidence] wrote {out_dir / (patient_id + '_evidence.png')}")

    print(f"\n[qc-evidence] done: {len(flagged)} patient evidence plot(s) in {out_dir}")


if __name__ == "__main__":
    main()
