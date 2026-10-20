"""Reporting & provenance (README.md §11).

Writes processed/_reports/run_summary.md (config + hashes + per-recording window
counts, reject drop fractions, alignment method/offset/drift) and limitations.md
(auto-includes the Pz-reference caveat, the 50 vs 60 Hz notch note, and any
recording flagged continuous_label_valid=false with its reason).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .align import AlignmentResult
from .config import Config
from .ica import ICAReport
from .psd import PSDCheck
from .reject import RejectStats


@dataclass
class RecordingSummary:
    key: str
    patient_id: str
    condition: str
    align: AlignmentResult
    reject: RejectStats
    psd: PSDCheck
    n_windows: int
    ica: ICAReport | None = None


def write_reports(cfg: Config, summaries: list[RecordingSummary]) -> None:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    _run_summary(cfg, summaries)
    _limitations(cfg, summaries)


def _run_summary(cfg: Config, summaries: list[RecordingSummary]) -> None:
    lines = ["# Run summary", "",
             f"- Generated: {datetime.now(timezone.utc).isoformat()}",
             f"- Config hash: `{cfg.config_hash}`",
             f"- Git SHA: `{cfg.git_sha}`",
             f"- Seed: {cfg.run.seed}",
             f"- Window: {cfg.window.length_s}s @ {cfg.signal.target_sfreq}Hz "
             f"({cfg.window.samples_per_window(cfg.signal.target_sfreq)} samples), "
             f"train_stride={cfg.window.train_stride_s}s",
             f"- Channels ({len(cfg.channels.keep)}): {', '.join(cfg.channels.keep)}",
             f"- Reject: peak-abs < {cfg.reject.threshold_uV} uV (enabled={cfg.reject.enabled})",
             "",
             "## Per-recording",
             "",
             "| recording | cond | windows | drop % | align corr (s) | fit | reconciled | cont_valid | PSD | ICA excluded |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    total = 0
    for s in summaries:
        total += s.n_windows
        ica_col = (f"{len(s.ica.excluded_idx)}/{s.ica.n_components} "
                  f"(iclabel={s.ica.iclabel_excluded_idx}, eog={s.ica.eog_excluded_idx})"
                  if s.ica is not None else "disabled")
        lines.append(
            f"| {s.key} | {s.condition} | {s.n_windows} | "
            f"{s.reject.drop_frac*100:.1f} | {s.align.correction_s:+d} | "
            f"{s.align.n_fit}/{s.align.n_usable} | {s.align.reconciled} | "
            f"{s.align.continuous_label_valid} | {'OK' if s.psd.ok else 'WARN'} | {ica_col} |")
    lines += ["", f"**Total windows: {total}** across {len(summaries)} recordings.", ""]
    (cfg.reports_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {cfg.reports_dir / 'run_summary.md'}")


def _limitations(cfg: Config, summaries: list[RecordingSummary]) -> None:
    lines = ["# Limitations & documented caveats", "",
             "Auto-generated; feeds the dissertation's 'design decisions, documented "
             "with reasoning' requirement.", "",
             "## Reference montage",
             f"{cfg.channels.reference_note}"]
    if cfg.reref.enabled:
        lines += [
            f"Re-referencing is applied as the FIRST transformation, on the raw "
            f"preloaded signal before any filtering or channel selection, using "
            f"reference channels {cfg.reref.ref_channels} (scheme "
            f"'{cfg.reref.scheme}'). The data was recorded against a single scalp "
            f"electrode (Pz); subtracting the mean of the ear channels cancels the "
            f"common Pz term algebraically and replaces it with a quiet off-scalp "
            f"linked-ears reference. This removes the single-electrode reference "
            f"noise that was injected (sign-flipped) into every channel and "
            f"inflated the whole montage at once. Outputs for this scheme are "
            f"isolated in `{cfg.run_dir.name}/` so alternative reference choices "
            f"never overwrite each other.",
            "",
        ]
    else:
        lines += [
            "No re-referencing applied: the recorded single-Pz reference is kept "
            "(reref.ref_channels is empty). CodeBrain was pretrained on a REF/"
            "linked-ear monopolar montage; a Pz-referenced montage is a genuine "
            "distributional difference that biases *which channels express which "
            "markers*.",
            "",
        ]
    lines += ["## ICA artifact/blink removal"]
    if cfg.ica.enabled:
        lines += [
            f"Enabled: extended infomax ICA ({cfg.ica.n_components} components, "
            f"seed={cfg.ica.random_state}), fit per recording on a "
            f"{cfg.ica.fit_highpass_hz} Hz high-passed, {cfg.ica.fit_sfreq} Hz copy, "
            f"applied to the {cfg.signal.bandpass.l_freq}-{cfg.signal.bandpass.h_freq} Hz "
            "data. Two independent, unioned exclusion rules: (1) ICLabel winning-label "
            f"in {cfg.ica.exclude_labels} with probability >= {cfg.ica.iclabel_min_prob}; "
            f"(2) eye blinks via frontal-channel correlation "
            f"(`ICA.find_bads_eog`, channels {cfg.ica.eog_channels}, "
            f"|r| >= {cfg.ica.eog_corr_threshold}) rather than ICLabel's label -- ICLabel "
            "runs off-distribution on this linked-ears/mobile/Parkinsonian data and was "
            "found to mislabel real blink components 'brain'/'other' (winning probability "
            "0.10-0.25). Per-recording removals are in `ica_report_<key>.json`.",
            "",
        ]
    else:
        lines += ["Disabled for this run (`ica.enabled: false`); no artifact removal applied.", ""]
    lines += [
             "## Mains notch (50 Hz vs pretraining 60 Hz)",
             f"We notch at {cfg.signal.notch_freq} Hz (UK mains). CodeBrain/CBraMod "
             "pretraining used 60 Hz (US recordings). 50 Hz is precedented in the same "
             "codebase for European data (CBraMod preprocessing_mumtaz).",
             "",
             "## Amplitude rejection",
             (f"DISABLED for this run: all windows retained regardless of the "
              f"{cfg.reject.threshold_uV} uV whole-window peak-abs threshold. Under the "
              f"original single-Pz reference this threshold discarded ~98% of windows "
              f"(23 of 1352 survived), which motivated the linked-ears re-referencing in "
              f"this build; the would-be rejection at {cfg.reject.threshold_uV} uV under "
              f"the current reference scheme ('{cfg.reref.scheme}') is measured separately "
              f"(see reref_report.py) and the threshold decision is deferred to the cohort "
              f"phase. Retaining all windows keeps the correctness harness intact."
              if not cfg.reject.enabled else
              f"Enabled: whole-window peak-abs < {cfg.reject.threshold_uV} uV (single "
              f"global constant, non-adaptive; CBraMod pretraining convention)."),
             "",
             "## Alignment anchor",
             "No EDF annotations or usable trigger markers exist in this dataset, so "
             "task->EEG alignment relies solely on wall-clock (meas_date) with a "
             "whole-hour timezone/DST correction chosen by task-fit. Sub-second offsets "
             "are NOT fitted from the tasks themselves (circular).",
             ""]

    invalid = [s for s in summaries if not s.align.continuous_label_valid]
    partial = [s for s in summaries if s.align.continuous_label_valid and not s.align.reconciled]
    high_drop = [s for s in summaries if s.reject.drop_frac > cfg.reject.warn_drop_frac]

    lines += ["## Per-recording continuous-label validity"]
    if invalid:
        for s in invalid:
            lines.append(f"- **{s.key}: continuous_label_valid=FALSE** — {s.align.note} "
                         "Windows retained with binary condition label; continuous target masked.")
    else:
        lines.append("- All recordings have a resolved constant-offset alignment "
                     "(continuous_label_valid=true).")
    if partial:
        lines += ["", "### Partial alignment (some tasks outside recording, clamped)"]
        for s in partial:
            lines.append(f"- {s.key}: {s.align.note}")
    if high_drop:
        lines += ["", "### High amplitude-rejection drop fractions"]
        for s in high_drop:
            lines.append(f"- {s.key}: dropped {s.reject.drop_frac:.1%} of windows "
                         f"(> {cfg.reject.warn_drop_frac:.0%}); may interact with Pz reference.")
    lines.append("")
    (cfg.reports_dir / "limitations.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {cfg.reports_dir / 'limitations.md'}")
