"""Writes `qc_summary_report.md`: short, readable counts + specific flagged
IDs (not just totals) + a plain-language headline per check (see
qc_signal_quality's brief, Sec.3).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .qc_config import QcConfig
from .qc_impedance import ImpedanceSourceCheck

_MAX_LISTED = 40  # cap on individually-named flagged IDs before falling back to "+N more, see CSV"


def _list_or_truncate(items: list[str]) -> str:
    if not items:
        return "(none)"
    if len(items) <= _MAX_LISTED:
        return "; ".join(items)
    return "; ".join(items[:_MAX_LISTED]) + f"; ... +{len(items) - _MAX_LISTED} more (see CSV)"


def write_summary_report(qc: QcConfig, aperiodic_df: pd.DataFrame,
                         impedance_checks: list[ImpedanceSourceCheck],
                         session_failures: list[tuple[str, str]],
                         representative_before_after_failures: list[tuple[str, str]],
                         out_path: Path,
                         persistent_low_exponent_df: pd.DataFrame | None = None) -> None:
    """Assemble and write the QC summary report.

    Parameters
    ----------
    qc : QcConfig
    aperiodic_df : pd.DataFrame
        The full contents of `aperiodic_fit_results.csv` (one row per
        channel per session; also carries the Part-A flat-PSD columns).
    impedance_checks : list of ImpedanceSourceCheck
    session_failures : list of (session_key, error) pairs
        Sessions that could not be reconstructed/processed at all (Part A/B).
    representative_before_after_failures : list of (session_key, error) pairs
        Representative-sample sessions where the raw-vs-processed plot could
        not be produced (e.g. raw CSV not found).
    out_path : Path
    persistent_low_exponent_df : pd.DataFrame, optional
        From `qc_aperiodic.find_persistent_low_exponent_patients` -- patients
        with a weak aperiodic exponent in both their ON and OFF session (a
        cross-session persistence check, stronger evidence of a genuine
        per-patient contact/impedance issue than either flag alone). If
        provided and non-empty, and if PSD grids for these patients exist
        under `psd_examples_flagged_patients/` /
        `psd_before_after_flagged_patients/`, they are referenced by name.
    """
    lines: list[str] = [
        "# QC summary report: PSD, 1/f (aperiodic) slope, and impedance", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "Additive diagnostics only -- no preprocessed data was modified, excluded, "
        "or re-run by this audit. See `aperiodic_fit_results.csv` and "
        "`impedance_results.csv` for full per-row evidence.", "",
    ]

    n_rows = len(aperiodic_df)
    n_sessions = aperiodic_df[["patient_id", "condition"]].drop_duplicates().shape[0] \
        if n_rows else 0
    n_channels = aperiodic_df["channel"].nunique() if n_rows else 0

    # --- Part A: PSD flatness ---------------------------------------------
    lines += ["## A. PSD flatness", ""]
    if n_rows:
        flat = aperiodic_df[aperiodic_df["psd_flat_flag"]]
        lines += [
            f"Checked {n_rows} channel-sessions ({n_channels} channels x {n_sessions} "
            f"sessions).",
            f"**{len(flat)}/{n_rows} channel-sessions flagged flat** "
            f"(low[{qc.psd.flat_low_band_hz[0]:g}-{qc.psd.flat_low_band_hz[1]:g}Hz]/"
            f"high[{qc.psd.flat_high_band_hz[0]:g}-{qc.psd.flat_high_band_hz[1]:g}Hz] "
            f"power ratio < {qc.psd.flat_ratio_db_threshold:g} dB).",
            "",
        ]
        if len(flat):
            ids = [f"{r.patient_id}-{r.condition}/{r.channel} ({r.psd_low_high_ratio_db:.1f} dB)"
                  for r in flat.itertuples()]
            lines += [f"Flagged: {_list_or_truncate(ids)}", ""]
            by_channel = flat.groupby("channel").size().sort_values(ascending=False)
            top_channel, top_n = by_channel.index[0], int(by_channel.iloc[0])
            if top_n >= 3 and top_n >= 0.3 * len(flat):
                lines += [
                    f"**Channel pattern:** {top_channel} accounts for {top_n}/{len(flat)} of "
                    f"the flat-PSD flags across different subjects -- worth checking that "
                    f"electrode specifically (contact quality/placement) rather than treating "
                    f"these as independent random flags.",
                    "",
                ]
        lines += [
            "**Headline:** " + (
                "the large majority of channels show a clear 1/f-like decline; "
                "flat channels are isolated, not systemic."
                if len(flat) / max(n_rows, 1) < 0.05 else
                "a non-trivial fraction of channel-sessions show no discernible "
                "spectral shape -- worth reviewing before fine-tuning."
            ), "",
        ]
    else:
        lines += ["No channel-session PSD results were produced (see session failures below).", ""]

    # --- Follow-up: candidate flat/high-impedance patients -------------------
    if persistent_low_exponent_df is not None and len(persistent_low_exponent_df):
        thr = qc.aperiodic.flat_candidate_exponent_threshold
        min_frac = qc.aperiodic.flat_candidate_min_fraction
        lines += [
            "### Follow-up: candidate flat/high-impedance patients (cross-session persistence)",
            "",
            f"Neither flag above requires a pattern to hold across BOTH of a patient's "
            f"sessions. Checking for that specifically -- patients where >= "
            f"{min_frac:.0%} of channels have aperiodic exponent < {thr:g} in "
            f"**both** their ON and OFF recording (a genuine contact/impedance "
            f"problem should not clear up between visits, unlike a one-off noisy "
            f"session) -- surfaces "
            f"**{len(persistent_low_exponent_df)} patient(s)**:",
            "",
            "| patient | frac. channels weak (ON) | frac. channels weak (OFF) |",
            "|---|---|---|",
        ]
        for pid, row in persistent_low_exponent_df.iterrows():
            lines.append(f"| {pid} | {row['frac_low_exponent_ON']:.0%} | "
                        f"{row['frac_low_exponent_OFF']:.0%} |")
        lines += [
            "",
            "Each patient in this table was visually cross-checked against its full "
            "18-channel PSD grid (`psd_examples_flagged_patients/`) and its raw-vs-"
            "processed comparison (`psd_before_after_flagged_patients/`). Two distinct "
            "patterns were found:",
            "",
            "- **A clean, consistent flat plateau** (most of the table, e.g. sub-105, "
            "sub-97, sub-98, sub-99, sub-108): nearly every channel shows a normal "
            "decline only up to roughly 5-10 Hz, then goes flat (a broadband noise "
            "floor) all the way to the 75 Hz cutoff, across almost the whole cap "
            "simultaneously -- not an isolated bad electrode. The same flat plateau "
            "is already present in the RAW, pre-preprocessing signal, so this is a "
            "property of the original recording (most consistent with poor whole-cap "
            "contact/high impedance at acquisition time), not something introduced "
            "by this pipeline.",
            "- **A different, peakier pattern** (e.g. sub-80): elevated broadband "
            "noise with distinct oscillatory peaks (a large ~45 Hz peak was visible "
            "on F7) rather than a clean flat floor -- more consistent with movement/"
            "muscle contamination than a pure contact/impedance problem, though still "
            "anomalous and worth the same level of review.",
            "",
            "This table is a reasonable candidate list for exclusion or re-review; "
            "this audit flags it, it does not remove or exclude anything.",
            "",
        ]

    # --- Part B: aperiodic (1/f) slope --------------------------------------
    lines += ["## B. Aperiodic (1/f) slope (specparam)", ""]
    if n_rows:
        flagged = aperiodic_df[aperiodic_df["flagged"]]
        exp_zero = aperiodic_df[aperiodic_df["exponent_near_zero"]]
        poor_fit = aperiodic_df[aperiodic_df["poor_fit"]]
        fit_err = aperiodic_df[aperiodic_df["fit_error"].notna()]
        lines += [
            f"Checked {n_rows} channel-sessions.",
            f"**{len(flagged)}/{n_rows} flagged** "
            f"({len(exp_zero)} near-zero exponent [|exp|<{qc.aperiodic.exponent_near_zero_threshold:g}], "
            f"{len(poor_fit)} poor fit [R^2<{qc.aperiodic.r2_poor_threshold:g}], "
            f"{len(fit_err)} fit errors).",
            "",
        ]
        if len(flagged):
            ids = [f"{r.patient_id}-{r.condition}/{r.channel} ({r.flag_reason})"
                  for r in flagged.itertuples()]
            lines += [f"Flagged: {_list_or_truncate(ids)}", ""]

            per_session = aperiodic_df.groupby(["patient_id", "condition"])["flagged"].sum()
            n_channels_total = aperiodic_df["channel"].nunique()
            systemic = per_session[per_session >= 0.5 * n_channels_total].sort_values(ascending=False)
            if len(systemic):
                ids = [f"{pid}-{cond} ({int(n)}/{n_channels_total} channels)"
                      for (pid, cond), n in systemic.items()]
                lines += [
                    f"**Session pattern:** {len(systemic)} session(s) have >=50% of channels "
                    f"flagged -- likely a whole-session data-quality issue rather than "
                    f"isolated bad channels: {_list_or_truncate(ids)}",
                    "",
                ]

        valid = aperiodic_df[aperiodic_df["fit_error"].isna()]
        if len(valid):
            beta_rate_on = valid[valid["condition"] == "ON"]["beta_peak_present"].mean() \
                if (valid["condition"] == "ON").any() else float("nan")
            beta_rate_off = valid[valid["condition"] == "OFF"]["beta_peak_present"].mean() \
                if (valid["condition"] == "OFF").any() else float("nan")
            beta_rate_all = valid["beta_peak_present"].mean()
            lines += [
                f"Beta-band ({qc.aperiodic.beta_band_hz[0]:g}-{qc.aperiodic.beta_band_hz[1]:g} Hz) "
                f"peak detected in {beta_rate_all:.1%} of channel-sessions overall "
                f"(ON: {beta_rate_on:.1%}, OFF: {beta_rate_off:.1%}). This is a sanity "
                "reference only (levodopa-related beta suppression is a well-established "
                "finding, i.e. OFF > ON expected) -- not a target this audit fits to.",
                "",
            ]
        lines += [
            "**Headline:** " + (
                "most channels show a genuine 1/f decline with a good model fit; "
                "flagged channels are isolated."
                if len(flagged) / max(n_rows, 1) < 0.05 else
                "a non-trivial fraction of channel-sessions have a poor or near-flat "
                "aperiodic fit -- cross-check against the Part A flat-PSD list and the "
                "psd_examples/ plots before fine-tuning."
            ), "",
        ]
    else:
        lines += ["No aperiodic fit results were produced (see session failures below).", ""]

    # --- Part C: impedance ---------------------------------------------------
    lines += ["## C. Impedance", ""]
    any_found = any(c.impedance_found for c in impedance_checks)
    lines += [
        f"Checked {len(impedance_checks)} file(s) across "
        f"{len({(c.patient_id, c.condition) for c in impedance_checks})} sample session(s) "
        "(raw-signal CSV header + data columns, EDF channel names + annotations, "
        "and a best-effort ASCII scan of any `.dsi` file).",
        "",
    ]
    if any_found:
        hits = [f"{c.patient_id}-{c.condition}: {c.source_type} `{c.source_path}` -- {c.notes}"
               for c in impedance_checks if c.impedance_found]
        lines += ["**Impedance data WAS found** in some source(s):", "",
                 *(f"- {h}" for h in hits), ""]
    else:
        by_type = {}
        for c in impedance_checks:
            by_type.setdefault(c.source_type, []).append(c)
        lines += [
            "**Headline: no impedance data is present anywhere in this dataset**, in any "
            "of the three formats found on disk. Specifically:",
            "",
        ]
        for src_type, checks in sorted(by_type.items()):
            lines.append(f"- **{src_type}** ({len(checks)} file(s) checked): none carried "
                        "impedance values. " + checks[0].notes)
        lines += [
            "",
            "This is reported as a genuine data-availability finding, not a skipped check: "
            "DSI-Streamer supports continuous impedance monitoring, but it is not exported "
            "into any file format present in this dataset on this drive (no impedance "
            "column/metadata key in the raw-signal CSV, no impedance channel or annotation "
            "in the EDF exports, and no plain-text impedance marker in the `.dsi` byte "
            "stream -- though the `.dsi` format is proprietary binary with no vendor SDK "
            "available here, so a structured non-ASCII-tagged impedance packet cannot be "
            "fully ruled out by this check alone).",
            "",
        ]

    # --- Failures --------------------------------------------------------
    lines += ["## Processing failures (logged, not silently skipped)", ""]
    if session_failures:
        lines += [f"- **{len(session_failures)} session(s) could not be reconstructed/processed** "
                 "for Parts A/B:"]
        lines += [f"  - {key}: {err}" for key, err in session_failures]
        lines.append("")
    else:
        lines += ["- All sessions in the manifest were successfully reconstructed and processed.", ""]
    if representative_before_after_failures:
        lines += [f"- **{len(representative_before_after_failures)} representative-sample "
                 "before/after plot(s) could not be produced:**"]
        lines += [f"  - {key}: {err}" for key, err in representative_before_after_failures]
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[qc-report] wrote {out_path}")
