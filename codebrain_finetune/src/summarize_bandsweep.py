"""Turn the 469-band multi-width sweep into a defensible statement (Tier C-2).

`run_xai_bandsweep.py` produces raw dBA for every (position, width, seed). Raw dBA
on its own supports almost no claim, for three reasons this script exists to
handle:

1. **Width is confounded with everything.** A wider notch removes more signal, so
   dBA must rise with width whatever the model learned. Every comparison here is
   therefore made WITHIN a width, never across widths.
2. **Power removal is confounded with position.** A low-frequency band removes
   vastly more power than a high-frequency band of identical Hz-width (EEG is 1/f),
   so "wider hurts more" and "lower hurts more" are the same trivial effect wearing
   two hats. The fix is an isotonic fit of dBA on `power_removed_frac` — the
   monotone "what any notch removing this much power would cost" curve — and to
   read the RESIDUAL. A band that hurts more than its power removal predicts is
   the only kind that says something about the model.
3. **469 bands is a lot of draws.** The argmax of 469 correlated tests is not a
   discovery. Reported alongside every peak: how many of the 5 seeds agree in
   sign, the within-width z, and a patient-paired CI that uses the persisted
   per-patient BAs.

Outputs a markdown report; `--json` also writes the machine-readable table.

Usage:
    python codebrain_finetune/src/summarize_bandsweep.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_RUN_ROOT = Path(__file__).resolve().parent.parent
DEFAULT = _RUN_ROOT / "results" / "xai" / "bandsweep_multiwidth"

# The 2 Hz sweep this supersedes, for the continuity row in the report.
TWOHZ = (_RUN_ROOT / "results" / "xai" /
         "xai_seed1__toast_to_tea__full_ft__seed1" / "perturbation_summary.json")


def load(d: Path):
    z = np.load(d / "bandsweep.npz", allow_pickle=False)
    meta = json.loads((d / "bandsweep_meta.json").read_text(encoding="utf-8"))
    return z, meta


def paired_ci(delta_pp: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """Bootstrap CI over PATIENTS of a per-patient BA drop, seed-averaged.

    Patients are the unit of resampling because the headline metric is a
    patient-mean; resampling windows would understate the CI by ignoring the
    within-patient correlation that the patient-mean already collapses.
    """
    rng = np.random.default_rng(seed)
    n = len(delta_pp)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = delta_pp[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarise the multi-width band sweep")
    ap.add_argument("--dir", default=str(DEFAULT))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    d = Path(args.dir)
    z, meta = load(d)
    lo, hi, width = z["lo"], z["hi"], z["width"]
    pp = z["per_patient_ba"]                     # (n_band, n_seed, n_pat)
    base_pp = z["baseline_per_patient_ba"]       # (n_seed, n_pat)
    seeds, patients = z["seeds"], z["patients"]
    removed = z["power_removed_frac"]

    base_ba = base_pp.mean(axis=1)               # (n_seed,)
    band_ba = pp.mean(axis=2)                    # (n_band, n_seed)
    dba = base_ba[None, :] - band_ba             # (n_band, n_seed)
    dmean, dsd = dba.mean(axis=1), dba.std(axis=1)
    agree = (np.sign(dba) == np.sign(dmean)[:, None]).sum(axis=1)

    # per-patient drop, seed-averaged: (n_band, n_pat)
    dpp = (base_pp[None, :, :] - pp).mean(axis=1)

    # ---- control 1: within-width standardisation -----------------------------
    zw = np.full(len(lo), np.nan)
    for w in np.unique(width):
        m = width == w
        s = dmean[m].std()
        zw[m] = (dmean[m] - dmean[m].mean()) / (s if s > 1e-12 else np.inf)

    # ---- control 2: isotonic fit of dBA on power removed, WITHIN EACH WIDTH ---
    # Fitting one isotonic curve over all 469 bands is WRONG and was the first
    # thing tried: at any given low power-removal value the pooled fit averages
    # 5 Hz bands against 18 Hz bands, so `expected` becomes a width-blind mean and
    # the residual silently re-encodes "wider hurts more" — the exact confound
    # this control exists to remove. The tell was that the top-10 residuals came
    # back all width 18. Fitting per width makes the residual width-free by
    # construction: it can only ever compare a band against other bands of its own
    # width, which is the same discipline as the within-width z above.
    from sklearn.isotonic import IsotonicRegression
    expected = np.full(len(lo), np.nan)
    for w in np.unique(width):
        m = width == w
        expected[m] = (IsotonicRegression(increasing=True, out_of_bounds="clip")
                       .fit(removed[m], dmean[m]).predict(removed[m]))
    resid = dmean - expected

    L = []
    A = L.append
    A("# Multi-width band-stop sweep — pooled_tasksplit_chanz_matched")
    A("")
    A(f"Experiment: **{meta['experiment']}**, direction `{meta['direction']}`, arm "
      f"`full_ft`. {meta['n_bands']} bands, widths "
      f"{min(meta['widths']):.0f}-{max(meta['widths']):.0f} Hz (step "
      f"{meta['widths'][1] - meta['widths'][0]:.0f}), positions "
      f"{meta['fmin']:.0f}-{meta['fmax']:.0f} Hz at {meta['stride']:.0f} Hz stride, "
      f"scored on {meta['n_windows']} windows / {meta['n_patients']} patients across "
      f"{len(seeds)} seeds.")
    A("")
    A(f"Baseline patient-mean BA per seed: "
      + ", ".join(f"{s}={b:.4f}" for s, b in zip(seeds, base_ba))
      + f" (mean {base_ba.mean():.4f}).")
    A("")

    # ---- continuity with the 2 Hz sweep --------------------------------------
    if TWOHZ.is_file():
        two = json.loads(TWOHZ.read_text(encoding="utf-8")).get("bandstop", {})
        if two:
            v = np.array(list(two.values()))
            A("## Continuity with the 2 Hz sweep")
            A("")
            A(f"The superseded 2 Hz sweep (seed 1, same filters, same frame) had "
              f"mean |dBA| {np.abs(v).mean():.4f} and a maximum of {v.max():.4f}. "
              f"This sweep's widths reach the values below, which is the point: the "
              f"2 Hz null was a statement about 2 Hz, not about the model.")
            A("")

    # ---- headline: best band per width ---------------------------------------
    A("## Best band at each width (within-width comparison — the only valid one)")
    A("")
    A("| width | best band | dBA | seed sd | seeds agreeing | power removed | within-width z |")
    A("|---|---|---|---|---|---|---|")
    for w in np.unique(width):
        m = np.where(width == w)[0]
        b = m[np.argmax(dmean[m])]
        A(f"| {w:.0f} Hz | {lo[b]:.0f}-{hi[b]:.0f} Hz | **{dmean[b]:+.4f}** | "
          f"{dsd[b]:.4f} | {agree[b]}/{len(seeds)} | {100 * removed[b]:.1f}% | "
          f"{zw[b]:+.2f} |")
    A("")
    A("Width is NOT comparable down this column — a wider notch removes more signal "
      "by construction. The column shows where within each width the model is most "
      "vulnerable, and whether that location is stable as the width grows.")
    A("")
    # Edge truncation: if the best band sits against the sweep boundary, the true
    # optimum is outside the search range and the reported edges are an artefact of
    # where we stopped looking, not of the model.
    trunc = [w for w in np.unique(width)
             if (lambda b: lo[b] <= meta["fmin"] + 1e-9 or hi[b] >= meta["fmax"] - 1e-9)(
                 np.where(width == w)[0][np.argmax(dmean[width == w])])]
    if trunc:
        A(f"> **Edge-truncation warning.** At widths "
          f"{', '.join(f'{w:.0f}' for w in trunc)} Hz the best band is pinned to the "
          f"sweep boundary ({meta['fmin']:.0f} or {meta['fmax']:.0f} Hz). The optimum "
          f"wants to extend outside the searched range, so those band edges are an "
          f"artefact of where the search stopped and must not be quoted as a result. "
          f"Only widths whose optimum sits strictly inside the range are interpretable.")
        A("")

    # ---- power-removal residual: the only claim that survives the controls ----
    A("## Controlling for how much power the notch removes")
    A("")
    A("EEG is 1/f, so a low band and a high band of equal Hz-width are not equal "
      "interventions. `expected` is an isotonic (monotone) fit of dBA on the "
      "fraction of 1-45 Hz power removed, fitted **separately within each width**: "
      "what a notch OF THAT WIDTH destroying that much power costs, regardless of "
      "where it sits. The residual is what the position itself buys. Fitting one "
      "curve across all widths instead makes the residual a restatement of the "
      "width confound — see the code comment.")
    A("")
    order = [np.where(width == w)[0][np.argmax(resid[width == w])]
             for w in np.unique(width)]
    A("The CI column is the patient-paired 95% CI of the RAW dBA, not of the "
      "residual — it says the drop is real, not that the residual is.")
    A("")
    A("| band | width | dBA | expected(power) | residual | power removed | seeds | patient-paired 95% CI on dBA |")
    A("|---|---|---|---|---|---|---|---|")
    for b in order:
        cl, cu = paired_ci(dpp[b])
        A(f"| {lo[b]:.0f}-{hi[b]:.0f} Hz | {width[b]:.0f} | {dmean[b]:+.4f} | "
          f"{expected[b]:+.4f} | **{resid[b]:+.4f}** | {100 * removed[b]:.1f}% | "
          f"{agree[b]}/{len(seeds)} | [{cl:+.4f}, {cu:+.4f}] |")
    A("")

    # ---- efficiency: dBA per unit power removed ------------------------------
    A("## Cost per unit of power removed")
    A("")
    A("dBA divided by the fraction of power removed. Large values mark bands the "
      "model depends on out of all proportion to their energy — the signature of a "
      "feature that is informative but low-power, which in this cohort is more "
      "consistent with a recording/contact fingerprint than with a rhythm.")
    A("")
    A("Reported as the best band WITHIN each width, because the ratio also rises "
      "with width and a pooled top-10 would simply list the widest bands.")
    A("")
    eff = dmean / np.clip(removed, 1e-4, None)
    A("| width | most efficient band | dBA | power removed | dBA per unit power |")
    A("|---|---|---|---|---|")
    for w in np.unique(width):
        m = np.where(width == w)[0]
        b = m[np.argmax(eff[m])]
        A(f"| {w:.0f} Hz | {lo[b]:.0f}-{hi[b]:.0f} Hz | {dmean[b]:+.4f} | "
          f"{100 * removed[b]:.2f}% | {eff[b]:.2f} |")
    A("")

    # ---- where the vulnerability sits, marginalised over width ---------------
    A("## Frequency profile, marginalised over width")
    A("")
    A("Mean dBA of every band COVERING each 1 Hz bin, averaged within width first "
      "so wide bands do not dominate. This is the profile to report; it has no "
      "band edges in it and so cannot be band-fished.")
    A("")
    bins = np.arange(meta["fmin"], meta["fmax"])
    prof = []
    for f in bins:
        cov = (lo <= f) & (hi > f)
        if not cov.any():
            prof.append(np.nan)
            continue
        # average within width, then across widths: equal weight per width
        prof.append(np.nanmean([dmean[cov & (width == w)].mean()
                                for w in np.unique(width[cov])]))
    prof = np.array(prof)
    A("| Hz | mean dBA of covering bands |")
    A("|---|---|")
    for f, v in zip(bins, prof):
        bar = "#" * int(round(max(v, 0) / max(np.nanmax(prof), 1e-9) * 40))
        A(f"| {f:.0f}-{f + 1:.0f} | {v:+.4f} {bar} |")
    A("")

    A("## Caveats carried by every number above")
    A("")
    for k in ("frequency_caveat", "width_caveat", "fishing_caveat", "scope_caveat"):
        A(f"* {meta[k]}")
    A("")
    seed_spread = base_ba.max() - base_ba.min()
    A(f"* Seed spread of the baseline is {seed_spread:.3f} "
      f"({base_ba.min():.3f}-{base_ba.max():.3f}); any dBA of comparable size is "
      f"not separable from which seed was trained.")

    report = "\n".join(L)
    out = d / "bandsweep_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[wrote {out}]")

    if args.json:
        rows = [{"lo": float(lo[i]), "hi": float(hi[i]), "width": float(width[i]),
                 "dba_mean": float(dmean[i]), "dba_sd": float(dsd[i]),
                 "seeds_agreeing": int(agree[i]),
                 "dba_per_seed": [float(v) for v in dba[i]],
                 "power_removed_frac": float(removed[i]),
                 "within_width_z": float(zw[i]),
                 "expected_from_power": float(expected[i]),
                 "residual": float(resid[i])} for i in range(len(lo))]
        (d / "bandsweep_table.json").write_text(
            json.dumps({"meta": meta, "bands": rows}, indent=2), encoding="utf-8")
        print(f"[wrote {d / 'bandsweep_table.json'}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
