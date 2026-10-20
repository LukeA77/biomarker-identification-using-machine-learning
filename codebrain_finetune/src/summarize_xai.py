"""Cross-seed summary of the XAI package — the numbers that go in the briefing.

Pulls together Tier C (perturbation) and Tier D (discriminative) across every
available seed and reports:

  * seed stability of the headline metric;
  * Tier C: band-stop profile and phase randomisation.

**WITHDRAWN 2026-08-15 — every channel-resolved statistic.** This script used to
report per-channel occlusion, the channel-ranking Spearman across seeds, and the
whole of Tier D (cross-patient agreement of channel-importance profiles, top-k
importance transfer, Jaccard set overlap). All of it assumed an electrode label
indexes a stable scalp location. It does not: the cap **moves substantially during
a recording session** in this ambulatory protocol, so neither a within-session
importance value nor a between-patient channel comparison means what those
analyses took it to mean. `results/xai/xai_seed*/discriminative_summary.json` is
left on disk and `run_xai_discriminative.py` still runs; the numbers are simply no
longer reported. See EXPERIMENTS.md 15c/15d and `plot_xai.py`.

The 2 Hz band-stop profile summarised here is itself superseded by the multi-width
sweep — `summarize_bandsweep.py`. A 2 Hz notch removes ~4% of 1-45 Hz power, so its
null result was a statement about 2 Hz, not about the model.

Usage:
    python codebrain_finetune/src/summarize_xai.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_RUN_ROOT = Path(__file__).resolve().parent.parent
RES = _RUN_ROOT / "results" / "xai"


def _load():
    out = {}
    for d in sorted(RES.glob("xai_seed*__toast_to_tea__full_ft__seed*")):
        s = int(d.name.split("seed")[-1])
        rec = {}
        for key, fn in (("pert", "perturbation_summary.json"),
                        ("disc", "discriminative_summary.json")):
            p = d / fn
            if p.is_file():
                rec[key] = json.loads(p.read_text(encoding="utf-8"))
        if rec:
            out[s] = rec
    return dict(sorted(out.items()))


def main() -> int:
    seeds = _load()
    if not seeds:
        print("no results under", RES)
        return 1
    pert = {s: r["pert"] for s, r in seeds.items() if "pert" in r}
    disc = {s: r["disc"] for s, r in seeds.items() if "disc" in r}
    print(f"seeds with Tier C: {list(pert)} | with Tier D: {list(disc)}\n")

    # ---- Tier C ------------------------------------------------------------
    if pert:
        base = np.array([r["baseline_ba"] for r in pert.values()])
        PR = np.array([r["phase_randomised_dBA"] for r in pert.values()])
        print("TIER C  (channel-resolved rows withdrawn — see module docstring)")
        print(f"  baseline BA        {base.mean():.4f}  (min {base.min():.4f}, max {base.max():.4f})")
        print(f"  phase randomised   dBA {PR.mean():.4f} +/- {PR.std():.4f}")

        bands = list(pert[next(iter(pert))]["bandstop"])
        V = np.array([[r["bandstop"][b] for b in bands] for r in pert.values()])
        top = np.argsort(-V.mean(0))[:3]
        print("  band-stop: top 3 " + ", ".join(
            f"{bands[j]} Hz {V.mean(0)[j]:+.4f}" for j in top))
        print(f"  band-stop: mean |dBA| over all bands {np.abs(V.mean(0)).mean():.4f}")

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
