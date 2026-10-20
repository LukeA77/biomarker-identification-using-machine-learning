"""Actual accuracy (not delta) across frequency, at ONE fixed band-stop width,
both directions on the same axes.

Supervisor feedback 2026-08-19, second point: showing the DROP hides that
toast->tea and tea->toast start from different baselines (~0.83 vs ~0.71), so a
bigger drop in one direction is not obviously a bigger effect. This script plots
the actual balanced accuracy curve for both directions instead, at a single
width picked by `select_bandsweep_width.py`, so the reader sees where each
direction starts, where it ends up, and can judge the comparison directly rather
than through a delta that buries the starting point.

Run twice — once per chosen width, so "does the width picked FROM one direction
also show up when applied TO the other" is a direct visual cross-check rather
than an assumption:

    python codebrain_finetune/src/plot_bandsweep_fixed_width.py --width 5  --picked-from toast_to_tea
    python codebrain_finetune/src/plot_bandsweep_fixed_width.py --width 12 --picked-from tea_to_toast
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_RUN_ROOT = Path(__file__).resolve().parent.parent
FIGDIR = _RUN_ROOT / "logs" / "figures"
DIRS = {
    "toast_to_tea": _RUN_ROOT / "results" / "xai" / "bandsweep_multiwidth",
    "tea_to_toast": _RUN_ROOT / "results" / "xai" / "bandsweep_multiwidth_tea_to_toast",
}

BLUE, VERM = "#0072B2", "#D55E00"
GREY, LGREY, INK = "#5A5A5A", "#B8B8B8", "#222222"
COLOR = {"toast_to_tea": VERM, "tea_to_toast": BLUE}
NAME = {"toast_to_tea": "toast→tea", "tea_to_toast": "tea→toast"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=8.5)
    ax.grid(True, color=LGREY, lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)


def load_at_width(d: Path, width: float):
    z = np.load(d / "bandsweep.npz", allow_pickle=False)
    lo, hi, w = z["lo"], z["hi"], z["width"]
    m = np.where(np.isclose(w, width))[0]
    if len(m) == 0:
        raise SystemExit(f"width {width} not found in {d} (available: {sorted(set(w))})")
    order = np.argsort(lo[m])
    m = m[order]
    centre = (lo[m] + hi[m]) / 2.0

    base_pp = z["baseline_per_patient_ba"]           # (n_seed, n_pat)
    pp = z["per_patient_ba"][m]                       # (n_band, n_seed, n_pat)
    base_ba_per_seed = base_pp.mean(axis=1)            # (n_seed,)
    ba_per_seed = pp.mean(axis=2)                       # (n_band, n_seed)
    ba_mean = ba_per_seed.mean(axis=1)
    ba_lo = ba_per_seed.min(axis=1)
    ba_hi = ba_per_seed.max(axis=1)
    return {
        "centre": centre, "lo": lo[m], "hi": hi[m],
        "ba_mean": ba_mean, "ba_lo": ba_lo, "ba_hi": ba_hi,
        "baseline_mean": float(base_ba_per_seed.mean()),
        "baseline_lo": float(base_ba_per_seed.min()),
        "baseline_hi": float(base_ba_per_seed.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Actual accuracy vs frequency at one fixed width, both directions")
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--picked-from", required=True, choices=list(DIRS))
    args = ap.parse_args()

    data = {name: load_at_width(d, args.width) for name, d in DIRS.items()}

    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    for name, dd in data.items():
        c = COLOR[name]
        ax.fill_between(dd["centre"], dd["ba_lo"], dd["ba_hi"], color=c, alpha=0.15, lw=0)
        ax.plot(dd["centre"], dd["ba_mean"], color=c, lw=2.0,
                label=f"{NAME[name]}  (baseline {dd['baseline_mean']:.3f})")
        ax.axhline(dd["baseline_mean"], color=c, lw=1.2, ls="--", alpha=0.7)

    ax.set_xlabel("centre frequency of the removed band (Hz)", fontsize=10, color=GREY)
    ax.set_ylabel("actual balanced accuracy\nwith that band removed", fontsize=10, color=GREY)
    ax.set_title(f"Accuracy at a fixed {args.width:.0f} Hz band-stop width, both directions",
                 fontsize=12.5, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    _style(ax)

    picked_name = NAME[args.picked_from]
    fig.text(0.02, -0.02,
              f"Width {args.width:.0f} Hz chosen from the {picked_name} sweep (narrowest interior "
              f"width whose effect there clears the seed-to-seed noise floor; see "
              f"fig_bandsweep_width_selection_{args.picked_from}). Applied unchanged to both "
              "directions here, so this is a direct check of whether that width still shows an "
              "effect on the OTHER direction, not just the one it was chosen from. Dashed lines "
              "are each direction's own baseline (no band removed) — the two directions do not "
              "start from the same accuracy, so read the drop against its own baseline, not "
              "against the other line.  Shaded band = range across the 5 seeds.",
              fontsize=8, color=GREY, ha="left", va="top", wrap=True)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = f"fig_bandsweep_fixedwidth_w{args.width:.0f}_from_{args.picked_from}"
    for ext in ("png", "pdf"):
        p = FIGDIR / f"{out}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
