"""Direct toast→tea vs tea→toast comparison of the multi-width band sweep.

Built to answer one question: are the two directions' vulnerability profiles the
same shape? They are not, and this figure is the evidence — both panels plot the
two directions on the SAME axes so the difference is a direct visual comparison,
not two separate figures the reader has to hold in their head at once.

  A  marginal frequency profile (width-averaged ΔBA per Hz) — same construction as
     the single-direction report's table, one line per direction.
  B  best-band-within-width ΔBA, one line per direction, widths 5-7 Hz only (the
     only widths whose optimum is interior for BOTH directions — see the
     edge-truncation warning in each direction's own report).

Usage:
    python codebrain_finetune/src/plot_bandsweep_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_RUN_ROOT = Path(__file__).resolve().parent.parent
FIGDIR = _RUN_ROOT / "logs" / "figures"
DIRS = {
    "toast→tea": _RUN_ROOT / "results" / "xai" / "bandsweep_multiwidth",
    "tea→toast": _RUN_ROOT / "results" / "xai" / "bandsweep_multiwidth_tea_to_toast",
}

BLUE, VERM = "#0072B2", "#D55E00"
GREY, LGREY, INK = "#5A5A5A", "#B8B8B8", "#222222"
COLOR = {"toast→tea": VERM, "tea→toast": BLUE}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.grid(True, color=LGREY, lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)


def load(d):
    z = np.load(d / "bandsweep.npz", allow_pickle=False)
    meta = json.loads((d / "bandsweep_meta.json").read_text(encoding="utf-8"))
    lo, hi, width = z["lo"], z["hi"], z["width"]
    base = z["baseline_per_patient_ba"].mean(axis=1)
    dba = (base[None, :] - z["per_patient_ba"].mean(axis=2)).mean(axis=1)
    return lo, hi, width, dba, float(base.mean()), meta


def marginal_profile(lo, hi, width, dba, fmin, fmax):
    bins = np.arange(fmin, fmax)
    prof = np.full(len(bins), np.nan)
    for i, f in enumerate(bins):
        cov = (lo <= f) & (hi > f)
        if cov.any():
            prof[i] = np.nanmean([dba[cov & (width == w)].mean()
                                  for w in np.unique(width[cov])])
    return bins, prof


def main() -> int:
    data = {name: load(d) for name, d in DIRS.items()}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # ---- A: marginal frequency profile, both directions on one axis ----------
    for name, (lo, hi, width, dba, base, meta) in data.items():
        bins, prof = marginal_profile(lo, hi, width, dba, meta["fmin"], meta["fmax"])
        axA.plot(bins + 0.5, prof, color=COLOR[name], lw=2.2, label=f"{name} (base BA {base:.3f})")
    axA.axhline(0, color=GREY, lw=1.0)
    axA.set_xlabel("Hz", fontsize=9, color=GREY)
    axA.set_ylabel("mean ΔBA of bands covering this Hz\n(width-averaged)", fontsize=9, color=GREY)
    axA.set_title("A  marginal vulnerability profile", fontsize=10.5, color=INK, loc="left", pad=6)
    axA.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    _style(axA)

    # ---- B: best-band-within-width, widths 5-7 Hz only (both interior) -------
    for name, (lo, hi, width, dba, base, meta) in data.items():
        xs, ys = [], []
        for w in (5, 6, 7):
            m = np.where(width == w)[0]
            b = m[np.argmax(dba[m])]
            xs.append(w)
            ys.append(dba[b])
        axB.plot(xs, ys, color=COLOR[name], lw=2.2, marker="o", ms=7,
                 markerfacecolor="white", markeredgecolor=COLOR[name],
                 markeredgewidth=1.8, label=name)
    axB.set_xticks([5, 6, 7])
    axB.set_xlabel("band width (Hz)", fontsize=9, color=GREY)
    axB.set_ylabel("best-in-width ΔBA (interior optima only)", fontsize=9, color=GREY)
    axB.set_title("B  strongest interior effect, widths 5-7 Hz", fontsize=10.5,
                  color=INK, loc="left", pad=6)
    axB.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    _style(axB)

    fig.suptitle("toast→tea vs tea→toast — multi-width band-stop sweep",
                 fontsize=13.5, color=INK, x=0.045, ha="left", y=1.03)
    fig.text(0.045, 0.945,
             "Same 469-band grid (widths 5-18 Hz, positions 1-45 Hz), same 5 seeds "
             "per direction, pooled_tasksplit_chanz_matched full_ft.  Widths ≥ 8 Hz "
             "excluded from panel B: their optimum is pinned to the 1 Hz sweep edge "
             "in at least one direction (edge-truncation warning, see each "
             "direction's own report) and is not interpretable.",
             fontsize=8.4, color=GREY, ha="left", va="top")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = FIGDIR / f"fig_xai_bandsweep_compare.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
