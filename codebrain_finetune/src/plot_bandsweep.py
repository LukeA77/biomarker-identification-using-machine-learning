"""Figure for the multi-width band-stop sweep (Tier C-2).

Four panels, in the order the argument has to be made:

  A  the raw sweep — dBA over (band centre x band width). Sequential, one hue:
     this is magnitude, and magnitude never gets a rainbow.
  B  the same data as profiles, one line per width, so the reader can see whether
     the vulnerable region SHARPENS or MIGRATES as the notch widens. A peak that
     stays put as width grows is a real feature of the model; one that slides with
     the window is an edge effect.
  C  the confound, drawn: dBA against the fraction of 1-45 Hz power the notch
     removes, with the isotonic "what any notch removing this much costs" curve.
     Panels A and B are unreadable without this one, because EEG is 1/f and a low
     band is simply a bigger intervention than a high band of equal Hz-width.
  D  the residual from C, back on the (centre x width) grid. Diverging, because
     the sign is the whole point: red = the model depends on this position MORE
     than its energy explains. This is the only panel that carries a claim.

Colour: house Okabe-Ito (plot_xai.py / plot_nested_lora.py). Width is an ORDERED
variable, so panel B steps a single-hue ramp rather than cycling categorical hues
across 14 series. Panel D is the Okabe-Ito blue/vermillion diverging pair with a
neutral grey midpoint. Text stays in ink tokens; colour is carried by the marks.

Usage:
    python codebrain_finetune/src/plot_bandsweep.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

_RUN_ROOT = Path(__file__).resolve().parent.parent
DEFAULT = _RUN_ROOT / "results" / "xai" / "bandsweep_multiwidth"
FIGDIR = _RUN_ROOT / "logs" / "figures"

BLUE, VERM = "#0072B2", "#D55E00"
GREY, LGREY, INK = "#5A5A5A", "#B8B8B8", "#222222"

SEQ = LinearSegmentedColormap.from_list("seq_verm", ["#FFF6EF", "#F0B27F", VERM, "#6E2F00"])
DIV = LinearSegmentedColormap.from_list("div_bo", [BLUE, "#9CC7DE", "#F2F2F2", "#F0B27F", VERM])


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.grid(True, color=LGREY, lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)


def _mesh(ax, lo, hi, width, values, cmap, stride, norm=None, label=""):
    """Draw the ragged (centre x width) sweep one width-row at a time.

    A single rectangular grid over all widths does NOT work: band centres are
    integers for even widths and half-integers for odd ones, so a shared 0.5 Hz
    x-grid leaves every other cell empty and the panel renders as stripes that
    look like missing data. Each row instead gets its own edge array — bands at
    `stride` spacing, cells `stride` wide — which tiles exactly within the row and
    keeps the true ragged outline across rows.
    """
    centre = (lo + hi) / 2.0
    if norm is None:
        norm = plt.Normalize(np.nanmin(values), np.nanmax(values))
    im = None
    for w in np.unique(width):
        m = np.where(width == w)[0]
        o = m[np.argsort(centre[m])]
        ex = np.append(centre[o] - stride / 2, centre[o][-1] + stride / 2)
        im = ax.pcolormesh(ex, [w - 0.5, w + 0.5], values[o][None, :],
                           cmap=cmap, norm=norm, shading="flat", edgecolors="none")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=GREY)
    cb.outline.set_visible(False)
    cb.set_label(label, fontsize=8, color=GREY)
    ax.set_xlabel("centre frequency of the removed band (Hz)", fontsize=9, color=GREY)
    ax.set_ylabel("band width (Hz)", fontsize=9, color=GREY)
    _style(ax)
    ax.grid(False)
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot the multi-width band sweep")
    ap.add_argument("--dir", default=str(DEFAULT))
    ap.add_argument("--out", default="fig_xai_bandsweep")
    args = ap.parse_args()

    d = Path(args.dir)
    z = np.load(d / "bandsweep.npz", allow_pickle=False)
    meta = json.loads((d / "bandsweep_meta.json").read_text(encoding="utf-8"))
    lo, hi, width = z["lo"], z["hi"], z["width"]
    removed = z["power_removed_frac"]
    base_ba = z["baseline_per_patient_ba"].mean(axis=1)
    dba = (base_ba[None, :] - z["per_patient_ba"].mean(axis=2)).mean(axis=1)

    # Isotonic fit WITHIN EACH WIDTH — must match summarize_bandsweep.py. A single
    # pooled fit averages 5 Hz bands against 18 Hz bands at the same power value,
    # so its residual just restates "wider hurts more" (see that script's comment).
    from sklearn.isotonic import IsotonicRegression
    expected = np.full(len(lo), np.nan)
    for w in np.unique(width):
        m = width == w
        expected[m] = (IsotonicRegression(increasing=True, out_of_bounds="clip")
                       .fit(removed[m], dba[m]).predict(removed[m]))
    resid = dba - expected

    fig = plt.figure(figsize=(14.0, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30)
    stride = float(meta["stride"])

    # ---- A: raw sweep --------------------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    _mesh(axA, lo, hi, width, dba, SEQ, stride, label="ΔBA")
    axA.set_title("A  band-stop cost over position and width", fontsize=10.5,
                  color=INK, loc="left", pad=6)

    # ---- B: profiles by width ------------------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    uw = np.unique(width)
    for w in uw:
        m = width == w
        axB.plot((lo[m] + hi[m]) / 2.0, dba[m], lw=1.8,
                 color=SEQ(0.20 + 0.75 * (w - uw.min()) / max(np.ptp(uw), 1)))
    axB.axhline(0, color=GREY, lw=1.0)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(uw.min(), uw.max()))
    cb = fig.colorbar(sm, ax=axB, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=GREY)
    cb.outline.set_visible(False)
    cb.set_label("width (Hz)", fontsize=8, color=GREY)
    axB.set_xlabel("centre frequency of the removed band (Hz)", fontsize=9, color=GREY)
    axB.set_ylabel("drop in balanced accuracy", fontsize=9, color=GREY)
    axB.set_title("B  does the vulnerable region hold as the notch widens?",
                  fontsize=10.5, color=INK, loc="left", pad=6)
    _style(axB)

    # ---- C: the width/power confound, drawn ----------------------------------
    axC = fig.add_subplot(gs[1, 0])
    sc = axC.scatter(100 * removed, dba, c=width, cmap=SEQ, s=16,
                     edgecolors="white", linewidths=0.3, zorder=3)
    for i, w in enumerate(np.unique(width)):
        m = np.where(width == w)[0]
        o = m[np.argsort(removed[m])]
        axC.plot(100 * removed[o], expected[o], color=BLUE, lw=1.4, alpha=0.75,
                 zorder=4, label="isotonic fit, per width" if i == 0 else None)
    cb = fig.colorbar(sc, ax=axC, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=GREY)
    cb.outline.set_visible(False)
    cb.set_label("width (Hz)", fontsize=8, color=GREY)
    axC.set_xscale("symlog", linthresh=0.5)
    axC.set_xlabel("% of 1-45 Hz power removed by the notch (symlog)", fontsize=9, color=GREY)
    axC.set_ylabel("drop in balanced accuracy", fontsize=9, color=GREY)
    axC.set_title("C  the confound: wider and lower notches remove more",
                  fontsize=10.5, color=INK, loc="left", pad=6)
    axC.legend(frameon=False, fontsize=7.6, labelcolor=INK, loc="upper left")
    _style(axC)

    # ---- D: residual — the only panel that carries a claim --------------------
    axD = fig.add_subplot(gs[1, 1])
    lim = float(np.nanmax(np.abs(resid)))
    _mesh(axD, lo, hi, width, resid, DIV, stride,
          norm=TwoSlopeNorm(0.0, -lim, lim), label="ΔBA beyond power removal")
    axD.set_title("D  residual after controlling for power removed", fontsize=10.5,
                  color=INK, loc="left", pad=6)

    fig.suptitle("Multi-width band-stop sweep — pooled_tasksplit_chanz_matched, "
                 "full_ft, toast→tea", fontsize=13.5, color=INK, x=0.045, ha="left",
                 y=0.985)
    fig.text(0.045, 0.947,
             f"{meta['n_bands']} bands, widths {min(meta['widths']):.0f}–"
             f"{max(meta['widths']):.0f} Hz across {meta['fmin']:.0f}–{meta['fmax']:.0f} Hz, "
             f"{len(meta['seeds'])} seeds, {meta['n_windows']} windows / "
             f"{meta['n_patients']} patients.  Baseline BA "
             f"{base_ba.mean():.3f} (seed spread {np.ptp(base_ba):.3f}).\n"
             "Width is not comparable across rows of A — a wider notch removes more "
             "signal by construction; that is what panels C and D exist to remove.  "
             f"{meta['n_bands']} bands were tested, so the argmax is a maximum over "
             f"{meta['n_bands']} draws: read the profile, not the peak.\n"
             "Frequency axes are a post-hoc lens — the model never saw a band "
             "definition.  0.832 here vs 0.435 leak-free (recid 0.920): Tier E "
             "candidates, not biomarkers.",
             fontsize=8.4, color=GREY, ha="left", va="top")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = FIGDIR / f"{args.out}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
