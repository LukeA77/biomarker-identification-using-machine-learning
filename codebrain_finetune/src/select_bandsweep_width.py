"""Pick ONE band-stop width per direction, and show the reasoning.

Supervisor feedback 2026-08-19 on the multi-width sweep figures (see
the multi-width band-sweep finding (EXPERIMENTS.md 16c/16d)):
  1. The marginal-profile figure averaged dBA across every width from 5-18 Hz.
     That is misleading — a wider notch removes more signal almost by
     construction, so the average partly just says "wide bands hurt more", not
     where in the spectrum the model is vulnerable. We must pick ONE width.
  2. Showing the drop (delta) instead of the actual accuracy hides that the two
     directions start from different baselines (toast->tea ~0.83, tea->toast
     ~0.71), so a bigger delta in one direction is not obviously a bigger effect.

This script handles point 1: it picks the narrowest width whose effect is (a)
interior — not pinned to the sweep's frequency boundary, which the existing
edge-truncation check flags as an artefact of where the search stopped, not a
real position — and (b) bigger than the noise you would see just from retraining
with a different random seed (the baseline's seed-to-seed spread, already
persisted per direction). That second bar is the plain-language version of
"a clear effect": if the drop is smaller than what different seeds already
disagree on, it is not a width we can defend, however non-zero the drop is.

Run once per direction; writes width_selection.json into the same results dir,
and a figure showing effect size against width, per width, with the noise floor
and the interior/edge-truncated split drawn so the choice is visible, not just
asserted.

Usage:
    python codebrain_finetune/src/select_bandsweep_width.py --dir results/xai/bandsweep_multiwidth --label toast_to_tea
    python codebrain_finetune/src/select_bandsweep_width.py --dir results/xai/bandsweep_multiwidth_tea_to_toast --label tea_to_toast
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


def analyse(d: Path):
    z = np.load(d / "bandsweep.npz", allow_pickle=False)
    meta = json.loads((d / "bandsweep_meta.json").read_text(encoding="utf-8"))
    lo, hi, width = z["lo"], z["hi"], z["width"]
    base_pp = z["baseline_per_patient_ba"]
    pp = z["per_patient_ba"]
    base_ba = base_pp.mean(axis=1)                  # (n_seed,)
    band_ba = pp.mean(axis=2)                        # (n_band, n_seed)
    dba = base_ba[None, :] - band_ba
    dmean = dba.mean(axis=1)
    noise_floor = float(base_ba.max() - base_ba.min())

    fmin, fmax = meta["fmin"], meta["fmax"]
    rows = []
    for w in np.unique(width):
        m = np.where(width == w)[0]
        b = m[np.argmax(dmean[m])]
        interior = not (lo[b] <= fmin + 1e-9 or hi[b] >= fmax - 1e-9)
        rows.append({
            "width": float(w), "lo": float(lo[b]), "hi": float(hi[b]),
            "dba": float(dmean[b]), "interior": bool(interior),
            "clear": bool(interior and dmean[b] > noise_floor),
        })

    candidates = [r for r in rows if r["clear"]]
    if candidates:
        chosen = min(candidates, key=lambda r: r["width"])
        fallback = False
    else:
        # No width both clears the noise floor and is interior: fall back to the
        # widest interior width and say so, rather than silently picking an
        # edge-truncated (uninterpretable) band.
        interior_rows = [r for r in rows if r["interior"]]
        chosen = max(interior_rows, key=lambda r: r["width"]) if interior_rows else rows[-1]
        fallback = True

    return {
        "direction": meta["direction"], "rows": rows, "noise_floor": noise_floor,
        "base_ba_mean": float(base_ba.mean()), "chosen": chosen, "fallback": fallback,
        "n_patients": meta["n_patients"], "n_windows": meta["n_windows"],
    }


def plot(result: dict, out_name: str):
    direction = result["direction"]
    rows = result["rows"]
    color = COLOR[direction]
    ws = [r["width"] for r in rows]
    dbas = [r["dba"] for r in rows]
    interior = [r["interior"] for r in rows]
    chosen_w = result["chosen"]["width"]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    nf = result["noise_floor"]
    ax.axhspan(0, nf, color=LGREY, alpha=0.35, zorder=0)
    ax.text(max(ws), nf, f"  noise floor: seed-to-seed spread of the baseline ({nf:.3f})",
            fontsize=7.8, color=GREY, ha="right", va="bottom")

    ax.plot(ws, dbas, color=color, lw=1.6, zorder=2)
    for r in rows:
        solid = r["interior"]
        ax.plot(r["width"], r["dba"], marker="o",
                markersize=8 if solid else 7,
                markerfacecolor=color if solid else "white",
                markeredgecolor=color, markeredgewidth=1.6, zorder=3)

    ch = result["chosen"]
    ax.scatter([ch["width"]], [ch["dba"]], s=220, facecolor="none",
               edgecolor=INK, linewidth=1.8, zorder=4)
    label = "chosen width" if not result["fallback"] else "chosen width (fallback: no width cleared the noise floor)"
    ax.annotate(f"{label}\n{ch['width']:.0f} Hz  ({ch['lo']:.0f}-{ch['hi']:.0f} Hz band)",
                xy=(ch["width"], ch["dba"]), xytext=(0.5, 0.92),
                textcoords="axes fraction", fontsize=9, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1.0,
                                connectionstyle="arc3,rad=0.15"))

    ax.set_xticks(ws)
    ax.set_xlabel("band-stop width (Hz)", fontsize=10, color=GREY)
    ax.set_ylabel("drop in balanced accuracy\n(best band within that width)", fontsize=10, color=GREY)
    ax.set_title(f"How the width was chosen — {NAME[direction]}", fontsize=12.5,
                 color=INK, loc="left", pad=10)
    _style(ax)

    solid_handle = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                               markeredgecolor=color, markersize=8, label="usable (interior) width")
    hollow_handle = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                                markeredgecolor=color, markersize=7,
                                label="not usable: best band pinned to sweep edge")
    ax.legend(handles=[solid_handle, hollow_handle], frameon=False, fontsize=8,
              labelcolor=INK, loc="upper left")

    fig.text(0.02, -0.02,
              f"Baseline balanced accuracy {result['base_ba_mean']:.3f}, "
              f"{result['n_patients']} patients, {result['n_windows']} windows, 5 seeds. "
              "A width counts as usable only if its strongest band sits strictly inside "
              "1-45 Hz (not against the sweep boundary), and clear only if that band's "
              "accuracy drop is bigger than the spread you already get across seeds.",
              fontsize=8, color=GREY, ha="left", va="top", wrap=True)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = FIGDIR / f"{out_name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", p)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pick one band-stop width per direction")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    result = analyse(d)
    direction = result["direction"]

    out_json = d / "width_selection.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[wrote {out_json}]")
    ch = result["chosen"]
    print(f"{direction}: chosen width = {ch['width']:.0f} Hz, band {ch['lo']:.0f}-{ch['hi']:.0f} Hz, "
          f"dBA {ch['dba']:+.4f} (noise floor {result['noise_floor']:.4f})"
          + ("  [FALLBACK: nothing cleared the noise floor]" if result["fallback"] else ""))

    out_name = args.out or f"fig_bandsweep_width_selection_{direction}"
    plot(result, out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
