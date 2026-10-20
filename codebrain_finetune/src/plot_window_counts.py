"""Figures for the window-count / duration-confound audit.

Reads what ``analyze_window_counts.py`` wrote and produces one four-panel figure:

  A  paired ON/OFF eval-window count per patient, faceted by task — the primary
     view: it shows both the spread across patients and, per patient, how far
     apart the two sessions are (the quantity a fixed-K rule has to erase).
  B  the marginal distribution of per-cell counts, by task and state.
  C  how many patients keep ALL FOUR of their cells as a global/per-task K rises.
  D  what fraction of the eval windows each fixed-K level retains.

Colours follow plot_nested_lora.py (Okabe-Ito) so the figures read as one set.
Verified pairwise separation for the ON/OFF pair: OKLab dE 31.2 normal vision,
worst-case dichromat 22.2 — comfortably above the dE>=8 CVD floor.

Usage:
    python codebrain_finetune/src/plot_window_counts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent
RES = _RUN_ROOT / "results" / "window_counts"
FIGDIR = _RUN_ROOT / "logs" / "figures"

ON_C = "#0072B2"     # condition A = ON
OFF_C = "#D55E00"    # condition C = OFF
GREY = "#5A5A5A"
LGREY = "#B8B8B8"
INK = "#222222"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.grid(True, color=LGREY, lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)


def panel_paired(ax, pairs: pd.DataFrame, task: str):
    sub = pairs[pairs.task_group == task].copy()
    sub = sub.sort_values("ON").reset_index(drop=True)
    y = np.arange(len(sub))
    for i, r in sub.iterrows():
        worse = OFF_C if r["OFF"] > r["ON"] else ON_C
        ax.plot([r["ON"], r["OFF"]], [i, i], color=worse, lw=1.6, alpha=0.55, zorder=2,
                solid_capstyle="round")
    ax.scatter(sub["ON"], y, s=30, facecolors="white", edgecolors=ON_C, linewidths=1.5,
               zorder=4)
    ax.scatter(sub["OFF"], y, s=30, facecolors=OFF_C, edgecolors="white", linewidths=0.9,
               zorder=4)
    kmin = int(sub[["ON", "OFF"]].min().min())
    ax.axvline(kmin, color=GREY, ls=(0, (4, 4)), lw=1.0, zorder=1)
    ax.text(kmin, len(sub) - 0.2, f" smallest cell = {kmin}", color=GREY, fontsize=7.5,
            va="top", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["patient_id"], fontsize=6.2, color="#333")
    ax.set_ylim(-0.8, len(sub) - 0.2)
    ax.set_xlabel("non-overlapping 10 s windows in this task", fontsize=9, color=GREY)
    ax.set_title(f"{task}  —  each patient's two sessions", fontsize=10.5, color=INK,
                 loc="left", pad=6)
    _style(ax)


def panel_marginal(ax, cells: pd.DataFrame):
    tasks = ["tea", "toast"]
    positions, labels, data, colors = [], [], [], []
    for i, t in enumerate(tasks):
        for j, (state, c) in enumerate((("ON", ON_C), ("OFF", OFF_C))):
            v = cells[(cells.task_group == t) & (cells.state == state)]["n_eval"].to_numpy()
            positions.append(i * 1.0 + (j - 0.5) * 0.34)
            labels.append(f"{t}\n{state}")
            data.append(v)
            colors.append(c)
    bp = ax.boxplot(data, positions=positions, widths=0.26, patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", lw=1.6),
                    whiskerprops=dict(color=GREY, lw=1.0), capprops=dict(color=GREY, lw=1.0))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("white")
        patch.set_linewidth(1.2)
        patch.set_alpha(0.85)
    rng = np.random.default_rng(0)
    for pos, v, c in zip(positions, data, colors):
        ax.scatter(pos + rng.uniform(-0.075, 0.075, size=v.size), v, s=11,
                   facecolors="white", edgecolors=c, linewidths=0.9, zorder=5, alpha=0.95)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8, color="#333")
    ax.set_ylabel("windows per session x task", fontsize=9, color=GREY)
    ax.set_title("spread of the per-cell counts", fontsize=10.5, color=INK, loc="left", pad=6)
    _style(ax)


def panel_sweep(ax, sweep: pd.DataFrame, n_pat: int):
    ax.step(sweep["k"], sweep["patients_surviving_tea"], where="post", color=ON_C, lw=2.0,
            label="tea only (2 cells)")
    ax.step(sweep["k"], sweep["patients_surviving_toast"], where="post", color=OFF_C, lw=2.0,
            label="toast only (2 cells)")
    ax.step(sweep["k"], sweep["patients_surviving_both_tasks"], where="post", color=INK,
            lw=2.0, ls=(0, (3, 2)), label="both tasks (all 4 cells)")
    ax.axhline(n_pat, color=LGREY, lw=1.0, ls=(0, (2, 3)))
    ax.text(6, n_pat, f"all {n_pat} ", color=GREY, fontsize=7.5, va="bottom", ha="left")
    ax.set_xlim(5, 80)
    ax.set_ylim(-1.5, n_pat + 3.0)
    ax.set_xlabel("fixed K windows per session x task", fontsize=9, color=GREY)
    ax.set_ylabel("patients retained", fontsize=9, color=GREY)
    ax.set_title("cost of a single shared K", fontsize=10.5, color=INK, loc="left", pad=6)
    ax.legend(frameon=False, fontsize=7.6, labelcolor=INK, loc="upper right",
              bbox_to_anchor=(1.0, 0.80))
    _style(ax)


def panel_retention(ax, mo: dict):
    order = [("per_patient_task", "per patient x task\nK = min(ON, OFF)"),
             ("per_patient", "per patient\none K for all 4 cells"),
             ("per_task", "per task\nK_tea, K_toast"),
             ("global", "global\none K everywhere")]
    fracs = [mo[k]["retained_frac"] * 100 for k, _ in order]
    kept = [mo[k]["kept"] for k, _ in order]
    y = np.arange(len(order))
    ax.barh(y, fracs, height=0.5, color=ON_C, edgecolor="white", linewidth=1.4, alpha=0.9)
    for i, (f, n) in enumerate(zip(fracs, kept)):
        ax.text(f + 1.5, i, f"{f:.0f}%  ({n:,} windows)", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in order], fontsize=7.6, color="#333")
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of the 4,980 eval windows kept", fontsize=9, color=GREY)
    ax.set_title("what each matching level costs", fontsize=10.5, color=INK, loc="left", pad=6)
    _style(ax)
    ax.grid(axis="y", visible=False)


def main() -> int:
    cells = pd.read_csv(RES / "window_counts_cells.csv")
    pairs = pd.read_csv(RES / "window_counts_pairs_eval.csv")
    sweep = pd.read_csv(RES / "window_counts_k_sweep.csv")
    summary = json.loads((RES / "window_count_distribution.json").read_text(encoding="utf-8"))
    n_pat = summary["n_patients"]

    fig = plt.figure(figsize=(13.0, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.28)

    panel_paired(fig.add_subplot(gs[0, 0]), pairs, "tea")
    panel_paired(fig.add_subplot(gs[0, 1]), pairs, "toast")
    panel_marginal(fig.add_subplot(gs[0, 2]), cells)
    panel_sweep(fig.add_subplot(gs[1, 0]), sweep, n_pat)
    panel_retention(fig.add_subplot(gs[1, 1]), summary["matching_options_eval"])

    # text panel: the headline numbers, so the figure stands alone
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ca = summary["count_alone_classifier_eval"]
    lines = [
        ("Is the count itself a label?", True),
        ("", False),
        (f"tea:   ON longer in {ca['tea']['n_on_longer']}/{ca['tea']['n_patients']} patients"
         f"  (sign p = {ca['tea']['sign_test_p']:.3f})", False),
        (f"toast: ON longer in {ca['toast']['n_on_longer']}/{ca['toast']['n_patients']} patients"
         f"  (sign p = {ca['toast']['sign_test_p']:.3f})", False),
        ("", False),
        (f"median |ON-OFF| gap: {summary['asymmetry_eval']['tea']['imbalance_frac']['median']*100:.0f}%"
         f" of the longer session (tea),", False),
        (f"{summary['asymmetry_eval']['toast']['imbalance_frac']['median']*100:.0f}% (toast)", False),
        ("", False),
        ("'shorter session = OFF' rule, within patient:", False),
        (f"   tea {1-ca['tea']['within_patient_longer_is_off_bal_acc']:.3f}   "
         f"toast {1-ca['toast']['within_patient_longer_is_off_bal_acc']:.3f}  bal-acc", False),
        ("absolute-count threshold, leave-one-patient-out:", False),
        (f"   tea {ca['tea']['loo_absolute_threshold_bal_acc']:.3f}   "
         f"toast {ca['toast']['loo_absolute_threshold_bal_acc']:.3f}  bal-acc", False),
    ]
    yy = 0.97
    for text, bold in lines:
        ax.text(0.0, yy, text, fontsize=9.2 if bold else 8.4, color=INK if bold else "#333",
                fontweight="bold" if bold else "normal", va="top", ha="left",
                family="DejaVu Sans", transform=ax.transAxes)
        yy -= 0.075 if text else 0.035

    fig.suptitle(
        "Window counts per session x task — 27 eligible patients, non-overlapping 10 s windows",
        fontsize=13, color=INK, x=0.045, ha="left", y=0.975)
    fig.text(0.045, 0.945,
             "Filled = OFF (condition C), hollow = ON (condition A). A fixed-K rule must "
             "equalise the two ends of every line.",
             fontsize=9, color=GREY, ha="left")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = FIGDIR / f"fig_window_counts.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
