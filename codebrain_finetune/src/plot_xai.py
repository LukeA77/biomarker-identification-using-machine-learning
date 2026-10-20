"""Figures for the XAI chapter — what did the model learn?

Reads what run_xai_perturbation.py wrote for all available seeds. Panels:

  A  seed stability (the XAI_PLAN 9.5 gate, flagged at 0.031)
  B  band-stop sweep at 2 Hz — the narrow-width reference

**WITHDRAWN 2026-08-15 — all channel-resolved panels.** The former panels B
(channel occlusion), C (importance topomap), E (Tier D cross-patient consistency)
and F (Tier D importance transfer) have been removed, together with the Tier D
section of `summarize_xai.py`. Reason: the electrode cap **moves substantially
during a recording session** in this ambulatory protocol, so an electrode label
does not index a stable scalp location either within a session or across patients.
Every one of those panels required exactly that assumption — a topomap places dBA
at a nominal 10-20 coordinate, and the Tier D transfer analysis compares channel
*identity* between patients. The underlying JSONs are left on disk under
`results/xai/xai_seed*/` and the scripts still run; they are simply no longer
reported. See EXPERIMENTS.md 15c/15d.

*Not withdrawn, but no longer plotted here:* the zero-vs-shuffle contrast is
aggregated over all 18 channels and is an argument about session-stability rather
than scalp location, so cap movement does not straightforwardly invalidate it. It
is parked pending a decision rather than deleted.

The multi-width sweep that supersedes panel B lives in `plot_bandsweep.py`.

Colours follow plot_nested_lora.py / plot_window_counts.py (Okabe-Ito).
Every frequency axis carries the post-hoc-lens caveat (XAI_PLAN 9.1) in the
caption, and every panel carries the seed-spread flag.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_RUN_ROOT = Path(__file__).resolve().parent.parent
RES = _RUN_ROOT / "results" / "xai"
FIGDIR = _RUN_ROOT / "logs" / "figures"

BLUE, VERM, PURP = "#0072B2", "#D55E00", "#CC79A7"
GREY, LGREY, INK = "#5A5A5A", "#B8B8B8", "#222222"
PUBLISHED = 0.832

CH_RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.grid(True, color=LGREY, lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)


def _seeds():
    out = {}
    for d in sorted(RES.glob("xai_seed*__toast_to_tea__full_ft__seed*")):
        s = int(d.name.split("seed")[-1])
        rec = {}
        p = d / "perturbation_summary.json"
        q = d / "discriminative_summary.json"
        if p.is_file():
            rec["pert"] = json.loads(p.read_text(encoding="utf-8"))
        if q.is_file():
            rec["disc"] = json.loads(q.read_text(encoding="utf-8"))
        if rec:
            out[s] = rec
    return dict(sorted(out.items()))


def _training_ba():
    """Seed BAs as REPORTED BY TRAINING, so panel A matches check_seed_stability's
    gate exactly. The perturbation baselines differ by ~0.001 (fp32 inference vs
    the bf16 autocast used at training time) and would print a different spread."""
    out = {}
    for s in range(1, 6):
        p = _RUN_ROOT / "results" / f"xai_seed{s}" / "toast_to_tea_full_ft.json"
        if p.is_file():
            out[s] = json.loads(p.read_text(encoding="utf-8"))["pooled_patient_level"]["mean"]
    return out


def panel_seed(ax, seeds):
    ba = _training_ba() or {s: r["pert"]["baseline_ba"] for s, r in seeds.items() if "pert" in r}
    xs, ys = list(ba), [ba[s] for s in ba]
    ax.axhline(PUBLISHED, color=GREY, ls=(0, (4, 4)), lw=1.1)
    ax.text(xs[0] - 0.4, PUBLISHED, f" published {PUBLISHED:.3f}", color=GREY,
            fontsize=7.5, va="bottom", ha="left")
    ax.scatter(xs, ys, s=54, facecolors=BLUE, edgecolors="white", linewidths=1.2, zorder=4)
    spread = max(ys) - min(ys)
    ax.fill_between([min(xs) - 0.4, max(xs) + 0.4], min(ys), max(ys),
                    color=BLUE, alpha=0.10, lw=0)
    ax.set_xticks(xs)
    ax.set_xlabel("seed", fontsize=9, color=GREY)
    ax.set_ylabel("patient-mean balanced accuracy", fontsize=9, color=GREY)
    ax.set_title(f"A  seed stability — spread {spread:.3f}", fontsize=10.5, color=INK,
                 loc="left", pad=6)
    _style(ax)




def panel_band(ax, seeds):
    recs = [r["pert"] for r in seeds.values() if "pert" in r]
    bands = list(recs[0]["bandstop"])
    V = np.array([[r["bandstop"][b] for b in bands] for r in recs])
    x = [np.mean([float(v) for v in b.split("-")]) for b in bands]
    ax.axhline(0, color=GREY, lw=1.0)
    ax.plot(x, V.mean(0), color=VERM, lw=2.0, marker="o", ms=4,
            markerfacecolor="white", markeredgecolor=VERM)
    ax.fill_between(x, V.mean(0) - V.std(0), V.mean(0) + V.std(0),
                    color=VERM, alpha=0.15, lw=0)
    pr = np.array([r["phase_randomised_dBA"] for r in recs])
    ax.axhline(pr.mean(), color=BLUE, ls=(0, (4, 3)), lw=1.6)
    ax.text(x[-1], pr.mean(), f"destroying ALL phase costs {pr.mean():+.3f} ", color=BLUE,
            fontsize=7.5, va="bottom", ha="right")
    # Name the one non-flat point rather than letting the title imply nothing moved.
    j = int(np.argmax(V.mean(0)))
    ax.annotate(f"{bands[j]} Hz\n{V.mean(0)[j]:+.3f}", xy=(x[j], V.mean(0)[j]),
                xytext=(x[j] + 6, V.mean(0)[j] + 0.004), fontsize=7.5, color=VERM,
                arrowprops=dict(arrowstyle="-", color=VERM, lw=0.9))
    ax.set_xlabel("centre frequency of the removed 2 Hz band (Hz)", fontsize=9, color=GREY)
    ax.set_ylabel("drop in balanced accuracy", fontsize=9, color=GREY)
    ax.set_title("B  2 Hz band-stop — flat, but see plot_bandsweep.py", fontsize=10.5,
                 color=INK, loc="left", pad=6)
    _style(ax)




def main() -> int:
    seeds = _seeds()
    if not seeds:
        print("No XAI results found under", RES)
        return 1
    print("seeds found:", list(seeds))

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.26)
    panel_seed(fig.add_subplot(gs[0, 0]), seeds)
    panel_band(fig.add_subplot(gs[0, 1]), seeds)

    fig.suptitle("What did the pooled full fine-tune actually learn?  "
                 "(chan_z, duration-matched, toast→tea, 27 patients)",
                 fontsize=13.5, color=INK, x=0.04, ha="left", y=1.03)
    fig.text(0.04, 0.945,
             "Model scores 0.832 here but 0.435 leak-free (recid 0.920) — these are "
             "candidate generators, not biomarkers.  5-seed spread 0.031.  "
             "Frequency axes are a post-hoc lens: the model never saw a band definition.\n"
             "Channel-resolved panels withdrawn 2026-08-15: the cap moves during a "
             "session, so an electrode label does not index a stable scalp location.",
             fontsize=8.6, color=GREY, ha="left", va="top")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = FIGDIR / f"fig_xai_summary.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
