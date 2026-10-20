"""Dissertation figures for the nested-LOPO LoRA result.

Two publication-quality panels, saved as PNG (200 dpi) + PDF (vector):

  fig_roc          — pooled session-level ROC over the 70 held-out sessions, with a
                     patient-cluster bootstrap band (same resampling as the CI), the
                     frozen linear-probe ROC overlaid for the head-to-head, and the
                     chance diagonal. AUC + bootstrap CI annotated.
  fig_ordering     — per-patient within-subject ordering: each patient's ON vs OFF
                     mean P(OFF) as a dumbbell, sorted by the OFF-minus-ON shift.
                     Line colour encodes correct (OFF>ON) vs inverted ordering;
                     marker SHAPE encodes ON vs OFF so identity never rests on colour
                     alone. The 0.5 decision threshold is drawn to expose the
                     cross-subject calibration problem (both sessions often the same
                     side of 0.5).

Colours: Okabe-Ito CVD-safe palette (blue = LoRA / correct ordering; vermillion =
frozen probe / inverted ordering). Design per the dataviz skill: form-first, thin
recessive axes, direct labels, secondary (shape) encoding alongside colour.

Run:  python codebrain_finetune/src/plot_nested_lora.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_RUN_ROOT = _HERE.parent.parent                 # codebrain_finetune/
RESULTS = _RUN_ROOT / "results" / "nested_lora"
FULLFT = _RUN_ROOT / "results" / "nested_full_ft"
PROBE_JSON = _RUN_ROOT / "results" / "stage4_probe_lopo_mean_20260722T122148Z.json"
OUTDIR = _RUN_ROOT / "logs" / "figures"

# --- Okabe-Ito (CVD-safe) ---
BLUE = "#0072B2"      # LoRA / correct ordering
VERM = "#D55E00"      # frozen probe / inverted ordering
PURP = "#CC79A7"      # full fine-tuning
GREY = "#5A5A5A"
LGREY = "#B8B8B8"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score  # noqa: E402


def _load_fold_sessions(results_dir):
    """Pool full-precision session scores across a run's 35 fold JSONs."""
    prob, true, pat, cond = [], [], [], []
    for f in sorted(glob.glob(str(results_dir / "fold_*.json"))):
        sc = json.load(open(f))["session_scores"]
        prob.extend(sc["prob"]); true.extend(sc["y_true"]); pat.extend(sc["patients"])
        cond.extend(s.split("|")[-1] for s in sc["sessions"])
    return (np.array(prob, float), np.array(true, int), np.array(pat), np.array(cond))


def _load_lora_sessions():
    return _load_fold_sessions(RESULTS)


def _load_probe_sessions():
    r = json.load(open(PROBE_JSON))
    prob, true, pat = [], [], []
    for pf in r["per_fold"]:
        prob.extend(pf["prob"]); true.extend(pf["y_true"])
        pat.extend(s.split("|")[0] for s in pf["sessions"])
    return np.array(prob, float), np.array(true, int), np.array(pat)


def _bootstrap_roc_band(true, prob, pat, *, grid, n=2000, seed=1):
    """Patient-cluster bootstrap of the ROC curve -> (lo, hi) TPR band + AUC CI."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(pat)
    per = {p: np.where(pat == p)[0] for p in uniq}
    tprs, aucs = [], []
    for _ in range(n):
        chosen = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([per[uniq[c]] for c in chosen])
        yt, pr = true[idx], prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        fpr, tpr, _ = roc_curve(yt, pr)
        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(roc_auc_score(yt, pr))
    tprs = np.array(tprs)
    return (np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0),
            float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def _style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.grid(True, color=LGREY, lw=0.5, alpha=0.4)
    ax.set_axisbelow(True)


def fig_roc(lora, probe, fullft):
    prob, true, pat, _ = lora
    pprob, ptrue, _ = probe
    fprob, ftrue, _, _ = fullft
    grid = np.linspace(0, 1, 200)

    fpr_l, tpr_l, _ = roc_curve(true, prob)
    auc_l = roc_auc_score(true, prob)
    lo, hi, auc_lo, auc_hi = _bootstrap_roc_band(true, prob, pat, grid=grid)

    fpr_p, tpr_p, _ = roc_curve(ptrue, pprob)
    auc_p = roc_auc_score(ptrue, pprob)
    fpr_f, tpr_f, _ = roc_curve(ftrue, fprob)
    auc_f = roc_auc_score(ftrue, fprob)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color=LGREY, lw=1.2, zorder=1)
    ax.text(0.63, 0.56, "chance", color=GREY, fontsize=8.5, rotation=45,
            rotation_mode="anchor", va="bottom")

    ax.fill_between(grid, lo, hi, color=BLUE, alpha=0.15, lw=0, zorder=2,
                    label="LoRA 95% bootstrap band")
    ax.plot(fpr_l, tpr_l, color=BLUE, lw=2.2, zorder=5,
            label=f"LoRA  (AUC {auc_l:.3f}, 95% CI [{auc_lo:.3f}, {auc_hi:.3f}])")
    ax.plot(fpr_p, tpr_p, color=VERM, lw=1.6, ls=(0, (5, 2)), zorder=3,
            label=f"Frozen probe  (AUC {auc_p:.3f})")
    ax.plot(fpr_f, tpr_f, color=PURP, lw=1.6, ls=(0, (1, 1.6)), zorder=4,
            label=f"Full fine-tune  (AUC {auc_f:.3f})")

    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate  (ON mislabelled OFF)", fontsize=10, color=GREY)
    ax.set_ylabel("True positive rate  (OFF recall)", fontsize=10, color=GREY)
    ax.set_title("Pooled session-level ROC — 70 held-out sessions",
                 fontsize=11.5, color="#222", pad=10, loc="left")
    _style_axes(ax)
    ax.legend(loc="lower right", fontsize=8.3, frameon=False, borderpad=0.2,
              labelcolor="#222")
    fig.tight_layout()
    _save(fig, "fig_roc")


def fig_ordering(lora):
    prob, true, pat, cond = lora
    patients = np.unique(pat)
    on, off = {}, {}
    for p in patients:
        m = pat == p
        on[p] = float(prob[m & (cond == "A")][0])
        off[p] = float(prob[m & (cond == "C")][0])
    diff = {p: off[p] - on[p] for p in patients}
    order = sorted(patients, key=lambda p: diff[p])   # inverted (bottom) -> correct (top)
    n_correct = sum(diff[p] > 0 for p in patients)

    fig, ax = plt.subplots(figsize=(6.4, 8.4))
    ax.axvline(0.5, color=GREY, ls=(0, (4, 4)), lw=1.1, zorder=1)
    ax.text(0.5, len(order) + 0.4, "decision threshold 0.5", color=GREY,
            fontsize=8.3, ha="center", va="bottom")

    for i, p in enumerate(order):
        correct = diff[p] > 0
        c = BLUE if correct else VERM
        ax.plot([on[p], off[p]], [i, i], color=c, lw=1.8, alpha=0.85, zorder=2,
                solid_capstyle="round")
    # markers plotted once (for a clean legend): ON = open circle, OFF = filled diamond
    ax.scatter([on[p] for p in order], range(len(order)), s=42, facecolors="white",
               edgecolors=GREY, linewidths=1.3, zorder=3, label="ON session  P(OFF)")
    ax.scatter([off[p] for p in order], range(len(order)), s=46, marker="D",
               facecolors=GREY, edgecolors="white", linewidths=0.8, zorder=3,
               label="OFF session  P(OFF)")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7.2, color="#333")
    ax.set_ylim(-1, len(order) + 1.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean predicted P(OFF)", fontsize=10, color=GREY)
    ax.set_title(f"Within-subject ON vs OFF ordering — {n_correct}/{len(patients)} "
                 f"patients ordered correctly",
                 fontsize=11.5, color="#222", pad=22, loc="left")
    _style_axes(ax)
    ax.grid(axis="y", visible=False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", mfc="white", mec=GREY, mew=1.3, ls="",
               ms=8, label="ON session  P(OFF)"),
        Line2D([0], [0], marker="D", mfc=GREY, mec="white", ls="", ms=8,
               label="OFF session  P(OFF)"),
        Line2D([0], [0], color=BLUE, lw=2.4, label="correct (OFF > ON)"),
        Line2D([0], [0], color=VERM, lw=2.4, label="inverted (OFF < ON)"),
    ]
    # Legend outside the axes (bottom-right rows carry data) — no overlap.
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8.6, frameon=False, labelcolor="#222", handlelength=1.8,
              borderaxespad=0.0)
    fig.tight_layout()
    _save(fig, "fig_ordering")


def _save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUTDIR / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUTDIR / (name + '.png')}  (+ .pdf)")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    lora = _load_lora_sessions()
    probe = _load_probe_sessions()
    fullft = _load_fold_sessions(FULLFT)
    print(f"LoRA sessions: {len(lora[0])} | probe: {len(probe[0])} | full-FT: {len(fullft[0])}")
    fig_roc(lora, probe, fullft)
    fig_ordering(lora)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
