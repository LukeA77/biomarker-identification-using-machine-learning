"""Fig 4-5 — The pooling effect. Every value read from a results file.

Panel A: per-participant design (one model fitted per participant). Per-participant
  balanced accuracies shown as a strip, N=27, with the patient-mean and its cluster
  bootstrap 95% CI (5,000 resamples over participants).
  Sources: within_patient_tasksplit_summary.json -> arms.<arm>.pooled_patient_level
           controls_summary.json -> controls.chanamp.pooled_patient_level
Panel B: pooled design (one shared model), toast->tea direction, unmatched and
  duration-matched.
  Source: pooled_matched_vs_unmatched.csv (unmatched_BA/u_lo/u_hi, matched_BA/m_lo/m_hi)

Seed-stability note: the pooled full fine-tune arm is the arm carrying the
pre-registered seed-stability flag (five-seed spread 0.031, flag +/-0.03 crossed,
stop +/-0.05 not reached). Stated in the caption.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, chance_line, ROLE_COLOR, ROLE_MARKER, ANNOT_GREY, CHANCE_GREY

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

apply_style()
HERE = pathlib.Path(__file__).parent
rng = np.random.default_rng(0)   # jitter only; does not alter any plotted value

wp = json.loads((HERE / "within_patient_tasksplit_summary.json").read_text())
ctl = json.loads((HERE / "controls_summary.json").read_text())
pooled = pd.read_csv(HERE / "pooled_matched_vs_unmatched.csv")
pooled = pooled[pooled["direction"] == "toast_to_tea"]

# ---- Panel A data: per-participant design --------------------------------------
panelA = []
for role, label in [("frozen_probe", "Frozen\nprobe"), ("lora", "LoRA"), ("full_ft", "Full\nfine-tune")]:
    pl = wp["arms"][role]["pooled_patient_level"]
    panelA.append((label, role, pl["mean"], pl["ci_low"], pl["ci_high"],
                   list(pl["per_patient"].values())))
pl = ctl["controls"]["chanamp"]["pooled_patient_level"]
panelA.append(("Amplitude\ncontrol", "chanamp", pl["mean"], pl["ci_low"],
               pl["ci_high"], list(pl["per_patient"].values())))

# ---- Panel B data: pooled design ------------------------------------------------
order = [("frozen_probe", "Frozen\nprobe"), ("lora", "LoRA"),
         ("full_ft", "Full\nfine-tune"), ("chanamp", "Amplitude\ncontrol")]
panelB = []
for role, label in order:
    r = pooled[pooled["arm"] == role].iloc[0]
    panelB.append((label, role,
                   r["unmatched_BA"], r["u_lo"], r["u_hi"],
                   r["matched_BA"], r["m_lo"], r["m_hi"]))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.3, 4.05), sharey=True)

# A vertical separator ahead of the control column in both panels, so it reads as
# "three models, plus a sanity-check feature" rather than a fourth equal option.
sep_x = 2.5
for ax in (axA, axB):
    ax.axvline(sep_x, color="#cccccc", linewidth=0.9, linestyle=":", zorder=1)

# ---- Panel A -------------------------------------------------------------------
for xi, (label, role, mean, lo, hi, per_pt) in enumerate(panelA):
    c = ROLE_COLOR[role]
    is_control = role == "chanamp"
    # The control is drawn hollow (outline only): the amplitude feature carries no
    # cortical information, so its points should not read as a fourth trained model.
    jitter = rng.uniform(-0.13, 0.13, len(per_pt))
    axA.plot(xi + jitter, per_pt, marker=ROLE_MARKER[role], linestyle="none",
             markersize=2.8, color=c, alpha=0.40, zorder=3,
             markeredgecolor="none" if not is_control else c,
             markerfacecolor="none" if is_control else c)
    axA.plot([xi + 0.28, xi + 0.28], [lo, hi], color=c, linewidth=1.4,
             solid_capstyle="round", zorder=4)
    axA.plot(xi + 0.28, mean, marker=ROLE_MARKER[role],
              color=c, markersize=6, zorder=5,
              markerfacecolor="none" if is_control else c,
              markeredgecolor=c if is_control else "white",
              markeredgewidth=1.4 if is_control else 0.7)

axA.set_xticks(range(len(panelA)))
axA.set_xticklabels(["Frozen\nprobe", "LoRA", "Full\nfine-tune", "Amplitude\ncontrol"])
axA.get_xticklabels()[3].set_color(ANNOT_GREY)
axA.get_xticklabels()[3].set_fontstyle("italic")
axA.set_ylabel("Balanced accuracy")
axA.set_title("A  Per-participant design", loc="left", fontsize=9.5)
axA.grid(axis="x", visible=False)

# The headline fact this figure exists to show: a feature with zero capacity to
# carry cortical information outscores every trained arm. The arrow is kept vertical
# and directly above the mean/CI marker (not the jittered cloud, which sits at
# xi=3+/-0.13 vs. the marker at xi+0.28=3.28) so it does not cut across the data it
# is pointing at -- an earlier diagonal version crossed straight through the cluster.
ctl_mean = panelA[3][2]
axA.annotate("0.934: above every\ntrained arm here",
             xy=(3.28, ctl_mean), xycoords="data",
             xytext=(3.28, 1.055), textcoords="data",
             fontsize=8, color="black", ha="center", va="bottom",
             arrowprops=dict(arrowstyle="->", color="black", linewidth=1.0,
                              shrinkA=2, shrinkB=4))

# ---- Panel B -------------------------------------------------------------------
for xi, (label, role, u, ulo, uhi, m, mlo, mhi) in enumerate(panelB):
    c = ROLE_COLOR[role]
    for dx, val, vlo, vhi, mfc in ((-0.15, u, ulo, uhi, c), (0.15, m, mlo, mhi, "white")):
        axB.plot([xi + dx, xi + dx], [vlo, vhi], color=c, linewidth=1.4,
                 solid_capstyle="round", zorder=4)
        axB.plot(xi + dx, val, marker=ROLE_MARKER[role], color=c, markersize=6,
                 markerfacecolor=mfc, markeredgecolor=c, markeredgewidth=1.0, zorder=5)
    axB.plot([xi - 0.15, xi + 0.15], [u, m], color=c, linewidth=0.8, zorder=3)

axB.set_xticks(range(len(panelB)))
axB.set_xticklabels(["Frozen\nprobe", "LoRA", "Full\nfine-tune", "Amplitude\ncontrol"])
axB.get_xticklabels()[3].set_color(ANNOT_GREY)
axB.get_xticklabels()[3].set_fontstyle("italic")
axB.set_title("B  Pooled design", loc="left", fontsize=9.5)
axB.grid(axis="x", visible=False)
axB.text(0.03, 0.97, "filled  unmatched\nopen    duration-matched",
         transform=axB.transAxes, fontsize=7.5, color=ANNOT_GREY, va="top", ha="left")

# The point of this panel: matching changed nothing. Every filled/open pair overlaps.
max_move = max(abs(u - m) for _, _, u, _, _, m, _, _ in panelB)
axB.text(0.03, 0.80, "duration matching moved every arm\nby ≤ %.3f; all intervals overlap" % max_move,
          transform=axB.transAxes, fontsize=7.5, color=ANNOT_GREY, va="top", ha="left")

# One "chance" label is enough for two panels sharing a y-axis; panel B keeps the
# reference line but not a second, redundant text label.
chance_line(axA)
axB.axhline(0.5, color=CHANCE_GREY, linewidth=0.9, linestyle="--", zorder=1)

# Extended well past 1.02: with the annotation now sitting above the data (not
# threaded through it), it needs its own headroom, not just clearance for the points.
for ax in (axA, axB):
    ax.set_ylim(0.40, 1.10)
    ax.set_xlim(-0.55, 3.65)

# supxlabel (not fig.text): constrained_layout reserves space for it automatically,
# where a raw fig.text call at a guessed y-position collided with the tick labels.
fig.supxlabel(
    "Within-participant / pooled leakage upper bounds,\n"
    "not comparable to this chapter's cross-participant LOPO estimates.",
    fontsize=7.5, color=ANNOT_GREY, fontstyle="italic")

save_all(fig, HERE, "fig4-5")
print("saved fig4-5")
for label, role, mean, lo, hi, per_pt in panelA:
    print("  A %-22s mean=%.3f ci=[%.3f,%.3f] n=%d" % (role, mean, lo, hi, len(per_pt)))
for label, role, u, ulo, uhi, m, mlo, mhi in panelB:
    print("  B %-22s unmatched=%.3f matched=%.3f" % (role, u, m))
