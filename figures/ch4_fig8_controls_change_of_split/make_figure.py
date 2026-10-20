"""Fig 4-8. Controls and CodeBrain arms under a change of split.

Slope chart. Each arm is evaluated twice on the same task-split subset (N = 27):
once within participant, where the held-out participant's own recordings are present
in training, and once cross-participant, where they are genuinely absent. The drop
between the two is the fingerprint evidence.

Every value is read from a results file at render time. Sources:

  within participant
    frozen_probe, lora, full_ft   within_patient_tasksplit_summary.json
                                    -> arms.<arm>.pooled_patient_level
    chanamp, relbp, aperiodic     controls_norm_summary.json
                                    -> summaries.none.<control>.pooled_patient_level
    recid reference               controls_norm_summary.json
                                    -> summaries.none.recid.pooled_patient_level
    Cluster bootstrap over participants, 5,000 resamples.

  cross participant
    all six arms                  lopo_tasksplit_none_summary.json
                                    -> arms["<arm>|both_directions_pooled"]
    Cluster bootstrap over participants, 2,000 resamples, 108 sessions.

Normalisation scheme: "none" is used on both sides so the two columns are comparable.
The chan_z scheme is available in the copied files if a supplementary version is wanted.

Two small differences from the values quoted in the Chapter 4 draft, both explained in
NOTES.md: the draft reports the cross-participant channel-amplitude control as a
patient-level mean (0.495), whereas this figure uses the session-pooled value carried
in the results file (0.472) so that all six arms are measured the same way; and the
draft quotes the within-participant aperiodic control under chan_z (0.886) rather than
under none (0.970).
"""
import sys
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, OKABE_ITO, CHANCE_GREY, ANNOT_GREY

import matplotlib.pyplot as plt

apply_style()
HERE = pathlib.Path(__file__).parent

wp = json.loads((HERE / "within_patient_tasksplit_summary.json").read_text())
ctl = json.loads((HERE / "controls_norm_summary.json").read_text())["summaries"]["none"]
xp = json.loads((HERE / "lopo_tasksplit_none_summary.json").read_text())["arms"]

# key, display label, colour, marker, line width, line style
# Line styles: frozen_probe/blue (L=87) and full_ft/green (L=106) convert to close
# greyscale values (checked: PIL "L" luma), as do aperiodic/pink (L=151) and
# lora/orange (L=162) -- a second close pair beyond the one flagged in review. The
# lighter member of each pair is dashed so the six lines stay separable without
# colour, not just at their (already shape-distinct) endpoint markers.
ARMS = [
    ("aperiodic",    "Aperiodic exponent",     OKABE_ITO[4], "v", 1.2, "-"),
    ("chanamp",      "Channel amplitude",      OKABE_ITO[3], "X", 1.2, "-"),
    ("relbp",        "Relative band power",    OKABE_ITO[7], "s", 2.0, "-"),
    ("frozen_probe", "Frozen probe",           OKABE_ITO[0], "o", 1.2, "-"),
    ("full_ft",      "Full fine-tune",         OKABE_ITO[2], "^", 1.2, (0, (5, 1.5))),
    ("lora",         "LoRA",                   OKABE_ITO[1], "D", 1.2, (0, (1, 1.3))),
]


def within(key):
    if key in wp["arms"]:
        p = wp["arms"][key]["pooled_patient_level"]
    else:
        p = ctl[key]["pooled_patient_level"]
    return p["mean"], p["ci_low"], p["ci_high"]


def cross(key):
    v = xp["%s|both_directions_pooled" % key]
    return (v["aggregate"]["balanced_accuracy"],
            v["bootstrap"]["ci_low"], v["bootstrap"]["ci_high"])


def spread(values, min_gap):
    """Push labels apart vertically so none overlaps, preserving their order.

    Several arms land on identical cross-participant values (frozen probe and the
    aperiodic exponent are both 0.5185), so labels placed at the data value alone
    would sit on top of one another.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for n, i in enumerate(order):
        if n == 0:
            continue
        prev = out[order[n - 1]]
        if out[i] - prev < min_gap:
            out[i] = prev + min_gap
    return out


fig, ax = plt.subplots(figsize=(6.3, 3.8))

X0, X1 = 0.0, 1.0

rows = []
for key, label, colour, marker, lw, ls in ARMS:
    rows.append((key, label, colour, marker, lw, ls, within(key), cross(key)))

label_y = spread([r[7][0] for r in rows], min_gap=0.030)

for (key, label, colour, marker, lw, ls, (w_pt, w_lo, w_hi), (c_pt, c_lo, c_hi)), ly in zip(rows, label_y):
    ax.plot([X0, X1], [w_pt, c_pt], color=colour, linewidth=lw, zorder=3,
            linestyle=ls, solid_capstyle="round", dash_capstyle="round")
    for x, pt, lo, hi in ((X0, w_pt, w_lo, w_hi), (X1, c_pt, c_lo, c_hi)):
        ax.plot([x, x], [lo, hi], color=colour, linewidth=lw * 0.8, zorder=4,
                solid_capstyle="round")
        ax.plot(x, pt, marker=marker, color=colour, markersize=5.5, zorder=5,
                markeredgecolor="white", markeredgewidth=0.6)

    # Leader from the marker across to its displaced label. Kept uniform across all
    # six arms -- an earlier version appended the channel-amplitude collapse inline
    # here, making that one label much wider than its neighbours and unbalancing
    # the column; the collapse now gets its own callout near the line instead (below).
    if key == "chanamp":
        chanamp_c_pt, chanamp_w_pt = c_pt, w_pt
    ax.plot([X1 + 0.035, X1 + 0.105], [c_pt, ly], color=colour, linewidth=0.6,
            zorder=3, alpha=0.8)
    ax.text(X1 + 0.125, ly, label, fontsize=7.5, color=colour,
            ha="left", va="center")

# Explicit recording-identity target, within participant only. It has no
# cross-participant counterpart because the target itself is defined per recording.
# The leader runs nearly horizontal at the star's own height, threading the gap
# between the aperiodic marker (0.970) and the channel-amplitude marker (0.934)
# just above and below it -- an earlier version rose diagonally from lower down
# and visually cut across that stacked cluster on its way to the star.
recid = ctl["recid"]["pooled_patient_level"]
ax.plot(X0, recid["mean"], marker="*", markersize=8, color=ANNOT_GREY, zorder=6,
        markerfacecolor="white", markeredgewidth=0.9)
ax.plot([X0 - 0.20, X0 - 0.025], [recid["mean"], recid["mean"]], color=ANNOT_GREY,
        linewidth=0.6, zorder=3)
ax.text(X0 - 0.21, recid["mean"], "explicit recording-identity\ntarget (%.3f)"
        % recid["mean"], fontsize=7, color=ANNOT_GREY, ha="right", va="center")

# Channel-amplitude collapse: the cleanest single number in the figure (0.934 ->
# 0.472), otherwise left for the reader to read off both axes and subtract. Placed
# as its own callout, matching the recording-identity target's treatment, in the
# one pocket of the chart with no line passing through it (checked: at x <= 0.35,
# every arm's within-to-cross line stays above y = 0.60). The arrow runs mostly
# through that open pocket and approaches the channel-amplitude marker without
# crossing any other marker, though it passes near the full-fine-tune/LoRA lines
# in its final stretch since those arms land close to channel-amplitude at x = 1.
ax.annotate("channel amplitude:\n0.934 → 0.472 (%+.3f)" % (chanamp_c_pt - chanamp_w_pt),
            xy=(X1, chanamp_c_pt), xycoords="data",
            xytext=(0.02, 0.565), textcoords="data",
            fontsize=7, color=OKABE_ITO[3], ha="left", va="center", zorder=6,
            arrowprops=dict(arrowstyle="->", color=OKABE_ITO[3], linewidth=0.9))

ax.axhline(0.5, color=CHANCE_GREY, linewidth=0.9, linestyle="--", zorder=1)
ax.text(X0 - 0.090, 0.508, "chance", fontsize=7.5, color=ANNOT_GREY,
        ha="right", va="bottom")

ax.set_xticks([X0, X1])
ax.set_xticklabels(["Within participant\n(held-out recordings in training)",
                    "Cross participant\n(held-out recordings absent)"])
ax.set_xlim(-0.70, 1.72)
ax.set_ylim(0.38, 1.02)
ax.set_ylabel("Balanced accuracy")
ax.grid(axis="x", visible=False)

save_all(fig, HERE, "fig4-8")

print("saved fig4-8")
for key, label, _, _, _, _ in ARMS:
    w = within(key)
    c = cross(key)
    print("  %-22s within %.3f [%.3f, %.3f]   cross %.3f [%.3f, %.3f]   drop %+.3f"
          % (label, w[0], w[1], w[2], c[0], c[1], c[2], c[0] - w[0]))
print("  recording-identity target within participant %.3f" % recid["mean"])
