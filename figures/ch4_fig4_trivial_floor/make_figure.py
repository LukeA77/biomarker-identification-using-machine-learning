"""Fig 4-4 — The trivial floor: two no-EEG heuristics against the trained arms.

Heuristics read from file:
  duration-only  stage4g_duration_confound.json -> duration_only.bal_acc = 0.6142857
  elapsed-time   stage4f_leakage_checks.json    -> time_only.bal_acc     = 0.5714286
CodeBrain arm point estimates and CIs are the reported values from
dissertation/ch4_results.md Table 4-1 (cross-checked against runs_master.csv where present;
that CSV carries no interval columns). Flagged in NOTES.md.

The heuristics have no confidence interval on disk, so they are drawn as reference
lines rather than as points with fabricated error bars.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, chance_line, ROLE_COLOR, ROLE_MARKER, NULL_BAND_GREY, ANNOT_GREY, OKABE_ITO

import matplotlib.pyplot as plt
import numpy as np

apply_style()
HERE = pathlib.Path(__file__).parent

g = json.loads((HERE / "stage4g_duration_confound.json").read_text())
f = json.loads((HERE / "stage4f_leakage_checks.json").read_text())
duration_ba = g["duration_only"]["bal_acc"]
elapsed_ba = f["time_only"]["bal_acc"]

arms = [
    ("Frozen probe\n(mean-pool)", 0.443, 0.357, 0.529, "frozen_mean_pool"),
    ("Frozen probe\n(flatten)",   0.457, 0.371, 0.543, "frozen_flatten"),
    ("LoRA\n(primary arm)",       0.557, 0.457, 0.643, "lora"),
    ("Full\nfine-tune",           0.443, 0.357, 0.529, "full_ft"),
    ("LoRA,\nglobal-z input",     0.443, 0.329, 0.557, "lora_global_z"),
]

fig, ax = plt.subplots(figsize=(6.3, 3.4))
# The band's top edge (0.60) sits only 0.014 below the window-count heuristic line
# (0.614); without a distinct border and an explicit label, that gap reads as if the
# heuristic itself were the band's edge. Both are added so the two are not conflated.
ax.axhspan(0.40, 0.60, color=NULL_BAND_GREY, zorder=0)
ax.axhline(0.60, color="#aaaaaa", linewidth=0.8, zorder=1)
ax.text(0.012, 0.595, "pre-registered\nnull band (≤ 0.60)", transform=ax.get_yaxis_transform(),
        fontsize=7.5, color=ANNOT_GREY, ha="left", va="top", zorder=2)

x = np.arange(len(arms))
for xi, (label, pt, lo, hi, role) in zip(x, arms):
    c = ROLE_COLOR[role]
    is_primary = role == "lora"
    ax.plot([xi, xi], [lo, hi], color=c, linewidth=1.4, solid_capstyle="round", zorder=3)
    ax.plot(xi, pt, marker=ROLE_MARKER[role], color=c, markersize=5, zorder=4,
            markeredgecolor="black" if is_primary else "white",
            markeredgewidth=1.4 if is_primary else 0.7)

# The two heuristics, drawn as reference lines across the trained arms
ax.axhline(duration_ba, color=OKABE_ITO[7], linewidth=1.3, linestyle="-", zorder=5)
ax.text(0.988, duration_ba + 0.005, "window-count heuristic (no EEG)",
        transform=ax.get_yaxis_transform(), fontsize=7.5, color=OKABE_ITO[7],
        ha="right", va="bottom", zorder=6)
ax.axhline(elapsed_ba, color=OKABE_ITO[7], linewidth=1.1, linestyle=(0, (4, 2)), zorder=5)
ax.text(0.988, elapsed_ba + 0.005, "elapsed-time heuristic (no EEG)",
        transform=ax.get_yaxis_transform(), fontsize=7.5, color=OKABE_ITO[7],
        ha="right", va="bottom", zorder=6)

# The headline gap: LoRA is the highest-scoring trained arm and still falls short of
# the trivial floor. Stated on the figure so the reader is not left to subtract it.
# Placed on LoRA's left (the right side is already occupied by the two heuristic
# labels, which run underneath this height and collided with a right-side placement).
lora_x, lora_pt = 2, arms[2][1]
gap_x = lora_x - 0.30
# shrinkA/B=0: annotate's default 2pt end-shrink was leaving the arrow visibly short
# of the window-count line, decoupling the drawn span from the "0.057" in the label.
ax.annotate("", xy=(gap_x, duration_ba), xytext=(gap_x, lora_pt),
            arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.1,
                             shrinkA=0, shrinkB=0), zorder=6)
ax.text(gap_x - 0.09, (duration_ba + lora_pt) / 2,
        "%.3f short of\nthe trivial floor" % (duration_ba - lora_pt),
        fontsize=7.5, color="black", ha="right", va="center", zorder=6)

chance_line(ax)
ax.set_xticks(x)
labels = [a[0] for a in arms]
ax.set_xticklabels(labels)
ax.get_xticklabels()[2].set_fontweight("bold")
ax.set_ylim(0.40, 0.70)
ax.set_xlim(-0.5, len(arms) - 0.5)
ax.set_ylabel("Balanced accuracy")
ax.grid(axis="x", visible=False)

save_all(fig, HERE, "fig4-4")
print("saved fig4-4  duration %.4f  elapsed %.4f" % (duration_ba, elapsed_ba))
