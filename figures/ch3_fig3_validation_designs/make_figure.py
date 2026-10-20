"""Fig 3-3 — The three validation designs compared.

A schematic; no data underlies it. Structure follows dissertation/ch3_methodology_v2.md §3.6
(copied into source_text_section_3.6.md in this folder). The point the figure has to
make: in the per-participant design the held-out participant's own recordings appear
on both sides of the split, and in the pooled design every participant appears on both
sides; only LOPO removes the test participant entirely.

Colour convention held from the results figures: training material grey, evaluation
material orange, the leakage route marked in vermillion.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, OKABE_ITO
from diagram import blank_ax, box, arrow

import matplotlib.pyplot as plt

apply_style()
HERE = pathlib.Path(__file__).parent

TRAIN_FILL, TRAIN_EDGE = "#ececec", "#777777"
TEST_FILL, TEST_EDGE = "#fdf0e6", OKABE_ITO[1]
LEAK = OKABE_ITO[3]

fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.1))
for ax in axes:
    blank_ax(ax)

# ---------------------------------------------------------------- Panel A: LOPO
ax = axes[0]
ax.set_title("A  Leave-one-participant-out", loc="left", fontsize=8.5)
ax.text(0.5, 0.90, "primary design", ha="center", fontsize=7.5, color="#555555")
box(ax, 0.5, 0.66, 0.80, 0.26, "Participants 1–34", sub="both sessions, ON + OFF",
    fill=TRAIN_FILL, edge=TRAIN_EDGE, fontsize=7.5, subsize=6.5)
ax.text(0.5, 0.485, "train", ha="center", fontsize=7, color="#555555", style="italic")
arrow(ax, (0.5, 0.53), (0.5, 0.44))
box(ax, 0.5, 0.30, 0.80, 0.26, "Participant 35", sub="both sessions, ON + OFF",
    fill=TEST_FILL, edge=TEST_EDGE, fontsize=7.5, subsize=6.5)
ax.text(0.5, 0.13, "test", ha="center", fontsize=7, color=OKABE_ITO[1], style="italic")
ax.text(0.5, 0.03, "test participant absent\nfrom training", ha="center", fontsize=6.8,
        color="#2b7a3d", va="bottom")

# ------------------------------------------- Panel B: within-participant task split
ax = axes[1]
ax.set_title("B  Per-participant task split", loc="left", fontsize=8.5)
ax.text(0.5, 0.90, "secondary design", ha="center", fontsize=7.5, color="#555555")
box(ax, 0.5, 0.66, 0.86, 0.26, "One participant · task A",
    sub="ON + OFF, same recordings", fill=TRAIN_FILL, edge=TRAIN_EDGE,
    fontsize=7.5, subsize=6.5)
ax.text(0.5, 0.485, "train", ha="center", fontsize=7, color="#555555", style="italic")
arrow(ax, (0.5, 0.53), (0.5, 0.44))
box(ax, 0.5, 0.30, 0.86, 0.26, "Same participant · task B",
    sub="ON + OFF, same recordings", fill=TEST_FILL, edge=TEST_EDGE,
    fontsize=7.5, subsize=6.5)
ax.annotate("", xy=(0.90, 0.30), xytext=(0.90, 0.66),
            arrowprops=dict(arrowstyle="<->", color=LEAK, linewidth=1.1,
                            connectionstyle="arc3,rad=-0.45"))
ax.text(0.5, 0.03, "same two recordings on\nboth sides of the split", ha="center",
        fontsize=6.8, color=LEAK, va="bottom")

# ------------------------------------------------------- Panel C: pooled task split
ax = axes[2]
ax.set_title("C  Pooled task split", loc="left", fontsize=8.5)
ax.text(0.5, 0.90, "one shared model", ha="center", fontsize=7.5, color="#555555")
box(ax, 0.5, 0.66, 0.86, 0.26, "All participants · task A",
    sub="ON + OFF", fill=TRAIN_FILL, edge=TRAIN_EDGE, fontsize=7.5, subsize=6.5)
ax.text(0.5, 0.485, "train", ha="center", fontsize=7, color="#555555", style="italic")
arrow(ax, (0.5, 0.53), (0.5, 0.44))
box(ax, 0.5, 0.30, 0.86, 0.26, "All participants · task B",
    sub="ON + OFF", fill=TEST_FILL, edge=TEST_EDGE, fontsize=7.5, subsize=6.5)
ax.annotate("", xy=(0.90, 0.30), xytext=(0.90, 0.66),
            arrowprops=dict(arrowstyle="<->", color=LEAK, linewidth=1.1,
                            connectionstyle="arc3,rad=-0.45"))
ax.text(0.5, 0.03, "every participant on\nboth sides of the split", ha="center",
        fontsize=6.8, color=LEAK, va="bottom")

save_all(fig, HERE, "fig3-3")
print("saved fig3-3")
