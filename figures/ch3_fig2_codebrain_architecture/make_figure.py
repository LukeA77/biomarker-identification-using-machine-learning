"""Fig 3-2 — CodeBrain backbone with LoRA insertion points.

A schematic. Architecture facts are taken from codebrain_downstream_contract.md in
this folder (verified against CodeBrain's own code and a passing forward-pass gate):
  - backbone = SSSM (EEGSSM), 8 residual blocks, each SGConv GConv + single-token
    sliding-window nn.MultiheadAttention, ~15.2M parameters
  - input contract (B, 18, 10, 200); normalisation by division by 100
  - vector-quantised codebooks NOT loaded downstream (if_codebook=False)
  - backbone output is per-patch (B, 18, 10, 200); the TUAB-style flatten head would
    give 18*10*200 = 36,000 features, which is why a pooled head replaced it

LoRA is inserted on the attention projections only (lora_target = "attention" in the
run records), rank 4, ~90k trainable parameters. Frozen components are shaded.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, OKABE_ITO
from diagram import blank_ax, box, arrow

import matplotlib.pyplot as plt

apply_style()
HERE = pathlib.Path(__file__).parent

FROZEN_FILL, FROZEN_EDGE = "#ececec", "#888888"
TRAINED_FILL, TRAINED_EDGE = "#fdf0e6", OKABE_ITO[1]
NOTUSED_FILL, NOTUSED_EDGE = "#ffffff", "#cccccc"

fig, ax = plt.subplots(figsize=(6.3, 3.9))
blank_ax(ax)

# ---- left column: the data path -------------------------------------------------
xL = 0.24
box(ax, xL, 0.93, 0.34, 0.105, "Input window",
    sub="(B, 18, 10, 200), ÷100", fill="#ffffff", edge="#555555",
    fontsize=8, subsize=6.8)
arrow(ax, (xL, 0.877), (xL, 0.838))

# the eight residual blocks, drawn as a stack
top_blk, blk_h, blk_gap = 0.795, 0.058, 0.070
for i in range(8):
    y = top_blk - i * blk_gap
    box(ax, xL, y, 0.34, blk_h, "EEGSSM block %d" % i, fill=FROZEN_FILL,
        edge=FROZEN_EDGE, fontsize=7.2)
    if i < 7:
        arrow(ax, (xL, y - blk_h / 2), (xL, y - blk_gap + blk_h / 2), lw=0.7)

y_last = top_blk - 7 * blk_gap
arrow(ax, (xL, y_last - blk_h / 2), (xL, 0.205))
box(ax, xL, 0.155, 0.34, 0.100, "Pooled head",
    sub="mean over patches → 2 classes", fill=TRAINED_FILL, edge=TRAINED_EDGE,
    fontsize=8, subsize=6.8)

# ---- right column: what is inside one block -------------------------------------
xR = 0.735
ax.plot([xL + 0.175, 0.545], [0.795 + blk_h / 2, 0.90], color="#aaaaaa",
        linewidth=0.7, linestyle="--", zorder=1)
ax.plot([xL + 0.175, 0.545], [0.795 - blk_h / 2, 0.545], color="#aaaaaa",
        linewidth=0.7, linestyle="--", zorder=1)

box(ax, xR, 0.90, 0.36, 0.075, "SGConv (GConv)", fill=FROZEN_FILL, edge=FROZEN_EDGE,
    fontsize=7.5)
arrow(ax, (xR, 0.8625), (xR, 0.825), lw=0.7)
box(ax, xR, 0.775, 0.36, 0.085, "Sliding-window attention",
    sub="Q, K, V, O projections", fill=FROZEN_FILL, edge=FROZEN_EDGE,
    fontsize=7.5, subsize=6.5)
box(ax, xR, 0.655, 0.36, 0.080, "LoRA adapters",
    sub="rank 4 · ≈90k trainable", fill=TRAINED_FILL, edge=TRAINED_EDGE,
    fontsize=7.5, subsize=6.5)
# LoRA is added to the attention projections, so the connector is a two-headed
# "injected into" link, not a feed-forward step in the data path.
ax.annotate("", xy=(xR, 0.7325), xytext=(xR, 0.695),
            arrowprops=dict(arrowstyle="<->", color=OKABE_ITO[1], linewidth=1.0))
box(ax, xR, 0.545, 0.36, 0.070, "Residual add + norm", fill=FROZEN_FILL,
    edge=FROZEN_EDGE, fontsize=7.5)
# Data path continues attention -> residual, routed clear of the LoRA box.
ax.annotate("", xy=(xR - 0.185, 0.580), xytext=(xR - 0.185, 0.7325),
            arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.9,
                            connectionstyle="arc3,rad=0.35"))
ax.text(xR, 0.965, "inside one block", ha="center", fontsize=7.5, color="#555555")

# ---- not loaded downstream ------------------------------------------------------
box(ax, xR, 0.325, 0.36, 0.090, "VQ codebooks / tokenizer",
    sub="not loaded downstream", fill=NOTUSED_FILL, edge=NOTUSED_EDGE,
    fontsize=7.5, subsize=6.5, textcolor="#999999")

# ---- legend ---------------------------------------------------------------------
ly = 0.135
for i, (fill, edge, lab, tc) in enumerate([
        (FROZEN_FILL, FROZEN_EDGE, "frozen (≈15.2M params)", "#0b0b0b"),
        (TRAINED_FILL, TRAINED_EDGE, "trained", "#0b0b0b"),
        (NOTUSED_FILL, NOTUSED_EDGE, "not used", "#999999")]):
    yy = ly - i * 0.052
    box(ax, xR - 0.155, yy, 0.030, 0.030, "", fill=fill, edge=edge)
    ax.text(xR - 0.132, yy, lab, ha="left", va="center", fontsize=7, color=tc)

save_all(fig, HERE, "fig3-2")
print("saved fig3-2")
