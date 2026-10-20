"""Fig 4-3 — Negative and positive controls (two panels).

Panel A: refit-based null, 100 pipeline refits with labels shuffled within participant.
  Source: stage4g_duration_confound.json -> null_refit.null (100 values).
  NOTE: Section 4.3 of the draft describes the negative control as "100 refits centred
  at 0.500 (sd 0.058)". The only 100-value refit null on disk is this one
  (mean 0.4996, sd 0.0578) — the same distribution Section 4.4 uses as the refit null
  for the classical arm. stage4f_leakage_checks.json also holds a negative_control
  block but it has only 10 scores (mean 0.526), so it is not the one reported.

Panel B: positive control, participant-identity decoding across the nine layers.
  Source: stage7_positive_control.json -> arms.our_order.*.subject_id_top1,
  raw_bandpower_subject_id.top1, pass_thresholds.chance. All read from file.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, ROLE_COLOR, CHANCE_GREY, ANNOT_GREY, OKABE_ITO

import matplotlib.pyplot as plt
import numpy as np

apply_style()
HERE = pathlib.Path(__file__).parent

null_refit = json.loads((HERE / "stage4g_duration_confound.json").read_text())["null_refit"]
null_vals = np.array(null_refit["null"])

pc = json.loads((HERE / "stage7_positive_control.json").read_text())
layers = ["block0", "block1", "block2", "block3", "block4", "block5", "block6", "block7", "final"]
ident = [pc["arms"]["our_order"][L]["subject_id_top1"] for L in layers]
raw_ceiling = pc["raw_bandpower_subject_id"]["top1"]
identity_chance = pc["pass_thresholds"]["chance"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.3, 3.0))

# ---- Panel A: negative control -------------------------------------------------
axA.hist(null_vals, bins=np.arange(0.34, 0.70, 0.02), color="#bbbbbb",
         edgecolor="white", linewidth=0.6, zorder=2)
axA.axvline(0.5, color=CHANCE_GREY, linewidth=0.9, linestyle="--", zorder=3)
axA.text(0.5, axA.get_ylim()[1] * 0.97, " chance", fontsize=7.5, color=ANNOT_GREY,
         ha="left", va="top", zorder=4)
# The null mean coincides with chance to three decimals, so it is stated rather than
# drawn as a second line on top of the chance line. Placed upper-right, the one corner
# clear of both the tall central bars and the "chance" label (upper-left was over them).
axA.text(0.97, 0.97, "mean %.3f\nSD %.3f" % (null_vals.mean(), null_vals.std(ddof=1)),
         transform=axA.transAxes, fontsize=7.5, color=ANNOT_GREY, ha="right", va="top")
axA.set_xlabel("Balanced accuracy")
axA.set_ylabel("Refits")
axA.set_title("A  Negative control", loc="left", fontsize=9.5)
axA.grid(axis="x", visible=False)

# ---- Panel B: positive control --------------------------------------------------
# No per-layer interval is on disk (stage7_positive_control.json carries point
# estimates only, one train/test split per layer), stated in the caption rather
# than fabricated, so the curve's smoothness is not mistaken for measured precision.
x = np.arange(len(layers))
best_i = int(np.argmax(ident))
final_i = len(layers) - 1
axB.plot(x, ident, marker="o", markersize=4.5, color=ROLE_COLOR["frozen_probe"],
         linewidth=1.4, zorder=4, markeredgecolor="white", markeredgewidth=0.6,
         label="CodeBrain representation")
axB.axhline(raw_ceiling, color=OKABE_ITO[1], linewidth=1.2, linestyle="-.", zorder=3)
# Left end of the line, clear above it: the right end ran off the axes at 9 categories,
# and too small a gap here let the line cut through the text's descenders.
axB.text(0.15, raw_ceiling + 0.022, "raw band-power ceiling",
         fontsize=7.5, color=OKABE_ITO[1], ha="left", va="bottom", zorder=6)
axB.axhline(identity_chance, color=CHANCE_GREY, linewidth=0.9, linestyle="--", zorder=2)
axB.text(-0.4, identity_chance + 0.012, "chance", fontsize=7.5, color=ANNOT_GREY,
         ha="left", va="bottom")

# The headline number: best layer against chance, drawn so the reader does not have
# to divide it out themselves.
ratio = ident[best_i] / identity_chance
arrow_x = best_i
axB.annotate("", xy=(arrow_x, ident[best_i]), xytext=(arrow_x, identity_chance),
             arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.1), zorder=5)
axB.text(arrow_x + 0.15, (ident[best_i] + identity_chance) / 2,
          "%s (peak):\n≈%d× chance" % (layers[best_i].replace("block", "block "), round(ratio)),
          fontsize=8, color="black", ha="left", va="center", zorder=5)

# The final layer is the one every arm in Section 4.2 actually uses, and it is the
# worst of the nine, both facts are in the text but were not visible on the figure.
axB.annotate("final layer: used by every\narm in §4.2, lowest of\nthe 9 layers tested",
             xy=(final_i, ident[final_i]), xycoords="data",
             xytext=(final_i - 2.6, 0.145), textcoords="data",
             fontsize=7.5, color=ANNOT_GREY, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=ANNOT_GREY, linewidth=0.9))

axB.set_xticks(x)
axB.set_xticklabels([L.replace("block", "block ") for L in layers], rotation=40, ha="right")
axB.set_xlabel("Representation layer")
axB.set_ylabel("Participant identity, top-1 accuracy")
axB.set_ylim(0, 0.52)
axB.set_xlim(-0.5, len(layers) - 0.1)
axB.set_title("B  Positive control", loc="left", fontsize=9.5)
axB.grid(axis="x", visible=False)

save_all(fig, HERE, "fig4-3")
print("saved fig4-3  null mean %.4f sd %.4f n=%d | raw ceiling %.4f | final %.4f | best %.4f (%s) | ratio %.1fx"
      % (null_vals.mean(), null_vals.std(ddof=1), len(null_vals), raw_ceiling, ident[-1],
         ident[best_i], layers[best_i], ratio))
