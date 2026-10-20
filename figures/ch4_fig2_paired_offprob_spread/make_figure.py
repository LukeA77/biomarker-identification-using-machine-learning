"""Fig 4-2 (provisional slot) -- Paired P(OFF) lines and per-participant spread, LoRA arm.

Panel A: each participant's ON-session and OFF-session predicted P(OFF), joined by a
  line, ordered by within-participant difference (OFF minus ON). Threshold-free
  evidence behind the Section 4.2 sign test (20/35 correctly ordered, p=0.25) and
  Wilcoxon signed-rank test (mean shift +0.040, p=0.18).
Panel B: per-participant session-level balanced accuracy. With one ON and one OFF
  session per participant, this can only take the values 0.0, 0.5 or 1.0, so it is
  drawn as a count bar chart rather than a continuous distribution, with the cohort
  mean (0.557, the pooled LOPO estimate reported in Table 4-1) marked for contrast.

Panels A and B use different notions of "correct" and do not reconcile numerically:
  A is threshold-free (does OFF's predicted probability exceed ON's), B applies the
  0.5 decision threshold to each session independently. A pair can be correctly
  ordered in A while one of its two sessions still lands on the wrong side of 0.5 in
  B. See the draft caption in NOTES.md for the caption clause stating this.

Source: lora_paired_session_probs.csv in this folder, derived from
  codebrain_finetune/results/nested_lora/fold_*.json (see NOTES.md for the
  cross-check against the reported Section 4.2 statistics).
"""
import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, chance_line, ROLE_COLOR, CHANCE_GREY, ANNOT_GREY, OKABE_ITO

import matplotlib.pyplot as plt
import numpy as np

apply_style()
HERE = pathlib.Path(__file__).parent
LORA = ROLE_COLOR["lora"]

rows = list(csv.DictReader((HERE / "lora_paired_session_probs.csv").open()))
p_on = np.array([float(r["p_off_on_session"]) for r in rows])
p_off = np.array([float(r["p_off_off_session"]) for r in rows])
diff = np.array([float(r["diff_off_minus_on"]) for r in rows])
session_ba = np.array([float(r["session_balanced_accuracy"]) for r in rows])
n = len(rows)

order = np.argsort(diff)
p_on_s, p_off_s = p_on[order], p_off[order]
n_correct = int((diff > 0).sum())
mean_shift = diff.mean()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.3, 3.2), gridspec_kw={"width_ratios": [1.55, 1]})

# ---- Panel A: paired ON/OFF predicted P(OFF), one line per participant ----------
# x carries only rank order, no continuous meaning, so a small fixed-seed jitter
# (same offset applied to both of a participant's points and their connecting line)
# is used purely to separate markers in the 0.55-0.68 band where several pairs cluster.
x = np.arange(n)
jitter = np.random.default_rng(0).uniform(-0.55, 0.55, size=n)
x_j = x + jitter
axA.vlines(x_j, p_on_s, p_off_s, color="#bbbbbb", linewidth=0.9, zorder=2)
axA.scatter(x_j, p_on_s, marker="D", s=20, facecolors="white", edgecolors=LORA,
            linewidths=1.1, zorder=3, label="ON session")
axA.scatter(x_j, p_off_s, marker="D", s=20, facecolors=LORA, edgecolors=LORA,
            linewidths=1.1, zorder=4, label="OFF session")
chance_line(axA, xpos=0.012)

axA.text(0.03, 0.03,
         "%d/%d correctly ordered\n(sign test p = 0.25)\nmean shift +%.3f\n(Wilcoxon p = 0.18)"
         % (n_correct, n, mean_shift),
         transform=axA.transAxes, fontsize=7.5, color=ANNOT_GREY, ha="left", va="bottom")
axA.legend(loc="upper left", ncol=2, handletextpad=0.4, columnspacing=1.0,
           borderaxespad=0.3)
axA.set_xlim(-1, n)
axA.set_ylim(-0.05, 1.22)
axA.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axA.set_xlabel("Participants, ordered by within-participant\ndifference (OFF minus ON)")
axA.set_ylabel("Predicted P(OFF), LoRA arm")
axA.set_xticks([])
axA.grid(axis="x", visible=False)
axA.set_title("A  Paired session probabilities", loc="left", fontsize=9.5)

# ---- Panel B: per-participant thresholded outcome, count bar chart --------------
# Three discrete outcomes (0, 1 or 2 of a participant's 2 sessions classified
# correctly at the 0.5 threshold), not points on a continuous accuracy scale -- the
# axis is labelled with the outcome names rather than 0.0/0.5/1.0 for that reason.
levels = [0.0, 0.5, 1.0]
xpos = [0, 2.3, 4.6]
counts = [int((session_ba == lv).sum()) for lv in levels]
# Single-line labels: multi-line tick labels via set_xticklabels rendered staggered
# and overlapping in this panel regardless of spacing, so line breaks are avoided here.
category_names = ["Both wrong", "One correct", "Both correct"]
fractions = ["0/2 correct", "1/2 correct", "2/2 correct"]

axB.bar(xpos, counts, width=1.3, color=LORA, zorder=3)
for xi, c, frac in zip(xpos, counts, fractions):
    axB.text(xi, c + 0.7, "%s\nn = %d (%.0f%%)" % (frac, c, 100 * c / n), ha="center",
              va="bottom", fontsize=7.5, color=ANNOT_GREY, zorder=4)

cohort_mean = session_ba.mean()
# Interpolated position of the cohort mean on this categorical axis, for a marker
# line only; the exact value is stated in the text box, not read off this axis.
# Height capped at the tallest bar (23) so the line stays clear of the bar-top
# annotations above it, and drawn above the bars (zorder) so it is not hidden
# behind the opaque fill.
mean_x = 2.3 + (cohort_mean - 0.5) / 0.5 * 2.3
axB.vlines(mean_x, 0, 23, color=OKABE_ITO[7], linewidth=1.3, linestyle="--", zorder=5)
axB.text(0.03, 0.97, "pooled session BA\n(cohort mean): %.3f\n(dashed line)" % cohort_mean,
          transform=axB.transAxes, fontsize=7.5, color=OKABE_ITO[7], ha="left", va="top")

axB.set_xticks(xpos)
axB.set_xticklabels(category_names, fontsize=7.3)
axB.set_xlim(-1.3, 5.9)
axB.set_ylim(0, max(counts) + 13)
axB.set_xlabel("Outcome across the participant's\ntwo sessions (0.5 threshold)")
axB.set_ylabel("Participants (N = %d)" % n)
axB.grid(axis="x", visible=False)
axB.set_title("B  Per-participant spread", loc="left", fontsize=9.5)

save_all(fig, HERE, "fig4-2")
print("saved fig4-2  n=%d  correctly_ordered=%d/%d  mean_shift=%.4f  session_BA_counts=%s  cohort_mean=%.4f"
      % (n, n_correct, n, mean_shift, dict(zip(levels, counts)), cohort_mean))
