"""Fig 3-4 — The ICLabel detector failure.

Every point is one independent component from the four inspected recordings.
Source: ica_frontal_blink_id.csv (72 components) — columns frontal_z_combined,
iclabel_eye_prob, iclabel_winning. Nothing derived or estimated.

The argument: components far to the right (strong frontal correlation, i.e. blink-like)
but low on the y-axis were not classified as eye by ICLabel. The vertical line is the
|r| >= 0.5 correlation criterion actually used; the horizontal line is the 0.5
probability a winning-class assignment implies.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, OKABE_ITO, ANNOT_GREY, CHANCE_GREY

import matplotlib.pyplot as plt
import pandas as pd

apply_style()
HERE = pathlib.Path(__file__).parent
df = pd.read_csv(HERE / "ica_frontal_blink_id.csv")

STYLE = {
    "eye blink": (OKABE_ITO[1], "D", "ICLabel: eye blink"),
    "brain":     (OKABE_ITO[0], "o", "ICLabel: brain"),
    "other":     ("#555555",    "s", "ICLabel: other"),
}

fig, ax = plt.subplots(figsize=(6.3, 3.6))

ax.axvspan(0.5, 1.02, color="#f2f2f2", zorder=0)
ax.axvline(0.5, color=OKABE_ITO[7], linewidth=1.0, linestyle="--", zorder=2)
ax.axhline(0.5, color=CHANCE_GREY, linewidth=0.9, linestyle=":", zorder=2)

for lab, (c, m, legend) in STYLE.items():
    sub = df[df["iclabel_winning"] == lab]
    ax.plot(sub["frontal_z_combined"], sub["iclabel_eye_prob"], marker=m,
            linestyle="none", markersize=5, color=c, label=legend, zorder=4,
            markeredgecolor="white", markeredgewidth=0.5, alpha=0.9)

# The components recovered by the correlation criterion but missed by ICLabel:
# strong frontal correlation, low eye probability.
missed = df[(df["frontal_z_combined"] >= 0.5) & (df["iclabel_winning"] != "eye blink")]
ax.annotate("recovered by frontal correlation,\nmissed by ICLabel (n = %d)" % len(missed),
            xy=(0.836, 0.135), xytext=(0.80, 0.30), fontsize=8, color=OKABE_ITO[3],
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color=OKABE_ITO[3], linewidth=0.9))

ax.set_xlabel("Frontal correlation with Fp1/Fp2 (|r|)")
ax.set_ylabel("ICLabel eye-blink probability")
ax.set_xlim(0, 1.02)
ax.set_ylim(-0.03, 1.02)
ax.legend(loc="upper left")

ax.text(0.755, 1.005, "correlation criterion |r| ≥ 0.5", fontsize=7.5,
        color=ANNOT_GREY, ha="center", va="top")

save_all(fig, HERE, "fig3-4")
print("saved fig3-4")
print("  n components %d | eye blink %d | missed-but-correlated %d"
      % (len(df), (df["iclabel_winning"] == "eye blink").sum(), len(missed)))
print("  missed eye_prob range %.4f .. %.4f" % (missed["iclabel_eye_prob"].min(),
                                                missed["iclabel_eye_prob"].max()))
