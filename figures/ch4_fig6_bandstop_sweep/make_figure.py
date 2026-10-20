"""Fig 4-6 — Band-stop sweep, both train/test directions on a shared colour scale.

Sources (all values read from file, nothing reconstructed):
  bandsweep_multiwidth/bandsweep_table.json               (toast->tea, 469 bands)
  bandsweep_multiwidth_tea_to_toast/bandsweep_table.json  (tea->toast, 469 bands)
Each band record carries lo, hi, width and dba_mean (mean drop in balanced accuracy
across the five seeds). Baselines come from meta.baseline_ba_per_seed.

Plots actual post-filter balanced accuracy (baseline - dba_mean), not the drop.
Revised per review: plotting the drop let a reader misread "biggest drop" (top-left
of panel A) as the headline result on its own, with no visual anchor for how that
related to the arm's actual accuracy. Both panels keep one shared colour scale (the
reverse direction's effect being much smaller is a real finding that a shared scale
makes visible, not an artefact to hide behind per-panel normalisation); each panel's
own baseline is marked as a reference line directly on the shared colorbar instead.

Both panels share one colour scale deliberately: the reverse direction's effect is
much smaller, and a shared scale is what makes that visible rather than hiding it
behind per-panel normalisation.

SEED-STABILITY FLAG: this sweep runs on the pooled duration-matched full fine-tune
arm, whose five-seed spread (0.031) crosses the pre-registered flag threshold
(+/-0.03) without reaching the stop threshold (+/-0.05). Stated in the caption.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, ANNOT_GREY

import matplotlib.pyplot as plt
import numpy as np

apply_style()
HERE = pathlib.Path(__file__).parent


def load(sub):
    """Return {width: (centres, ba_after)} plus baselines.

    Bands step by 1 Hz in lower edge at every width, so the band CENTRE grid is
    offset by half a step between odd and even widths. Building one shared 2-D
    array over all centres therefore leaves every other cell empty. Each width is
    kept as its own row instead and drawn as a separate strip.
    """
    d = json.loads((HERE / sub / "bandsweep_table.json").read_text())
    seeds = d["meta"]["baseline_ba_per_seed"]          # dict keyed by seed number
    base_vals = np.array([seeds[k] for k in sorted(seeds)])
    baseline = base_vals.mean()
    rows = {}
    for b in d["bands"]:
        rows.setdefault(b["width"], []).append(((b["lo"] + b["hi"]) / 2, baseline - b["dba_mean"]))
    for w in rows:
        rows[w] = np.array(sorted(rows[w])).T          # -> (2, n): centres, ba_after
    return rows, base_vals, len(d["bands"])


def draw(ax, rows, vmin, vmax):
    """Draw each width as its own 1-cell-high strip, so no gaps appear."""
    for w, (centres, ba) in rows.items():
        edges_x = np.concatenate([centres - 0.5, [centres[-1] + 0.5]])
        ax.pcolormesh(edges_x, [w - 0.5, w + 0.5], ba[None, :], cmap="viridis",
                      vmin=vmin, vmax=vmax, shading="flat", rasterized=True, zorder=1)


rowsA, baseAv, nA = load("bandsweep_multiwidth_toast_to_tea")
rowsB, baseBv, nB = load("bandsweep_multiwidth_tea_to_toast")

allvals = np.concatenate([r[1] for r in rowsA.values()] + [r[1] for r in rowsB.values()])
vmin, vmax = float(allvals.min()), float(allvals.max())

fig, (axA, axB) = plt.subplots(2, 1, figsize=(6.3, 4.9), sharex=True)

for ax, rows, title in (
    (axA, rowsA, "A  toast → tea (principal direction)"),
    (axB, rowsB, "B  tea → toast (reverse direction)"),
):
    draw(ax, rows, vmin, vmax)
    # Widths >= 8 Hz return an optimum pinned to the sweep's lower boundary and are
    # not interpretable as located effects -- washed out and hatched so the brightest
    # part of the panel is not mistaken for the headline result, which lives below.
    ax.axhspan(7.5, 18.5, facecolor="white", alpha=0.5, edgecolor="none", zorder=4)
    ax.axhspan(7.5, 18.5, facecolor="none", hatch="////", edgecolor="white",
               linewidth=0.0, alpha=0.7, zorder=4)
    ax.axhline(7.5, color="white", linewidth=1.3, linestyle="--", zorder=5)
    ax.text(0.992, 7.5, " interpretable below this line ", transform=ax.get_yaxis_transform(),
            fontsize=7.3, color="black", ha="right", va="center", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    ax.text(0.5, 0.5, "not a located effect: optimum pinned to the 1 Hz sweep boundary",
            transform=ax.transAxes, fontsize=7.5, color="#333333", ha="center", va="center",
            zorder=6, bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=2))
    ax.set_ylabel("Stop-band width (Hz)")
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.grid(False)
    ax.set_yticks([5, 8, 10, 12, 14, 16, 18])
    ax.set_ylim(4.5, 18.5)

axB.set_xlabel("Stop-band centre frequency (Hz)")

sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
cb = fig.colorbar(sm, ax=(axA, axB), pad=0.015, fraction=0.045)
cb.set_label("Balanced accuracy after filtering")
cb.outline.set_linewidth(0.8)
# Numeric ticks moved to the colorbar's left (inner) edge, freeing the right side
# for the baseline callouts below -- at the default position the two collided.
cb.ax.yaxis.set_ticks_position("left")
cb.ax.yaxis.set_label_position("left")

# Each panel's own no-filtering baseline, marked directly on the shared colorbar --
# a white/black double-stroke line so it stays visible against both the dark and
# bright ends of the viridis scale.
for label, base_mean in (("A baseline", baseAv.mean()), ("B baseline", baseBv.mean())):
    cb.ax.axhline(base_mean, color="white", linewidth=2.6, zorder=5)
    cb.ax.axhline(base_mean, color="black", linewidth=1.0, zorder=6)
    cb.ax.text(1.25, base_mean, "%s\n%.3f" % (label, base_mean),
               transform=cb.ax.get_yaxis_transform(), fontsize=7, color="black",
               ha="left", va="center", zorder=6)

save_all(fig, HERE, "fig4-6")
print("saved fig4-6")
print("  seed baselines A:", [round(v,4) for v in baseAv], "mean %.4f spread %.4f" % (baseAv.mean(), baseAv.max()-baseAv.min()))
vA = np.concatenate([r[1] for r in rowsA.values()])
print("  A toast->tea  n=%d baseline=%.4f min BA=%.4f (max drop=%.4f)  n improving=%d"
      % (nA, baseAv.mean(), vA.min(), baseAv.mean() - vA.min(), int((vA > baseAv.mean()).sum())))
vB = np.concatenate([r[1] for r in rowsB.values()])
print("  B tea->toast  n=%d baseline=%.4f min BA=%.4f (max drop=%.4f)  n improving=%d"
      % (nB, baseBv.mean(), vB.min(), baseBv.mean() - vB.min(), int((vB > baseBv.mean()).sum())))
