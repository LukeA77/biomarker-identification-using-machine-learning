"""Fig 4-6b (provisional slot) -- Power-removed control, toast->tea only.

Rebuilt from panels C and D of codebrain_finetune/src/plot_bandsweep.py's
existing_render_toast_to_tea.png (the 4-panel fig_xai_bandsweep.png). That render's
panels A/B are superseded by fig4-6 (ch4_fig6_bandstop_sweep, actual-accuracy
toast-vs-tea comparison), so only the confound control (old C) and its residual
(old D) are rebuilt here as a standalone 2-panel figure. Relabelled A/B for this
figure, since every figure in this set starts its own lettering at A -- read A here
as the old C, B here as the old D.

Panel A: the confound the raw sweep can't separate from position -- a lower or
  wider notch removes more 1-45 Hz power by construction, and EEG is 1/f, so low
  bands are a bigger intervention than high bands of equal Hz-width. Points are
  every (position, width) band's drop in balanced accuracy against the fraction of
  power its notch removed; the line is what an isotonic fit, computed SEPARATELY
  WITHIN EACH WIDTH, says any notch removing that much power costs.

  Per-width fitting is not optional: pooled across widths, the isotonic fit would
  average a 5 Hz band against an 18 Hz band at the same power value, and its
  residual would silently re-encode "wider hurts more" rather than removing it.
  Confirmed against source rather than assumed: codebrain_finetune/src/
  plot_bandsweep.py fits IsotonicRegression separately per np.unique(width)
  (lines 108-113), matching codebrain_finetune/src/summarize_bandsweep.py, and the
  per-band expected_from_power / residual fields already on disk in
  bandsweep_table.json were computed that way -- read directly here, not refit.

Panel B: the residual (actual drop minus the isotonic prediction) back on the
  (position x width) grid, toast->tea only -- this control was never run in the
  reverse direction (the multi-width band-sweep finding (EXPERIMENTS.md 16c/16d); the reverse-direction
  residual at the same widths is reported 8-12x smaller in EXPERIMENTS.md #16e, so
  this is a scope boundary to state, not an oversight to fix). Widths >= 8 Hz
  hatched exactly as in fig4-6: the optimum there is pinned to the sweep's 1 Hz
  lower boundary and is not a located effect.

No subtitle text on the figure itself (the original 4-panel render's 5-line
subtitle -- bands tested, "read the profile not the peak", Tier E framing -- moves
to the caption; see NOTES.md draft caption for the exact wording to keep).

Source: bandsweep_multiwidth_toast_to_tea/bandsweep_table.json (local copy of
  codebrain_finetune/results/xai/bandsweep_multiwidth/bandsweep_table.json), whose
  per-band residual/expected_from_power fields were cross-checked directly against
  the band lo=4/hi=9/width=5 entry cited in the multi-width band-sweep finding (EXPERIMENTS.md 16c/16d)
  (residual +0.023, 5/5 seeds): matches to 3 dp.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, ANNOT_GREY

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

apply_style()
HERE = pathlib.Path(__file__).parent

d = json.loads((HERE / "bandsweep_multiwidth_toast_to_tea" / "bandsweep_table.json").read_text())
meta = d["meta"]
bands = d["bands"]

lo = np.array([b["lo"] for b in bands])
hi = np.array([b["hi"] for b in bands])
width = np.array([b["width"] for b in bands])
dba = np.array([b["dba_mean"] for b in bands])
removed = np.array([b["power_removed_frac"] for b in bands])
expected = np.array([b["expected_from_power"] for b in bands])
resid = np.array([b["residual"] for b in bands])
centre = (lo + hi) / 2.0
uw = np.unique(width)
stride = float(meta["stride"])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.3, 3.75),
                                gridspec_kw={"width_ratios": [1, 1.15]})

# ---- Panel A (old "C"): the confound, and the per-width isotonic control -------
sc = axA.scatter(100 * removed, dba, c=width, cmap="viridis", s=10,
                  edgecolors="white", linewidths=0.25, zorder=3)
for i, w in enumerate(uw):
    m = width == w
    o = np.argsort(removed[m])
    axA.plot(100 * removed[m][o], expected[m][o], color="black", linewidth=1.2,
              alpha=0.85, zorder=4, label="isotonic fit, per width" if i == 0 else None)

axA.set_xscale("symlog", linthresh=0.5)
axA.set_xlabel("% of 1-45 Hz power removed (symlog)")
axA.set_ylabel("Drop in balanced accuracy")
axA.legend(frameon=False, fontsize=7.5, loc="upper left")
axA.set_title("A  the confound", loc="left", fontsize=9.5)

axA.text(0.03, 0.80,
          "The plateau at high power removed indicates\nthat most of the drop tracks power removed,\n"
          "not notch position.",
          transform=axA.transAxes, fontsize=7.2, color="black", ha="left", va="top",
          zorder=10, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

cbA = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.02)
# No text label here: it duplicated panel B's y-axis label ("Stop-band width (Hz)",
# the same variable), which sits immediately to its right -- the two read as one
# garbled, stacked label at the default inter-panel spacing. The tick values (5-18)
# are unambiguous on their own once read alongside panel B's axis.
cbA.outline.set_linewidth(0.8)

# ---- Panel B (old "D"): residual after controlling for power removed -----------
lim = float(np.nanmax(np.abs(resid)))
norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
im = None
for w in uw:
    m = width == w
    o = np.argsort(centre[m])
    cvals = centre[m][o]
    edges_x = np.concatenate([cvals - stride / 2, [cvals[-1] + stride / 2]])
    # Faint cell edges: with a tight diverging scale most cells sit near zero
    # (correctly -- the finding is that nothing else survives the control) and
    # render near-white, indistinguishable from blank page without a boundary.
    im = axB.pcolormesh(edges_x, [w - 0.5, w + 0.5], resid[m][o][None, :],
                         cmap="RdBu_r", norm=norm, shading="flat", zorder=1,
                         edgecolors="#d9d9d9", linewidth=0.15)

# Same hatching convention as fig4-6: widths >= 8 Hz are not interpretable. This is
# one message (where the boundary is) and is kept visually separate from the claim
# annotation below (a different message, whether the optimum survives) -- an
# earlier version routed both to the same corner with converging arrows, and a
# reader could not tell whether hatching alone, or the dashed line, was doing the
# boundary work.
axB.axhspan(7.5, 18.5, facecolor="white", alpha=0.5, edgecolor="none", zorder=4)
axB.axhspan(7.5, 18.5, facecolor="none", hatch="////", edgecolor="white",
            linewidth=0.0, alpha=0.7, zorder=4)
axB.axhline(7.5, color="white", linewidth=1.3, linestyle="--", zorder=5)
axB.text(0.988, 7.5, " interpretable below this line ", transform=axB.get_yaxis_transform(),
          fontsize=6.8, color="black", ha="right", va="center", zorder=6,
          bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2))

# The one surviving claim, annotated directly on the panel -- this is the entire
# payoff of this panel and was previously invisible without the caption. Kept
# entirely inside the solid (interpretable, width < 8 Hz) region -- an earlier
# version placed the box up in the hatched zone, so the arrow crossed excluded
# territory and the finding visually associated with the region marked "not
# interpretable". Condensed to two short lines to fit the strip's ~3 Hz height.
claim_band = next(b for b in bands if b["width"] == 5 and b["lo"] == 4.0)
claim_centre = (claim_band["lo"] + claim_band["hi"]) / 2
# Outlined in black regardless of fill colour: a diverging red/blue scale is not
# reliably sign-legible in greyscale, so the cell that carries the claim is marked
# by shape as well as colour, and its sign is also stated in words in the label.
axB.add_patch(plt.Rectangle((claim_centre - stride / 2, claim_band["width"] - 0.5),
                             stride, 1.0, fill=False, edgecolor="black",
                             linewidth=1.4, zorder=6))
axB.annotate("residual ≈ +%.3f (5 Hz width,\n%d/5 seeds). Rest of strip is\nnoise; see caption."
              % (claim_band["residual"], claim_band["seeds_agreeing"]),
              xy=(claim_centre, 5), xycoords="data",
              xytext=(12.5, 6.9), textcoords="data",
              fontsize=6.9, color="black", ha="left", va="top", zorder=6,
              arrowprops=dict(arrowstyle="->", color="black", linewidth=1.0),
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.3))

# The rest of the interpretable region's pale scatter is noise, not structure --
# checked directly (see NOTES.md): mean residual 0.0000, SD 0.0055 across the 117
# interpretable-region bands, and only 3 bands besides the claim exceed |0.015|,
# all immediately adjacent to it (position 1-5 Hz, width 5-6), not scattered at
# random. A one-line on-panel version was tried and did not fit this strip without
# colliding with the other two annotations already competing for it (boundary line
# label, claim callout) -- stated in the caption instead (see NOTES.md), which is
# the explicit fallback for this case rather than crowding a fourth text block in.

axB.set_xlabel("Stop-band centre frequency (Hz)")
axB.set_ylabel("Stop-band width (Hz)")
axB.set_yticks([5, 8, 10, 12, 14, 16, 18])
axB.set_ylim(4.5, 18.5)
axB.grid(False)
axB.set_title("B  residual (toast → tea only)", loc="left", fontsize=9.5)

cbB = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.02)
cbB.set_label("ΔBA beyond power removal")
cbB.outline.set_linewidth(0.8)
cbB.ax.axhline(0.0, color="black", linewidth=0.8)

save_all(fig, HERE, "fig4-6b")
print("saved fig4-6b")
print("  claim band lo=%.0f hi=%.0f width=%.0f residual=%.4f seeds_agreeing=%d"
      % (claim_band["lo"], claim_band["hi"], claim_band["width"],
         claim_band["residual"], claim_band["seeds_agreeing"]))
print("  resid min/max %.4f / %.4f" % (resid.min(), resid.max()))
