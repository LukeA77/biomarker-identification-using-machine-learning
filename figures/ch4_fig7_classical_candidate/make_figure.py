"""Fig 4-7 -- The classical spectral candidate, two panels. All values from results files.

Panel C of the original three-panel version (refit-based null for the test-selected
estimate) is dropped here: it duplicates the marked refit-null histogram that lives
in fig4-3 panel A (same underlying null, codebrain_finetune identical distribution
mean 0.500 / SD 0.058 -- see fig4-3's NOTES.md, which already documented this overlap
and reserved the decision of where the marked version should live). This figure now
carries only the two panels that don't duplicate anything: the declared-variant
comparison and the spectral profile.

Panel A: the six pre-declared feature-set variants plus the nested selection-unbiased
  estimate, with the test-selected (selection-biased) estimate visually separated
  rather than sharing a row with the pre-declared variants.
  Sources: stage4b_bandpower_lopo.json -> results[*].aggregate/.bootstrap
           stage4e_nested_classical.json -> aggregate/bootstrap
  The test-selected rel-without-gamma variant (0.657, stage5b_input_norm.json ->
  reference.bandpower_rel_nogamma) is drawn as a point WITHOUT an interval, because
  only its point estimate is stored in the run artefacts. It is not given a fabricated
  error bar.
  NOTE: these bootstraps used 2,000 resamples (n_valid_resamples in the files), not
  the 5,000 used for the task-split arms. The caption states 2,000 for this figure.

Panel B: within-participant OFF-minus-ON relative power across 0.5 Hz bins, 2-14 Hz,
  with each participant's own peak frequency shown as a rug (N=35).
  Source: stage4d_spectral_profile.json -> freqs, profile_2_14Hz, per_patient_peak_hz

  IMPORTANT, checked directly against codebrain_finetune/src/stage4d_spectral_profile.py
  (lines 180-194) before drawing this rug: per_patient_peak_hz is an argmax computed
  over a HARDCODED 3.5-7.5 Hz window ("tremor_band"), not over the full 2-14 Hz range
  the curve above it plots. 13/35 participants land exactly at 3.5 Hz and 8/35 exactly
  at 7.5 Hz -- the two window edges, not a data-driven clustering. This is real and
  already correctly interpreted elsewhere in the project (EXPERIMENTS.md Exp 6c;
  the low-frequency slowing candidate): edge-pinning is the signature
  of a broad monotonic effect the window is too narrow to bound, i.e. evidence AGAINST
  a discrete tremor-frequency peak, not for one. The dissertation caption text drafted
  elsewhere (CAPTIONS.md) does not carry this caveat and reads as if the rug shows
  genuine per-participant peaks -- that would be misleading on its own. Fixed here by
  stating the window and its edges directly on the panel, not by removing the rug.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, ROLE_COLOR, CHANCE_GREY, ANNOT_GREY, OKABE_ITO

import matplotlib.pyplot as plt
import numpy as np

apply_style()
HERE = pathlib.Path(__file__).parent

b4 = json.loads((HERE / "stage4b_bandpower_lopo.json").read_text())
nested = json.loads((HERE / "stage4e_nested_classical.json").read_text())
prof = json.loads((HERE / "stage4d_spectral_profile.json").read_text())
selected = json.loads((HERE / "stage5b_input_norm.json").read_text())["reference"]["bandpower_rel_nogamma"]

LABEL = {"logabs": "Log absolute", "logabs+subjnorm": "Log abs\n+ subj-norm",
         "rel": "Relative", "rel+subjnorm": "Relative\n+ subj-norm",
         "beta": "Beta only", "beta+subjnorm": "Beta only\n+ subj-norm"}

fig, (axA, axB) = plt.subplots(2, 1, figsize=(6.3, 6.7), gridspec_kw={"height_ratios": [1.1, 1.0]})

# ---- Panel A ----------------------------------------------------------------------
rows = []
for r in b4["results"]:
    rows.append((LABEL[r["tag"]], r["tag"], r["aggregate"]["balanced_accuracy"],
                 r["bootstrap"]["ci_low"], r["bootstrap"]["ci_high"], False))
rows.append(("Nested\n(unbiased)", "nested", nested["aggregate"]["balanced_accuracy"],
             nested["bootstrap"]["ci_low"], nested["bootstrap"]["ci_high"], True))

# The test-selected point gets a visual gap (not just the next x slot) plus a
# vertical rule: it is a fundamentally different kind of number (selection-biased,
# no stored interval, already reported in full in Table 4-1 and the text) and an
# earlier version gave it equal billing in the same row as the six declared
# variants, with its label running off the plot edge.
GAP = 0.9
xs = list(range(len(rows)))
xi_sel = xs[-1] + 1 + GAP

for xi, (label, tag, pt, lo, hi, is_nested) in zip(xs, rows):
    c = OKABE_ITO[7] if is_nested else "#555555"
    mk = "^" if is_nested else "o"
    axA.plot([xi, xi], [lo, hi], color=c, linewidth=1.4, solid_capstyle="round", zorder=3)
    axA.plot(xi, pt, marker=mk, color=c, markersize=6, zorder=4,
              markeredgecolor="white", markeredgewidth=0.7)
    # The one finding this panel makes: relative power (undeclared subject-norm,
    # all bands) is the only declared variant whose interval excludes chance.
    if tag == "rel":
        axA.annotate("only declared variant\nwhose CI excludes chance",
                      xy=(xi, lo), xycoords="data", xytext=(xi + 0.55, 0.355),
                      textcoords="data", fontsize=7.2, color="#555555",
                      ha="left", va="bottom",
                      arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.9))
    # The a priori hypothesis (beta band, the textbook levodopa marker) failing is
    # a citable result, not just one more low bar -- annotated so it reads as one.
    if tag == "beta":
        axA.annotate("a priori hypothesis\n(beta band): at chance",
                      xy=(xi, pt), xycoords="data", xytext=(xi - 1.15, 0.365),
                      textcoords="data", fontsize=7.2, color="#555555",
                      ha="left", va="bottom",
                      arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.9))

axA.axvline((xs[-1] + xi_sel) / 2, color="#cccccc", linewidth=0.9, linestyle=":", zorder=1)

axA.plot(xi_sel, selected, marker="D", color=OKABE_ITO[3], markersize=6, zorder=4,
          markerfacecolor="white", markeredgecolor=OKABE_ITO[3], markeredgewidth=1.2)
axA.text(xi_sel, selected + 0.022, "test-selected (0.657)\nno stored interval",
          fontsize=7.2, color=OKABE_ITO[3], ha="center", va="bottom")

axA.axhline(0.5, color=CHANCE_GREY, linewidth=0.9, linestyle="--", zorder=1)
axA.text(0.988, 0.462, "chance", transform=axA.get_yaxis_transform(), fontsize=7.5,
          color=ANNOT_GREY, va="top", ha="right")
axA.set_xticks(xs + [xi_sel])
axA.set_xticklabels([r[0] for r in rows] + ["Relative,\nno gamma"])
axA.set_ylabel("Balanced accuracy")
axA.set_ylim(0.30, 0.80)
axA.set_xlim(-0.5, xi_sel + 0.5)
axA.grid(axis="x", visible=False)
axA.set_title("A  Declared feature-set variants", loc="left", fontsize=9.5)

# ---- Panel B ------------------------------------------------------------------
# Values are of order 1e-3; rescaled here (not left to matplotlib's automatic
# offset text, which is easy to miss) so the axis reads in plain 1-2 digit numbers.
SCALE = 1e3
freqs = np.array([f for f in prof["freqs"] if 2.0 <= f <= 14.0])
for key, ls, lab in (("all", "-", "All channels"), ("lateral", "--", "Lateral"),
                     ("midline", ":", "Midline")):
    axB.plot(freqs, np.array(prof["profile_2_14Hz"][key]) * SCALE, linestyle=ls, linewidth=1.4,
             color=ROLE_COLOR["relative_band_power"] if key == "all" else "#555555",
             label=lab, zorder=3)
axB.axhline(0, color=CHANCE_GREY, linewidth=0.9, linestyle="-", zorder=1)
# The 4.0-5.5 Hz shaded/named band was dropped here: no band-fishing is the working
# rule elsewhere in this chapter (differences between individual bands are not
# statistically distinguishable at N=35), and a shaded, named band on the figure
# itself contradicted that stance. The broad low-frequency profile is left to speak
# for itself without a highlighted sub-range.
axB.text(0.03, 0.94, "lateral and midline channels track each other closely:\n"
          "the elevation is broad, not localised to one montage",
          transform=axB.transAxes, fontsize=7.2, color="#555555", ha="left", va="top")

# Peak rug: values are an argmax within a hardcoded 3.5-7.5 Hz window, not the full
# 2-14 Hz range plotted here (checked against stage4d_spectral_profile.py directly,
# see module docstring). Stated on the panel, not left implicit, because the window
# edges account for 60% of the rug and are the actual finding, not a detail.
peaks = np.array(prof["per_patient_peak_hz"])
in_range = peaks[(peaks >= 2) & (peaks <= 14)]
n_lo_edge = int((peaks == 3.5).sum())
n_hi_edge = int((peaks == 7.5).sum())
ymin_data = min(np.array(prof["profile_2_14Hz"][k]).min() for k in ("all", "lateral", "midline")) * SCALE
axB.axvspan(3.5, 7.5, color="#f2f2f2", zorder=0)
axB.plot(in_range, np.full(len(in_range), ymin_data), marker="|", linestyle="none",
          color="#555555", markersize=7, markeredgewidth=1.0, alpha=0.55, zorder=4,
          clip_on=False)
axB.text(3.5, ymin_data, " 3.5-7.5 Hz peak-search window", fontsize=6.8, color=ANNOT_GREY,
          ha="left", va="bottom")
axB.annotate("%d/35 peaks pinned at the window edges\n(3.5 Hz: %d, 7.5 Hz: %d) -- broad effect,\n"
              "not a discrete peak, contra a tremor line"
              % (n_lo_edge + n_hi_edge, n_lo_edge, n_hi_edge),
              xy=(7.5, ymin_data), xycoords="data", xytext=(9.3, ymin_data * 0.55),
              textcoords="data", fontsize=6.8, color="#555555", ha="left", va="center",
              arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.8))

axB.set_xlabel("Frequency (Hz)")
axB.set_ylabel("OFF − ON relative power (×10⁻³)")
axB.set_xlim(2, 14)
axB.legend(loc="upper right")
axB.set_title("B  Spectral difference", loc="left", fontsize=9.5)

save_all(fig, HERE, "fig4-7")
print("saved fig4-7")
print("  nested %.4f ci=[%.4f,%.4f] n_res=%s" % (nested["aggregate"]["balanced_accuracy"],
      nested["bootstrap"]["ci_low"], nested["bootstrap"]["ci_high"],
      nested["bootstrap"]["n_valid_resamples"]))
print("  selected %.4f" % selected)
print("  peak median %.2f Hz sd %.2f, n patients %d, at 3.5Hz=%d at 7.5Hz=%d"
      % (prof["peak_median"], prof["peak_sd"], len(peaks), n_lo_edge, n_hi_edge))
