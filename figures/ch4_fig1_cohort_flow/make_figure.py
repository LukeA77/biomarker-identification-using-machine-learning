"""Fig 4-1 — Cohort flow, CONSORT-style.

Numbers traced to records, and they reconcile exactly (46 - 5 = 41; 41 - 6 = 35):

  46 participant folders on the study drive
      -> cohort-composition.md ("46 patient folders total, confirmed 2026-07-07")
  41-participant processed pool under the CSV-continuous pipeline
      -> cohort-composition.md ("reaching a 41-patient processed pool, locked to N=35")
     the 5 not reaching it: sub-82 and sub-100 (living-lab EEG never saved),
     sub-130 (one condition only), sub-111 (incompatible 13-channel montage),
     sub-78 (file-selection ambiguity confirmed a genuine exclude)
  35 primary cohort, 70 sessions, 37,174 windows, ON 20,122 / OFF 17,052
      -> cohort-patient-exclusions.md, cohort lock confirmed 2026-07-22;
         6 excluded on signal quality: sub-80, 97, 98, 99, 105, 108
  27 task-split subset, 11,744 windows
      -> within_patient_tasksplit_summary.json (27 eligible) and ch4_results.md §4.1

DELIBERATELY NOT USED: the older "18 qualify / 25 are a different DSI-only sub-study"
framing. cohort-composition.md records that framing as directly falsified by the
participants' own study notes and says not to reintroduce it. The 18-patient count
belongs to the superseded EDF-based rule, not to this pipeline.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, OKABE_ITO
from diagram import blank_ax, box, arrow

import matplotlib.pyplot as plt

apply_style()
HERE = pathlib.Path(__file__).parent

MAIN_FILL, MAIN_EDGE = "#f4f4f4", "#555555"
OUT_FILL, OUT_EDGE = "#ffffff", "#aaaaaa"
FINAL_FILL, FINAL_EDGE = "#fdf0e6", OKABE_ITO[1]

fig, ax = plt.subplots(figsize=(6.3, 4.1))
blank_ax(ax)

xc, w = 0.34, 0.48

levels = [
    (0.905, 0.115, "Participants on the study drive", "46 folders", MAIN_FILL, MAIN_EDGE),
    (0.660, 0.125, "Both medication states recorded\nand processable",
     "41 participants", MAIN_FILL, MAIN_EDGE),
    (0.395, 0.150, "Primary cohort", "N = 35 · 70 sessions\n37,174 windows "
     "(ON 20,122 / OFF 17,052)", FINAL_FILL, FINAL_EDGE),
    (0.130, 0.115, "Task-split subset", "N = 27 · 11,744 windows", MAIN_FILL, MAIN_EDGE),
]

for i, (y, h, label, sub, fill, edge) in enumerate(levels):
    box(ax, xc, y, w, h, label, sub=sub, fill=fill, edge=edge, fontsize=8, subsize=6.8)
    if i < len(levels) - 1:
        ny, nh = levels[i + 1][0], levels[i + 1][1]
        arrow(ax, (xc, y - h / 2), (xc, ny + nh / 2))

excl = [
    (0.785, 0.180, "5 not processable:\nsub-82, sub-100 (EEG never saved)\n"
                   "sub-130 (one condition only)\nsub-111 (incompatible montage)\n"
                   "sub-78 (file-selection ambiguity)"),
    (0.530, 0.080, "6 excluded on signal quality:\nsub-80, 97, 98, 99, 105, 108"),
    (0.262, 0.080, "8 without both kitchen tasks\nrecoverable in both blocks"),
]
for y, h, text in excl:
    ax.plot([xc, xc + 0.285], [y, y], color="#aaaaaa", linewidth=0.8, zorder=1)
    arrow(ax, (xc + 0.285, y), (xc + 0.312, y), color="#aaaaaa", lw=0.8)
    box(ax, xc + 0.485, y, 0.31, h, "", fill=OUT_FILL, edge=OUT_EDGE)
    ax.text(xc + 0.485, y, text, ha="center", va="center", fontsize=6.5,
            color="#555555", zorder=5)

ax.text(xc, 0.030, "sub-42, sub-70 and sub-121 enter the pool only after a "
                   "file-selection defect was corrected",
        ha="center", va="bottom", fontsize=6.5, color=OKABE_ITO[1])

save_all(fig, HERE, "fig4-1")
print("saved fig4-1  (46 - 5 = 41; 41 - 6 = 35; 35 - 8 = 27)")
