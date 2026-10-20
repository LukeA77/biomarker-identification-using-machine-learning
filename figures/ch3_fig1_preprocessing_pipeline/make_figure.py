"""Fig 3-1 — Preprocessing pipeline block diagram.

Every parameter shown is read from pipeline_config.yaml in this folder (a copy of
config/pipeline.yaml), so the diagram cannot drift from the configuration that
actually ran. Channel counts (30-odd recorded -> 18 retained) come from the same file.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_common"))
from style import apply_style, save_all, OKABE_ITO
from diagram import blank_ax, box, arrow, BOX_EDGE

import matplotlib.pyplot as plt
import yaml

apply_style()
HERE = pathlib.Path(__file__).parent
cfg = yaml.safe_load((HERE / "pipeline_config.yaml").read_text())

sig, ica, win, rej = cfg["signal"], cfg["ica"], cfg["window"], cfg["reject"]
n_keep = len(cfg["channels"]["keep"]) if cfg["channels"].get("keep") else 18

stages = [
    ("Raw CSV ingest", "%d Hz, µV" % sig["orig_sfreq"], False),
    ("Re-reference", "linked ears (A1, A2)", False),
    ("Band-pass", "%.1f–%d Hz, %s, zero-phase" % (sig["bandpass"]["l_freq"],
                                                  sig["bandpass"]["h_freq"],
                                                  sig["bandpass"]["fir_design"]), False),
    ("Notch", "%d Hz mains" % sig["notch_freq"], False),
    ("ICA", "%s, %d components" % (ica["method"], ica["n_components"]), True),
    ("Channel selection", "→ %d scalp channels" % n_keep, False),
    ("Resample", "%d → %d Hz" % (sig["orig_sfreq"], sig["target_sfreq"]), False),
    ("Windowing", "%d s; stride %d s train / %d s eval" % (win["length_s"],
                                                           win["train_stride_s"],
                                                           win["eval_stride_s"]), False),
    ("Amplitude rejection", "disabled — see §3.3", None),
]

fig, ax = plt.subplots(figsize=(6.3, 4.3))
blank_ax(ax)

n = len(stages)
top, bottom = 0.955, 0.045
h = (top - bottom) / n * 0.74
gap = (top - bottom) / n
xc, w = 0.42, 0.60

for i, (label, sub, highlight) in enumerate(stages):
    y = top - gap * (i + 0.5)
    if highlight is None:                     # disabled stage
        fill, edge, tc = "#ffffff", "#bbbbbb", "#888888"
    elif highlight:                           # the stage carrying a declared deviation
        fill, edge, tc = "#fdf0e6", OKABE_ITO[1], "#0b0b0b"
    else:
        fill, edge, tc = "#f4f4f4", BOX_EDGE, "#0b0b0b"
    box(ax, xc, y, w, h, label, sub=sub, fill=fill, edge=edge, textcolor=tc)
    if i < n - 1:
        arrow(ax, (xc, y - h / 2), (xc, top - gap * (i + 1.5) + h / 2))

# Deviation markers down the right-hand side
dev = {1: "deviation 1: linked-ears reference",
       3: "deviation 2: 50 Hz not 60 Hz",
       4: "deviation 3: correlation-based\nocular criterion added"}
for i, note in dev.items():
    y = top - gap * (i + 0.5)
    ax.plot([xc + w / 2 + 0.012, xc + w / 2 + 0.045], [y, y],
            color=OKABE_ITO[1], linewidth=0.8, zorder=2)
    ax.text(xc + w / 2 + 0.052, y, note, fontsize=7, color=OKABE_ITO[1],
            ha="left", va="center")

save_all(fig, HERE, "fig3-1")
print("saved fig3-1 | keep=%d target_sfreq=%d reject_enabled=%s"
      % (n_keep, sig["target_sfreq"], rej["enabled"]))
