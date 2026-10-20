"""Shared style for every dissertation figure. Import apply_style() and call it first.

Fixed spec — do not override per-figure except figsize height. Width is always 6.3in
full-width or 3.1in for side-by-side panels.

Note on bbox: savefig.bbox is deliberately NOT "tight". Tight bbox trims whitespace and
the exported width then no longer equals the requested figsize, which breaks the
"6.3in at export, never resized in Word" requirement. constrained_layout keeps labels
inside the canvas instead.
"""
import matplotlib as mpl

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

# Fixed role -> colour assignment, held constant across every figure it recurs in.
ROLE_COLOR = {
    "frozen_probe":        OKABE_ITO[0],  # blue
    "frozen_mean_pool":    OKABE_ITO[0],
    "frozen_flatten":      OKABE_ITO[5],  # sky blue
    "lora":                OKABE_ITO[1],  # orange — pre-registered primary arm
    "full_ft":             OKABE_ITO[2],  # bluish green
    "lora_global_z":       OKABE_ITO[4],  # reddish purple
    "chanamp":             OKABE_ITO[3],  # vermillion — identity fingerprint control
    "chanamp_control":     OKABE_ITO[3],
    "relative_band_power": OKABE_ITO[2],
    "duration_heuristic":  OKABE_ITO[7],  # black
    "elapsed_heuristic":   "#666666",
}

# Marker shapes, so colour is never the sole discriminator (greyscale-safe).
ROLE_MARKER = {
    "frozen_probe": "o", "frozen_mean_pool": "o", "frozen_flatten": "s",
    "lora": "D", "full_ft": "^", "lora_global_z": "v",
    "chanamp": "X", "chanamp_control": "X",
    "relative_band_power": "^",
    "duration_heuristic": "P", "elapsed_heuristic": "*",
}

CHANCE_GREY = "#999999"
NULL_BAND_GREY = "#e8e8e8"
ANNOT_GREY = "#666666"


def apply_style():
    mpl.rcParams.update({
        "figure.figsize": (6.3, 3.9),
        "figure.dpi": 150, "savefig.dpi": 600,
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def chance_line(ax, y=0.5, label="chance", xpos=0.012, va="bottom"):
    """Light dashed grey horizontal reference at chance, drawn behind the data."""
    ax.axhline(y, color=CHANCE_GREY, linewidth=0.9, linestyle="--", zorder=1)
    ax.text(xpos, y + 0.004, label, transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=ANNOT_GREY, va=va, ha="left", zorder=2)


def save_all(fig, out_dir, basename, greyscale_check=True):
    """Save PNG at 600 dpi and SVG; optionally write a greyscale proof alongside."""
    import pathlib
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{basename}.png"
    svg_path = out_dir / f"{basename}.svg"
    fig.savefig(png_path, format="png")
    fig.savefig(svg_path, format="svg")
    if greyscale_check:
        try:
            from PIL import Image
            Image.open(png_path).convert("L").save(out_dir / f"{basename}_greyscale.png")
        except Exception:
            pass
    return png_path, svg_path
