"""Small helpers for the hand-built schematic figures (Fig 3-1, 3-2, 3-3, 4-1).

These diagrams carry no data, so they are drawn on a blank axis in figure-relative
coordinates. Kept here so all four share one box/arrow vocabulary.
"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BOX_EDGE = "#555555"
BOX_FILL = "#f4f4f4"
TEXT = "#0b0b0b"
SUBTEXT = "#555555"


def blank_ax(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)
    return ax


def box(ax, x, y, w, h, label, sub=None, fill=BOX_FILL, edge=BOX_EDGE,
        fontsize=8, subsize=7, lw=0.9, textcolor=TEXT, zorder=3):
    """Rounded box centred on (x, y) in axis coordinates."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0,rounding_size=0.012",
                       linewidth=lw, edgecolor=edge, facecolor=fill, zorder=zorder)
    ax.add_patch(p)
    if sub:
        ax.text(x, y + h * 0.14, label, ha="center", va="center",
                fontsize=fontsize, color=textcolor, zorder=zorder + 1)
        ax.text(x, y - h * 0.22, sub, ha="center", va="center",
                fontsize=subsize, color=SUBTEXT, zorder=zorder + 1)
    else:
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                color=textcolor, zorder=zorder + 1)
    return p


def arrow(ax, xy_from, xy_to, color=BOX_EDGE, lw=0.9, style="->", zorder=2,
          connectionstyle=None):
    a = FancyArrowPatch(xy_from, xy_to, arrowstyle=style, color=color,
                        linewidth=lw, mutation_scale=8, zorder=zorder,
                        shrinkA=0, shrinkB=0,
                        connectionstyle=connectionstyle or "arc3,rad=0")
    ax.add_patch(a)
    return a
