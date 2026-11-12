"""Render the HRES2 energy-flow figure to PDF and PNG."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
PAGE_W, PAGE_H = 1100, 420


def box(ax, x, y, w, h, fill, stroke, lines, italic=None):
    top = PAGE_H - y - h
    patch = FancyBboxPatch(
        (x, top),
        w,
        h,
        boxstyle="round,pad=0.2,rounding_size=12",
        linewidth=1.5,
        facecolor=fill,
        edgecolor=stroke,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    cx, cy = x + w / 2, top + h / 2
    if len(lines) == 1:
        ax.text(cx, cy, lines[0], ha="center", va="center", fontsize=11, color="#1a1a1a")
        return
    ax.text(cx, cy + 11, lines[0], ha="center", va="center", fontsize=11, color="#1a1a1a")
    style = "italic" if italic else "normal"
    ax.text(cx, cy - 12, lines[1], ha="center", va="center", fontsize=10.5, color="#1a1a1a", style=style)


def arrow(ax, x1, y1, x2, y2, color="#666666", width=1.5):
    ax.add_patch(
        FancyArrowPatch(
            (x1, PAGE_H - y1),
            (x2, PAGE_H - y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=width,
            color=color,
            connectionstyle="arc3,rad=0",
            shrinkA=0,
            shrinkB=0,
        )
    )


def elbow(ax, points, color="#666666", width=1.5):
    xy = [(x, PAGE_H - y) for x, y in points]
    for (x1, y1), (x2, y2) in zip(xy, xy[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=width, solid_capstyle="round")
    arrow(ax, points[-2][0], points[-2][1], points[-1][0], points[-1][1], color, width)


def main():
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200)
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    box(ax, 30, 36, 210, 70, "#dae8fc", "#6c8ebf", ["Eólica", r"$P_{WT}$"], italic=True)
    box(ax, 30, 214, 210, 70, "#dae8fc", "#6c8ebf", ["Solar PV", r"$P_{PV}=200-P_{WT}$"], italic=True)
    box(ax, 320, 118, 190, 76, "#fde9d9", "#e0a080", ["Generación", "renovable"])
    box(ax, 590, 20, 210, 70, "#ffe6cc", "#d79b00", ["BESS", "almacena excedentes"])
    box(ax, 315, 272, 200, 68, "#ffe6cc", "#d79b00", ["Red", "vende excedentes"])
    box(ax, 590, 154, 210, 76, "#ffe6cc", "#d79b00", ["Electrolizador PEM", r"$P_{el}=5N_{el}$"], italic=True)
    box(ax, 870, 156, 170, 72, "#d5e8d4", "#82b366", ["H$_2$ verde", "producto final"])

    elbow(ax, [(240, 71), (300, 71), (300, 139), (320, 139)])
    elbow(ax, [(240, 249), (300, 249), (300, 173), (320, 173)])
    elbow(ax, [(457, 118), (457, 55), (590, 55)])
    elbow(ax, [(415, 194), (415, 272)])
    elbow(ax, [(510, 156), (550, 156), (550, 192), (590, 192)])
    elbow(ax, [(695, 90), (695, 154)])
    elbow(ax, [(800, 192), (870, 192)], color="#2D7600", width=2.4)

    fig.savefig(OUT / "hres2_flujo_energetico.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / "hres2_flujo_energetico.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
