"""Text-free versions of the DTW and ramp/plateau window figures of the conference slides.

No titles, labels, ticks or legends: only the drawing, so text can be placed
in Beamer. Curves are illustrative, not experimental results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]

PRIMARY = (15 / 255, 76 / 255, 110 / 255)
ACCENT = (224 / 255, 122 / 255, 40 / 255)
BEST = (0 / 255, 128 / 255, 64 / 255)
WARN = (183 / 255, 65 / 255, 14 / 255)
WINDOW = "#E32626"
RAMP = "#21B04B"
PLATEAU = "#F5862E"


def upper(t):
    return 1.7 + 1.2 / (1 + np.exp(-1.4 * (t - 3.5)))


def lower(t):
    return 1.2 / (1 + np.exp(-1.4 * (t - 6.5)))


def bare(ax):
    ax.set_axis_off()
    ax.margins(0.02)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


def dtw_panel(name, pairs, color):
    t = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for i, j in pairs:
        ax.plot([i, j], [upper(i), lower(j)], ls=(0, (4, 3)), lw=1.4, color=color)
    ax.plot(t, upper(t), color=PRIMARY, lw=3.2, solid_capstyle="round")
    ax.plot(t, lower(t), color=ACCENT, lw=3.2, solid_capstyle="round")
    bare(ax)
    save(fig, name)


def ventana_rampa_meseta():
    steps = [(1, 5), (3, 50), (5, 57), (6, 58), (7, 82), (17, 84), (18, 96), (31, 99),
             (33, 103), (35, 103.5), (72, 107), (101, 119), (270, 120), (302, 131)]
    x = [s[0] for s in steps] + [350]
    y = [s[1] for s in steps] + [steps[-1][1]]
    start, end, base = 30, 170, 96

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(x, y, where="post", color="black", lw=2.4)
    for xv in (start, end):
        ax.axvline(xv, ymin=0, ymax=0.92, color=WINDOW, lw=2.6)
    ax.plot([start + 1, end - 1], [base + 2, 127], color=RAMP, lw=3.2)
    ax.plot([start + 1, end - 1], [base, base], color=PLATEAU, lw=3.2)

    ax.set_xlim(0, 355)
    ax.set_ylim(0, 150)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(2.4)
    save(fig, "ventana_rampa_meseta")


if __name__ == "__main__":
    t = range(11)
    dtw_panel("dtw_punto_a_punto", [(i, i) for i in t], WARN)
    dtw_panel("dtw_alineacion",
              [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
               (5, 8), (6, 9), (7, 10), (8, 10), (9, 10), (10, 10)], BEST)
    ventana_rampa_meseta()
    print("figures written to", OUT)
