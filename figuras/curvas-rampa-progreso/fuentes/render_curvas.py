"""Render the illustrative ramp/progress/plateau curves of the conference slides.

The curves reproduce the TikZ/pgfplots schematics in
overleaf/presentacion conferencia/main.tex as standalone PNG and PDF figures.
They are illustrative patterns, not experimental results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]

PRIMARY = (15 / 255, 76 / 255, 110 / 255)
ACCENT = (224 / 255, 122 / 255, 40 / 255)
BEST = (0 / 255, 128 / 255, 64 / 255)
WARN = (183 / 255, 65 / 255, 14 / 255)
INK = (55 / 255, 55 / 255, 60 / 255)

plt.rcParams.update({
    "font.size": 13,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.4,
})


def clean_axis(ax, xlabel, ylabel):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def curva_progreso_estancamiento():
    """Slide 'Problema': convergence curve with progress and stagnation zones."""
    x = np.linspace(0, 100, 300)
    y = 10 + 78 * (1 - np.exp(-x / 9))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axvspan(0, 35, color=BEST, alpha=0.12, lw=0)
    ax.axvspan(35, 100, color=WARN, alpha=0.10, lw=0)
    ax.plot(x, y, color=PRIMARY, lw=3)
    ax.text(17.5, 32, "progreso", color=BEST, weight="bold", ha="center")
    ax.text(17.5, 20, "mejora\nrápido", color=BEST, ha="center", va="center", fontsize=11)
    ax.text(67, 52, "estancamiento", color=WARN, weight="bold", ha="center")
    ax.text(67, 42, "deja de mejorar", color=WARN, ha="center", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    clean_axis(ax, "Iteración", "Mejor resultado encontrado")
    save(fig, "curva_progreso_estancamiento")


def patrones_rampa_meseta():
    """Slide 'Dos patrones de comparación': ramp (progress) vs plateau."""
    k = np.arange(10)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    panels = [
        (axes[0], 1 + 0.9 * k, BEST, "Progreso (rampa)",
         "El mejor resultado sigue mejorando"),
        (axes[1], np.full_like(k, 6.0, dtype=float), WARN, "Estancamiento (meseta)",
         "El mejor resultado se mantiene igual"),
    ]
    for ax, y, color, title, subtitle in panels:
        ax.plot(k, y, color=color, lw=3, marker="o", ms=6)
        ax.set_title(f"{title}\n", color=color, weight="bold")
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
                fontsize=10, color=INK)
        ax.set_xlim(-0.5, 10)
        ax.set_ylim(0, 10.5)
        clean_axis(ax, "Posición en la ventana", "Mejor resultado")
    axes[1].set_ylabel("")
    save(fig, "patrones_rampa_meseta")


def signo_delta():
    """Slide 'El signo de la diferencia': window X between ramp R and plateau C."""
    k = np.arange(1, 11)
    ramp = 2 + 0.8 * (k - 1)
    plateau = np.full_like(k, 2.0, dtype=float)
    window = np.array([2, 2.7, 3.1, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3])

    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.plot(k, ramp, color=BEST, lw=3, ls="--")
    ax.plot(k, plateau, color=WARN, lw=3, ls="--")
    ax.plot(k, window, color=PRIMARY, lw=3, marker="o", ms=5)
    ax.text(10.15, ramp[-1], "$R$ rampa", color=BEST, weight="bold", va="center")
    ax.text(10.15, 3.55, "$X$ ventana", color=PRIMARY, weight="bold", va="center")
    ax.text(10.15, 1.75, "$C$ meseta", color=WARN, weight="bold", va="center")

    arrow = dict(arrowstyle="<->", lw=2, shrinkA=0, shrinkB=0)
    x1 = 6.5
    ax.annotate("", xy=(x1, 3.35), xytext=(x1, 2 + 0.8 * (x1 - 1)),
                arrowprops={**arrow, "color": BEST})
    ax.text(x1 + 0.15, 5.0, "$D_1$ grande", color=BEST, fontsize=12)
    x2 = 8.5
    ax.annotate("", xy=(x2, 2.03), xytext=(x2, 3.27),
                arrowprops={**arrow, "color": WARN})
    ax.text(x2 - 0.15, 2.65, "$D_2$ pequeña", color=WARN, fontsize=12,
            ha="right", va="center")

    ax.set_xlim(0.5, 10)
    ax.set_ylim(0, 10)
    clean_axis(ax, "Posición en la ventana ($k$)", "Mejor resultado")
    save(fig, "signo_delta_rampa_meseta")


if __name__ == "__main__":
    curva_progreso_estancamiento()
    patrones_rampa_meseta()
    signo_delta()
    print("figures written to", OUT)
