"""
Análisis individual de una MH Vanilla (sin DTW).
Ejecuta 1 corrida, muestra convergencia en consola y genera el plot.

Uso:
    python -m vanilla.run
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mkp_common import cargar_instancia

from .config import (
    INDICE_INSTANCIA,
    MH_CLASS,
    NUM_ITERACIONES,
    NUM_PARTICULAS,
    RUTA_INSTANCIA,
    SEMILLA,
)
from .runner import run_experiment

# Colores estilo paper
COLORS = {
    "fitness": "#2c3e50",
    "optimum": "#7f8c8d",
}


def print_iteration_table(res: dict, inst: dict) -> None:
    hist_fit = res["historial_fitness"]
    optimo = inst["optimo"]

    hdr = f"{'it':>4} | {'fitness':>12} | {'mode':>8}"
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print(
        f"  Vanilla Log — {MH_CLASS.__name__} | "
        f"seed={res.get('semilla', '?')} | "
        f"instance={Path(RUTA_INSTANCIA).stem}[{INDICE_INSTANCIA}]"
    )
    if optimo > 0:
        print(f"  Optimo conocido: {optimo}")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)

    for i, fitness in enumerate(hist_fit):
        print(f"{i:4d} | {fitness:12.1f} | EXPLOIT")

    print(sep)
    print(f"  Fitness final: {hist_fit[-1]:.1f}")
    if optimo > 0:
        gap = 100 - hist_fit[-1] / optimo * 100
        print(f"  Gap al optimo: {gap:.2f}%")
    print()


def plot_paper(res: dict, inst: dict, save_dir: str = "results/vanilla"):
    hist_fit = res["historial_fitness"]
    optimo = inst["optimo"]
    mh_name = MH_CLASS.__name__

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(10, 4.5))

    iters = np.arange(len(hist_fit))
    ax.plot(
        iters, hist_fit, color=COLORS["fitness"],
        linewidth=1.8, label="Best fitness (Vanilla)", zorder=3,
    )

    if optimo > 0:
        ax.axhline(
            y=optimo, color=COLORS["optimum"], linestyle="--",
            linewidth=1, alpha=0.7, label=f"Known optimum ({optimo})",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title(
        f"{mh_name} Vanilla — Convergence (seed={res.get('semilla', '?')})"
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle=":")

    plt.tight_layout()

    mh_dir = f"{save_dir}/{mh_name}"
    Path(mh_dir).mkdir(parents=True, exist_ok=True)
    run_id = res.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    inst_name = Path(RUTA_INSTANCIA).stem
    for ext in ("png", "pdf"):
        path = f"{mh_dir}/{mh_name}_{inst_name}_{INDICE_INSTANCIA}_{run_id}.{ext}"
        fig.savefig(path)
        print(f"  Saved: {path}")

    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)


def main():
    inst = cargar_instancia(RUTA_INSTANCIA, idx=INDICE_INSTANCIA)
    mh_name = MH_CLASS.__name__
    archivo = Path(RUTA_INSTANCIA).stem

    print(f"\n  Vanilla MH (No DTW): {mh_name}")
    print(f"  Instance: {archivo}[{INDICE_INSTANCIA}]"
          f" — n={inst['n']}, m={inst['m']}")
    print(f"  Pop: {NUM_PARTICULAS}, Iters: {NUM_ITERACIONES}")

    res = run_experiment(
        mh_class=MH_CLASS,
        inst=inst,
        num_particulas=NUM_PARTICULAS,
        num_iteraciones=NUM_ITERACIONES,
        semilla=SEMILLA,
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    res["semilla"] = SEMILLA
    res["run_id"] = run_id

    print_iteration_table(res, inst)
    plot_paper(res, inst)


if __name__ == "__main__":
    main()
