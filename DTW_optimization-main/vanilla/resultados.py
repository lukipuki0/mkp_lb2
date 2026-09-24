"""
Script maestro de resultados — Vanilla.
Corre todas las MHs sin DTW, muestra métricas en consola, guarda JSON y genera gráficos comparativos.

Uso:
    python -m vanilla.resultados
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from mkp_common import BinaryPSO, GeneticAlgorithm, BinaryGWO, BinaryDE, cargar_instancia
from mkp_common.results import save_results

from .runner import run_epochs
from .config import (
    INDICE_INSTANCIA,
    NUM_ITERACIONES,
    NUM_PARTICULAS,
    RUTA_INSTANCIA,
    EPOCHS,
    SEMILLA,
)

MHS = {
    "PSO": BinaryPSO,
    "GA": GeneticAlgorithm,
    "GWO": BinaryGWO,
    "DE": BinaryDE,
}

# Colores por MH (consistentes con las otras versiones)
MH_COLORS = {
    "PSO": "#2196F3",
    "BinaryPSO": "#2196F3",
    "GA": "#E91E63",
    "GWO": "#4CAF50",
    "DE": "#FF9800",
}

COLORS = {
    "optimum": "#7f8c8d",
}


def print_epoch_results(nombre: str, resultados: list, inst: dict):
    optimo = inst["optimo"]
    print(f"\n{'=' * 70}")
    print(f"  {nombre} (Vanilla)")
    print(f"{'=' * 70}")

    for res in resultados:
        gap = f"Gap={100 - res['ganancia']:.2f}%" if optimo > 0 else "Gap=N/A"
        print(
            f"  Epoch {res['epoch']:02d} | "
            f"Fitness={res['mejor_fitness']:.1f} | {gap} | "
            f"t={res['tiempo']:.2f}s"
        )

    fits = [r["mejor_fitness"] for r in resultados]
    print(f"\n  {'-' * 50}")
    print(f"  Mejor={np.max(fits):.1f}  Prom={np.mean(fits):.1f}  "
          f"Peor={np.min(fits):.1f}  Std={np.std(fits):.1f}")
    if optimo > 0:
        print(f"  Optimo={optimo:.0f}  Gap={100 - np.max(fits) / optimo * 100:.2f}%")


def generate_plots(resultados_por_mh: dict, inst: dict, save_dir: str):
    nombres = list(resultados_por_mh.keys())

    # Seleccionar el mejor epoch de cada MH
    mejores = {}
    for nombre in nombres:
        resultados = resultados_por_mh[nombre]
        idx = int(np.argmax([r["mejor_fitness"] for r in resultados]))
        mejores[nombre] = resultados[idx]

    optimo = inst["optimo"]

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

    fig, ax = plt.subplots(figsize=(10, 5))

    for nombre in nombres:
        res = mejores[nombre]
        hist_fit = res["historial_fitness"]
        color = MH_COLORS.get(nombre, "#666666")

        iters = range(len(hist_fit))
        ax.plot(
            iters, hist_fit,
            color=color, linewidth=1.8, linestyle="-",
            label=f"{nombre} (max={res['mejor_fitness']:.1f})",
        )

    if optimo > 0:
        ax.axhline(
            y=optimo, color=COLORS["optimum"], linestyle="--",
            linewidth=1, alpha=0.7, label=f"Known optimum ({optimo})",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Vanilla (exploit por defecto) — All MHs (seed={SEMILLA})")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle=":")

    plt.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    inst_name = Path(RUTA_INSTANCIA).stem
    for ext in ("png", "pdf"):
        path = f"{save_dir}/vanilla_{inst_name}_{INDICE_INSTANCIA}.{ext}"
        fig.savefig(path)
        print(f"  Saved: {path}")

    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)


def main():
    inst = cargar_instancia(RUTA_INSTANCIA, idx=INDICE_INSTANCIA)

    print("=" * 70)
    print("  RESULTADOS MAESTRO — Vanilla (Sin DTW)")
    print("=" * 70)
    print(f"  Instancia: {RUTA_INSTANCIA}[{INDICE_INSTANCIA}]")
    print(f"  n={inst['n']}, m={inst['m']}")
    print(f"  Poblacion: {NUM_PARTICULAS}, Iteraciones: {NUM_ITERACIONES}, Epochs: {EPOCHS}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    inst_name = Path(RUTA_INSTANCIA).stem
    save_dir = f"results/vanilla/todos/{inst_name}_{INDICE_INSTANCIA}/comparacion_mhs_{run_id}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    resultados_por_mh = {}

    for nombre, mh_class in MHS.items():
        print(f"\n  Ejecutando {nombre} Vanilla...")
        resultados = run_epochs(
            mh_class=mh_class,
            inst=inst,
            num_particulas=NUM_PARTICULAS,
            num_iteraciones=NUM_ITERACIONES,
            epochs=EPOCHS,
        )
        resultados_por_mh[nombre] = resultados
        print_epoch_results(nombre, resultados, inst)

        inst_name = Path(RUTA_INSTANCIA).stem
        save_results(
            resultados,
            path=f"{save_dir}/{nombre}_{inst_name}_{INDICE_INSTANCIA}.json",
            mh_name=nombre,
            optimo_conocido=inst["optimo"] if inst["optimo"] > 0 else None,
            extra_info={
                "estrategia": "vanilla",
                "instancia": RUTA_INSTANCIA,
                "idx": INDICE_INSTANCIA,
                "poblacion": NUM_PARTICULAS,
                "iteraciones": NUM_ITERACIONES,
            },
        )

    generate_plots(resultados_por_mh, inst, save_dir)


if __name__ == "__main__":
    main()
