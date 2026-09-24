"""
Continuous-Complex — intensidad continua sigmoide desde delta.
Corre todas las MHs, muestra métricas en consola, genera gráficos comparativos.

Uso (desde la raíz del proyecto):
    python -m continuous_complex.resultados
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
    DTW_CFG,
    INDICE_INSTANCIA,
    NUM_ITERACIONES,
    NUM_PARTICULAS,
    RUTA_INSTANCIA,
    EPOCHS,
    SEMILLA,
    K,
    CENTER,
)

# MHs a comparar
MHS = {
    "PSO": BinaryPSO,
    "GA": GeneticAlgorithm,
    "GWO": BinaryGWO,
    "DE": BinaryDE,
}

# Colores
MH_COLORS = {
    "PSO": "#2196F3",
    "BinaryPSO": "#2196F3",
    "GA": "#E91E63",
    "GWO": "#4CAF50",
    "DE": "#FF9800",
}


# =============================================================================
# CONSOLA
# =============================================================================


def print_epoch_results(nombre: str, resultados: list, inst: dict):
    optimo = inst["optimo"]

    print(f"\n{'=' * 70}")
    print(f"  {nombre}")
    print(f"{'=' * 70}")

    for res in resultados:
        gap = f"Gap={100 - res['ganancia']:.2f}%" if optimo > 0 else "Gap=N/A"
        print(
            f"\n  Epoch {res['epoch']:02d} | "
            f"Fitness={res['mejor_fitness']:.1f} | "
            f"IntAvg={res['intensity_promedio']:.3f} | {gap} | "
            f"t={res['tiempo']:.2f}s"
        )

        ready = [h for h in res["historial_dtw"] if h.get("ready")]
        if ready:
            last = ready[-1]
            d1s = [h["D1_vs_ramp"] for h in ready]
            d2s = [h["D2_vs_const"] for h in ready]
            deltas = [h["delta"] for h in ready]
            print(
                f"    DTW  D1={np.mean(d1s):7.1f} (+-{np.std(d1s):.1f}) | "
                f"D2={np.mean(d2s):7.1f} (+-{np.std(d2s):.1f}) | "
                f"delta={np.mean(deltas):+7.1f} | "
                f"th_c={last['theta_c']:.1f}  th_r={last['theta_r']:.1f}  "
                f"th_d={last['theta_delta']:.1f}"
            )

    fits = [r["mejor_fitness"] for r in resultados]
    print(f"\n  {'-' * 50}")
    print(f"  Mejor={np.max(fits):.1f}  Prom={np.mean(fits):.1f}  "
          f"Peor={np.min(fits):.1f}  Std={np.std(fits):.1f}")
    if optimo > 0:
        print(f"  Optimo={optimo:.0f}  Gap={100 - np.max(fits) / optimo * 100:.2f}%")


# =============================================================================
# PLOTS
# =============================================================================


def _get_explore_ranges(hist_mode):
    ranges = []
    in_explore = False
    start = 0
    for i, m in enumerate(hist_mode):
        if m == "explore" and not in_explore:
            start = i
            in_explore = True
        elif m != "explore" and in_explore:
            ranges.append((start, i))
            in_explore = False
    if in_explore:
        ranges.append((start, len(hist_mode) - 1))
    return ranges


def generate_plots(resultados_por_mh: dict, inst: dict, save_dir: str):
    """
    2 paneles:
      1. Fitness superpuesto con líneas punteadas en explore
      2. Intensidad (curva sigmoid) de cada MH
    """
    nombres = list(resultados_por_mh.keys())

    # Mejor epoch de cada MH
    mejores = {}
    for nombre in nombres:
        resultados = resultados_por_mh[nombre]
        idx = int(np.argmax([r["mejor_fitness"] for r in resultados]))
        mejores[nombre] = resultados[idx]

    optimo = inst["optimo"]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )

    # ─── Panel 1: Fitness ────────────────────────────────────────────────
    for nombre in nombres:
        res = mejores[nombre]
        hist_fit = res["historial_fitness"]
        hist_mode = res["historial_modos"]
        color = MH_COLORS.get(nombre, "#666666")

        explore_ranges = _get_explore_ranges(hist_mode)

        # Segmentos sólido/punteado según modo
        is_explore = [False] * len(hist_fit)
        for s, e in explore_ranges:
            for j in range(s, min(e + 1, len(hist_fit))):
                is_explore[j] = True

        prev_type = is_explore[0]
        seg_start = 0
        for k in range(1, len(hist_fit)):
            if is_explore[k] != prev_type:
                style = "--" if prev_type else "-"
                ax1.plot(range(seg_start, k + 1), hist_fit[seg_start:k + 1],
                        color=color, linewidth=1.5, linestyle=style, zorder=3)
                seg_start = k
                prev_type = is_explore[k]
        style = "--" if prev_type else "-"
        ax1.plot(range(seg_start, len(hist_fit)), hist_fit[seg_start:],
                color=color, linewidth=1.5, linestyle=style, zorder=3)

        ax1.plot([], [], color=color, linewidth=1.5, linestyle="-",
                label=f"{nombre} (max={res['mejor_fitness']:.0f}, int={res['intensity_promedio']:.3f})")

    if optimo > 0:
        ax1.axhline(y=optimo, color="#7f8c8d", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"Known optimum ({optimo})")

    ax1.plot([], [], color="gray", linewidth=1.5, linestyle="-", label="Exploit")
    ax1.plot([], [], color="gray", linewidth=1.5, linestyle="--", label="Explore")
    ax1.set_ylabel("Fitness")
    ax1.set_title(f"Continuous-Complex — All MHs (seed={SEMILLA})")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle=":")

    # ─── Panel 2: Intensity ──────────────────────────────────────────────
    for mh_idx, nombre in enumerate(nombres):
        res = mejores[nombre]
        hist_dtw = res["historial_dtw"]
        hist_int = res["historial_intensity"]
        color = MH_COLORS.get(nombre, "#666666")

        ready_iters, intensities = [], []
        for i, h in enumerate(hist_dtw):
            if h.get("ready"):
                ready_iters.append(i)
                intensities.append(hist_int[i])

        if ready_iters:
            ax2.plot(ready_iters, intensities, color=color,
                    linewidth=1.5, label=f"{nombre}")

    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold (0.5)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Intensity")
    ax2.set_title(r"DTW Stagnation Signal — Intensity (Continuous-Complex sigmoid)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax2.grid(True, alpha=0.2, linestyle=":")

    plt.tight_layout(h_pad=1.5)

    # Guardar
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    inst_name = Path(RUTA_INSTANCIA).stem
    for ext in ("png", "pdf"):
        path = f"{save_dir}/continuous_complex_{inst_name}_{INDICE_INSTANCIA}.{ext}"
        fig.savefig(path)
        print(f"  Saved: {path}")

    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def main():
    inst = cargar_instancia(RUTA_INSTANCIA, idx=INDICE_INSTANCIA)

    print("=" * 70)
    print("  RESULTADOS MAESTRO — Continuous-Complex")
    print("=" * 70)
    print(f"  Instancia: {RUTA_INSTANCIA}[{INDICE_INSTANCIA}]")
    print(f"  n={inst['n']}, m={inst['m']}")
    print(f"  Particulas/Pop: {NUM_PARTICULAS}, Iteraciones: {NUM_ITERACIONES}, "
          f"Epochs: {EPOCHS}")
    print(f"  DTW: window={DTW_CFG.window}, ddtw={DTW_CFG.use_ddtw}")
    print(f"  Continuous-Complex: k={K}, center={CENTER}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    inst_name = Path(RUTA_INSTANCIA).stem
    save_dir = f"results/continuous_complex/todos/{inst_name}_{INDICE_INSTANCIA}/comparacion_mhs_{run_id}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    resultados_por_mh = {}

    for nombre, mh_class in MHS.items():
        print(f"\n  Ejecutando {nombre}...")
        resultados = run_epochs(
            mh_class=mh_class, inst=inst, monitor_cfg=DTW_CFG,
            num_particulas=NUM_PARTICULAS, num_iteraciones=NUM_ITERACIONES,
            epochs=EPOCHS, verbose=False,
            k=K, center=CENTER,
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
                "estrategia": "continuous_complex",
                "k": K, "center": CENTER,
                "instancia": RUTA_INSTANCIA, "idx": INDICE_INSTANCIA,
                "poblacion": NUM_PARTICULAS, "iteraciones": NUM_ITERACIONES,
                "dtw_window": DTW_CFG.window,
            },
        )

    generate_plots(resultados_por_mh, inst, save_dir)


if __name__ == "__main__":
    main()
