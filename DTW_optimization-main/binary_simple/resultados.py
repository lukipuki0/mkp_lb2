"""
Binary-Simple — decisión de estancamiento por umbral D2.
Corre todas las MHs, muestra métricas DTW en consola, y genera gráficos.

Uso (desde la raíz del proyecto):
    python -m binary_simple.resultados
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
)

# MHs a comparar — agregar nuevas acá
MHS = {
    "PSO": BinaryPSO,
    "GA": GeneticAlgorithm,
    "GWO": BinaryGWO,
    "DE": BinaryDE,
}


# =============================================================================
# CONSOLA: métricas por epoch
# =============================================================================


def print_epoch_results(nombre: str, resultados: list, inst: dict):
    """Imprime fitness + DTW params por cada epoch."""
    optimo = inst["optimo"]

    print(f"\n{'=' * 70}")
    print(f"  {nombre}")
    print(f"{'=' * 70}")

    for res in resultados:
        gap = f"Gap={100 - res['ganancia']:.2f}%" if optimo > 0 else "Gap=N/A"
        print(
            f"\n  Epoch {res['epoch']:02d} | "
            f"Fitness={res['mejor_fitness']:.1f} | "
            f"Fires={res['fire_count']} | {gap} | "
            f"t={res['tiempo']:.2f}s"
        )

        ready = [h for h in res["historial_dtw"] if h.get("ready")]
        if ready:
            last = ready[-1]
            d2s = [h["D2_vs_const"] for h in ready]
            print(
                f"    DTW  D2={np.mean(d2s):7.1f} (+-{np.std(d2s):.1f}) | "
                f"th_c={last['theta_c']:.1f} | "
                f"Decision: fire = D2 <= theta_c"
            )

    fits = [r["mejor_fitness"] for r in resultados]
    print(f"\n  {'-' * 50}")
    print(f"  Mejor={np.max(fits):.1f}  Prom={np.mean(fits):.1f}  "
          f"Peor={np.min(fits):.1f}  Std={np.std(fits):.1f}")
    if optimo > 0:
        print(f"  Optimo={optimo:.0f}  Gap={100 - np.max(fits) / optimo * 100:.2f}%")


# Colores por MH
MH_COLORS = {
    "PSO": "#2196F3",
    "BinaryPSO": "#2196F3",
    "GA": "#E91E63",
    "GWO": "#4CAF50",
    "DE": "#FF9800",
}

# Colores comunes (estilo paper)
COLORS = {
    "fire_marker": "#c0392b",
    "explore_bg": "#e74c3c",
    "optimum": "#7f8c8d",
    "d2": "#2980b9",
    "theta_c": "#e67e22",
}


def _get_explore_ranges(hist_mode):
    """Detecta rangos contiguos donde mode == 'explore'."""
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


def _get_fire_transitions(hist_mode):
    """Iteraciones donde arranca un nuevo fire (exploit → explore)."""
    transitions = []
    for i in range(len(hist_mode)):
        prev = hist_mode[i - 1] if i > 0 else "exploit"
        if hist_mode[i] == "explore" and prev != "explore":
            transitions.append(i)
    return transitions


def generate_plots(resultados_por_mh: dict, inst: dict, save_dir: str):
    """
    Genera 2 paneles estilo paper:
      1. Fitness: todas las MHs superpuestas + líneas punteadas en explore + fire markers + óptimo
      2. D2 vs theta_c: señal de decisión de cada MH + fire markers
    """
    nombres = list(resultados_por_mh.keys())

    # Seleccionar el mejor epoch de cada MH
    mejores = {}
    for nombre in nombres:
        resultados = resultados_por_mh[nombre]
        idx = int(np.argmax([r["mejor_fitness"] for r in resultados]))
        mejores[nombre] = resultados[idx]

    optimo = inst["optimo"]

    # --- Estilo paper ---
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

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )

    # =========================================================================
    # Panel 1: FITNESS SUPERPUESTO con líneas punteadas en explore
    # =========================================================================
    for nombre in nombres:
        res = mejores[nombre]
        hist_fit = res["historial_fitness"]
        hist_mode = res["historial_modos"]
        color = MH_COLORS.get(nombre, "#666666")

        explore_ranges = _get_explore_ranges(hist_mode)
        fire_transitions = _get_fire_transitions(hist_mode)

        # Crear máscara de explore
        is_explore = [False] * len(hist_fit)
        for s, e in explore_ranges:
            for j in range(s, min(e + 1, len(hist_fit))):
                is_explore[j] = True

        # Dibujar segmentos con diferentes estilos
        prev_type = is_explore[0]
        seg_start = 0
        for k in range(1, len(hist_fit)):
            if is_explore[k] != prev_type:
                style = "--" if prev_type else "-"
                ax1.plot(
                    range(seg_start, k + 1),
                    hist_fit[seg_start:k + 1],
                    color=color, linewidth=1.5, linestyle=style,
                    zorder=3,
                )
                seg_start = k
                prev_type = is_explore[k]
        # Último segmento
        style = "--" if prev_type else "-"
        ax1.plot(
            range(seg_start, len(hist_fit)),
            hist_fit[seg_start:],
            color=color, linewidth=1.5, linestyle=style,
            zorder=3,
        )

        # Leyenda (una sola entrada por MH)
        ax1.plot(
            [], [], color=color, linewidth=1.5, linestyle="-",
            label=f"{nombre} (max={res['mejor_fitness']:.1f}, fires={res['fire_count']})",
        )

        # Fire markers (triángulos en los puntos de transición)
        if fire_transitions:
            fire_fits = [hist_fit[i] for i in fire_transitions]
            ax1.scatter(
                fire_transitions, fire_fits, color=COLORS["fire_marker"],
                marker="v", s=30, zorder=5,
                edgecolors="white", linewidths=0.3,
            )

    # Óptimo
    if optimo > 0:
        ax1.axhline(
            y=optimo, color=COLORS["optimum"], linestyle="--",
            linewidth=1, alpha=0.7, label=f"Known optimum ({optimo})",
        )

    # Leyenda de estilos de línea
    ax1.plot([], [], color="gray", linewidth=1.5, linestyle="-", label="Exploit")
    ax1.plot([], [], color="gray", linewidth=1.5, linestyle="--", label="Explore")

    ax1.set_ylabel("Fitness")
    ax1.set_title(f"Binary-Simple — All MHs (seed={SEMILLA})")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle=":")

    # =========================================================================
    # Panel 2: D2 vs theta_c (señal de decisión Binary-Simple)
    # =========================================================================
    for mh_idx, nombre in enumerate(nombres):
        res = mejores[nombre]
        hist_dtw = res["historial_dtw"]
        hist_mode = res["historial_modos"]
        color = MH_COLORS.get(nombre, "#666666")

        fire_transitions = _get_fire_transitions(hist_mode)

        ready_iters, d2_vals, theta_c_vals = [], [], []
        for i, h in enumerate(hist_dtw):
            if h.get("ready"):
                ready_iters.append(i)
                d2_vals.append(h["D2_vs_const"])
                theta_c_vals.append(h["theta_c"])

        if ready_iters:
            # D2
            ax2.plot(
                ready_iters, d2_vals, color=color,
                linewidth=1.3, label=rf"$D_2$ {nombre}", zorder=3,
            )
            # Theta_c (línea punteada)
            ax2.plot(
                ready_iters, theta_c_vals, color=color,
                linewidth=0.8, linestyle=":", alpha=0.5,
                label=rf"$\theta_c$ {nombre}", zorder=2,
            )

            # Fire markers en D2
            for fi in fire_transitions:
                if fi in ready_iters:
                    idx = ready_iters.index(fi)
                    ax2.scatter(
                        fi, d2_vals[idx], color=color,
                        marker="v", s=40, zorder=5,
                        edgecolors="white", linewidths=0.5,
                    )

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$D_2$ (DTW to plateau)")
    ax2.set_title(
        r"Binary-Simple — Fire when $D_2 \leq \theta_c$"
    )
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax2.grid(True, alpha=0.2, linestyle=":")

    plt.tight_layout(h_pad=1.5)

    # --- Guardar ---
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    inst_name = Path(RUTA_INSTANCIA).stem
    for ext in ("png", "pdf"):
        path = f"{save_dir}/binary_simple_{inst_name}_{INDICE_INSTANCIA}.{ext}"
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
    print("  RESULTADOS MAESTRO — Binary-Simple")
    print("=" * 70)
    print(f"  Decision rule: fire = D2 <= theta_c")
    print(f"  Instancia: {RUTA_INSTANCIA}[{INDICE_INSTANCIA}]")
    print(f"  n={inst['n']}, m={inst['m']}")
    print(f"  Particulas/Pop: {NUM_PARTICULAS}, Iteraciones: {NUM_ITERACIONES}, "
          f"Epochs: {EPOCHS}")
    print(f"  DTW: window={DTW_CFG.window}, ddtw={DTW_CFG.use_ddtw}")

    # Crear carpeta de salida con timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    inst_name = Path(RUTA_INSTANCIA).stem
    save_dir = f"results/binary_simple/todos/{inst_name}_{INDICE_INSTANCIA}/comparacion_mhs_{run_id}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    resultados_por_mh = {}

    for nombre, mh_class in MHS.items():
        print(f"\n  Ejecutando {nombre}...")
        resultados = run_epochs(
            mh_class=mh_class,
            inst=inst,
            monitor_cfg=DTW_CFG,
            num_particulas=NUM_PARTICULAS,
            num_iteraciones=NUM_ITERACIONES,
            epochs=EPOCHS,
            verbose=False,
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
                "estrategia": "binary_simple",
                "decision_rule": "D2 <= theta_c",
                "instancia": RUTA_INSTANCIA,
                "idx": INDICE_INSTANCIA,
                "poblacion": NUM_PARTICULAS,
                "iteraciones": NUM_ITERACIONES,
                "dtw_window": DTW_CFG.window,
            },
        )

    generate_plots(resultados_por_mh, inst, save_dir)


if __name__ == "__main__":
    main()
