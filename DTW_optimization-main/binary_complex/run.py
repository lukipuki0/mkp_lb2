"""
Análisis individual de una MH + DTW Fire Binario.
Ejecuta 1 corrida, muestra métricas DTW por iteración, genera plot para paper.

Uso (desde la raíz del proyecto):
    python -m fire_binario.plot_experiment
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
    DTW_CFG,
    INDICE_INSTANCIA,
    MH_CLASS,
    NUM_ITERACIONES,
    NUM_PARTICULAS,
    RUTA_INSTANCIA,
    SEMILLA,
    VERBOSE,
)
from .runner import run_experiment


# ─── Colores para el plot ───────────────────────────────────────────────────
COLORS = {
    "fitness": "#1a5276",
    "fire_marker": "#c0392b",
    "explore_bg": "#e74c3c",
    "optimum": "#7f8c8d",
    "delta": "#6c3483",
    "theta_delta": "#e67e22",
}


# =============================================================================
# CONSOLA: tabla iteración por iteración
# =============================================================================


def print_iteration_table(res: dict, inst: dict) -> None:
    """Imprime tabla con todos los parámetros DTW por iteración."""
    hist_fit = res["historial_fitness"]
    hist_dtw = res["historial_dtw"]
    hist_mode = res["historial_modos"]
    optimo = inst["optimo"]

    hdr = (
        f"{'it':>4} | {'fitness':>10} | {'mode':>7} | {'fire':>5} | "
        f"{'D1':>7} | {'D2':>7} | {'delta':>8} | "
        f"{'th_c':>7} | {'th_r':>7} | {'th_d':>7} | "
        f"{'noimp':>5} | {'strk':>4}"
    )
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print(
        f"  DTW Log — {MH_CLASS.__name__} | "
        f"seed={res.get('semilla', '?')} | "
        f"instance={Path(RUTA_INSTANCIA).stem}[{INDICE_INSTANCIA}]"
    )
    if optimo > 0:
        print(f"  Optimo conocido: {optimo}")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)

    for i in range(len(hist_fit)):
        fitness = hist_fit[i]
        mode = "EXPLORE" if hist_mode[i] == "explore" else "EXPLOIT"
        dtw = hist_dtw[i]

        if dtw.get("ready"):
            fire_s = "  T  " if dtw["fire"] else "  F  "
            row = (
                f"{i:4d} | {fitness:10.1f} | {mode:>7} | {fire_s} | "
                f"{dtw['D1_vs_ramp']:7.1f} | {dtw['D2_vs_const']:7.1f} | "
                f"{dtw['delta']:+8.1f} | "
                f"{dtw['theta_c']:7.1f} | {dtw['theta_r']:7.1f} | "
                f"{dtw['theta_delta']:7.1f} | "
                f"{dtw['no_improve_len']:5d} | "
                f"{dtw['trigger_streak']:4d}"
            )
        else:
            row = (
                f"{i:4d} | {fitness:10.1f} | {mode:>7} |  ---  | "
                f"{'---':>7} | {'---':>7} | {'---':>8} | "
                f"{'---':>7} | {'---':>7} | {'---':>7} | "
                f"{dtw['no_improve_len']:5d} |  ---"
            )

        print(row)

    print(sep)
    print(f"  Fitness final: {hist_fit[-1]:.1f} | Fires: {res['fire_count']}")
    if optimo > 0:
        gap = 100 - hist_fit[-1] / optimo * 100
        print(f"  Gap al optimo: {gap:.2f}%")
    print()


# =============================================================================
# PLOT: 2 paneles para paper
# =============================================================================


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


def plot_paper(res: dict, inst: dict, save_dir: str = "results/fire_binario"):
    """
    Plot de 2 paneles para paper científico.

    Panel 1: Fitness + zonas explore + fire markers + óptimo
    Panel 2: Delta + theta_delta + fire markers
    """
    hist_fit = res["historial_fitness"]
    hist_dtw = res["historial_dtw"]
    hist_mode = res["historial_modos"]
    optimo = inst["optimo"]
    mh_name = MH_CLASS.__name__

    # --- Extraer datos DTW ---
    ready_iters, deltas, thetas_d = [], [], []
    for i, h in enumerate(hist_dtw):
        if h.get("ready"):
            ready_iters.append(i)
            deltas.append(h["delta"])
            thetas_d.append(h["theta_delta"])

    explore_ranges = _get_explore_ranges(hist_mode)
    fire_transitions = _get_fire_transitions(hist_mode)

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

    # ─── Panel 1: Fitness ────────────────────────────────────────────────
    iters = np.arange(len(hist_fit))
    max_fit = max(hist_fit)
    ax1.plot(
        iters, hist_fit, color=COLORS["fitness"],
        linewidth=1.5, label=f"Best fitness (max={max_fit:.0f})", zorder=3,
    )

    # Zonas explore (sombreado)
    for s, e in explore_ranges:
        ax1.axvspan(s, e, color=COLORS["explore_bg"], alpha=0.08, zorder=1)

    # Fire markers (triángulos en los puntos de transición)
    if fire_transitions:
        fire_fits = [hist_fit[i] for i in fire_transitions]
        ax1.scatter(
            fire_transitions, fire_fits, color=COLORS["fire_marker"],
            marker="v", s=50, zorder=5,
            label=f"Fire event ({len(fire_transitions)})",
            edgecolors="white", linewidths=0.5,
        )

    # Óptimo
    if optimo > 0:
        ax1.axhline(
            y=optimo, color=COLORS["optimum"], linestyle="--",
            linewidth=1, alpha=0.7, label=f"Known optimum ({optimo})",
        )

    ax1.set_ylabel("Fitness")
    ax1.set_title(
        f"{mh_name} — Convergence (seed={res.get('semilla', '?')})"
    )
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle=":")

    # ─── Panel 2: Delta ──────────────────────────────────────────────────
    ax2.plot(
        ready_iters, deltas, color=COLORS["delta"],
        linewidth=1.3, label=r"$\Delta$ (D1 $-$ D2)", zorder=3,
    )
    ax2.plot(
        ready_iters, thetas_d, color=COLORS["theta_delta"],
        linewidth=1, linestyle="--", alpha=0.7,
        label=r"$\theta_\Delta$ (threshold)", zorder=2,
    )
    ax2.axhline(y=0, color="#95a5a6", linestyle="-", linewidth=0.8, alpha=0.4)

    # Fire markers en delta
    for fi in fire_transitions:
        if fi in ready_iters:
            idx = ready_iters.index(fi)
            ax2.scatter(
                fi, deltas[idx], color=COLORS["fire_marker"],
                marker="v", s=35, zorder=5,
                edgecolors="white", linewidths=0.5,
            )

    # Sombreado delta > 0
    ax2.fill_between(
        ready_iters, deltas, 0,
        where=[d > 0 for d in deltas],
        color=COLORS["delta"], alpha=0.06,
    )

    # Zonas explore
    for s, e in explore_ranges:
        ax2.axvspan(s, e, color=COLORS["explore_bg"], alpha=0.08, zorder=1)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\Delta$ (D1 $-$ D2)")
    ax2.set_title(
        r"DTW Stagnation Signal — $\Delta > 0$ indicates stagnation"
    )
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.2, linestyle=":")

    plt.tight_layout(h_pad=1.5)

    # --- Guardar en subcarpeta por MH ---
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


# =============================================================================
# MAIN
# =============================================================================


def main():
    inst = cargar_instancia(RUTA_INSTANCIA, idx=INDICE_INSTANCIA)
    mh_name = MH_CLASS.__name__
    archivo = Path(RUTA_INSTANCIA).stem

    print(f"\n  MH: {mh_name}")
    print(f"  Instance: {archivo}[{INDICE_INSTANCIA}]"
          f" — n={inst['n']}, m={inst['m']}")
    print(f"  Pop: {NUM_PARTICULAS}, Iters: {NUM_ITERACIONES}")
    print(f"  DTW: window={DTW_CFG.window}, patience={DTW_CFG.patience}, "
          f"ddtw={DTW_CFG.use_ddtw}, adapt_th={DTW_CFG.adapt_thresholds}")

    res = run_experiment(
        mh_class=MH_CLASS,
        inst=inst,
        monitor_cfg=DTW_CFG,
        num_particulas=NUM_PARTICULAS,
        num_iteraciones=NUM_ITERACIONES,
        semilla=SEMILLA,
        verbose=VERBOSE,
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    res["semilla"] = SEMILLA
    res["run_id"] = run_id

    print_iteration_table(res, inst)
    plot_paper(res, inst)


if __name__ == "__main__":
    main()
