"""
rotating_benchmark.py
---------------------
Script maestro para el Pipeline Hibrido de Rotacion de Metaheuristicas.

Ejecuta el orquestador DTW y genera:
  - Log de consola con cada switch de MH
  - Grafico de convergencia global coloreado por algoritmo
  - Reporte TXT con el resumen de la ejecucion
"""

import os
import random
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mkp_core.data_loader import cargar_instancias, seleccionar_instancia
from mkp_core.problem     import MKPInstance
from dtw_stagnation       import StagnationConfig
from hybrid_mkp.orchestrator import ejecutar_pipeline, COLORES_MH

# ── Configuracion ──────────────────────────────────────────────────────────────

INSTANCE_URL   = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb1.txt"
INSTANCE_INDEX = 9

TIEMPO_MAX  = 120    # segundos totales de ejecucion
RANDOM_SEED = None   # None -> no fijar semilla; int -> reproducible
OUTPUT_DIR  = os.path.join("resultados", "pipeline_hibrido")

# DTW Stagnation params
STAG_WINDOW      = 30
STAG_BAND        = 0
STAG_MIN_SLOPE   = 0.0
STAG_PLATEAU_MAX = 15
STAG_PATIENCE    = 3
STAG_USE_DDTW    = False
STAG_ADAPT       = True
STAG_P_LOW       = 30.0
STAG_P_HIGH      = 70.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Cargar instancia
    instancias = cargar_instancias(INSTANCE_URL)
    data       = seleccionar_instancia(instancias, INSTANCE_INDEX)
    inst       = MKPInstance.from_dict(data)

    print(f"\n  Instancia : {inst.n} items, {inst.m} restricciones")
    print(f"  Optimo    : {inst.valor_optimo}")
    print(f"  Tiempo max: {TIEMPO_MAX}s")

    # Configuracion DTW
    stag_cfg = StagnationConfig(
        window           = STAG_WINDOW,
        band             = STAG_BAND,
        min_slope        = STAG_MIN_SLOPE,
        plateau_max      = STAG_PLATEAU_MAX,
        patience         = STAG_PATIENCE,
        use_ddtw         = STAG_USE_DDTW,
        adapt_thresholds = STAG_ADAPT,
        p_low            = STAG_P_LOW,
        p_high           = STAG_P_HIGH,
    )

    # ── Ejecutar pipeline ──────────────────────────────────────────────────────
    resultado = ejecutar_pipeline(
        inst       = inst,
        tiempo_max = TIEMPO_MAX,
        stag_cfg   = stag_cfg,
        verbose    = True,
    )

    # ── Reporte en consola ─────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("  RESUMEN FINAL DEL PIPELINE")
    print(sep)
    print(f"  Mejor valor global : {resultado.mejor_valor_global:.1f}")
    print(f"  Optimo conocido    : {resultado.valor_optimo:.1f}")
    if resultado.gap_pct is not None:
        print(f"  Gap relativo       : {resultado.gap_pct:.2f}%")
    print(f"  Total de switches  : {resultado.n_switches}")
    print()
    print(f"  {'#':<3} {'MH':<5} {'Tipo':<14} {'Mejor':>10}  {'Inicio':>7}  {'Fin':>7}  {'Iters':>6}")
    print("  " + "-" * 58)
    for i, sw in enumerate(resultado.log_switches, 1):
        print(f"  {i:<3} {sw.mh_nombre:<5} {sw.tipo:<14} {sw.mejor_valor:>10.1f}"
              f"  {sw.t_inicio:>6.1f}s  {sw.t_fin:>6.1f}s  {sw.n_iters:>6}")

    # ── Guardar reporte TXT ────────────────────────────────────────────────────
    report_path = os.path.join(output_dir, "resumen_pipeline.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Mejor valor global : {resultado.mejor_valor_global:.1f}\n")
        f.write(f"Optimo conocido    : {resultado.valor_optimo:.1f}\n")
        if resultado.gap_pct is not None:
            f.write(f"Gap relativo       : {resultado.gap_pct:.2f}%\n")
        f.write(f"Total switches     : {resultado.n_switches}\n\n")
        for i, sw in enumerate(resultado.log_switches, 1):
            f.write(f"{i}. {sw.mh_nombre} ({sw.tipo}) | mejor={sw.mejor_valor:.1f}"
                    f" | {sw.t_inicio:.1f}s-{sw.t_fin:.1f}s | iters={sw.n_iters}\n")
    print(f"\n  [txt] Guardado en '{report_path}'")

    # ── Grafico de convergencia coloreado por MH ───────────────────────────────
    if resultado.historial_global:
        fig, ax = plt.subplots(figsize=(14, 6))

        n_total = len(resultado.historial_global)
        t_total = resultado.log_switches[-1].t_fin if resultado.log_switches else 1.0

        offset = 0
        legend_patches = []
        seen = set()

        for sw in resultado.log_switches:
            n_seg = sw.n_iters
            if n_seg == 0:
                continue
            seg  = resultado.historial_global[offset: offset + n_seg]
            xs   = range(offset, offset + len(seg))
            col  = COLORES_MH.get(sw.mh_nombre, "gray")
            ax.plot(xs, seg, color=col, linewidth=1.6, alpha=0.85)
            ax.axvline(x=offset, color=col, linestyle="--", linewidth=0.8, alpha=0.4)
            offset += n_seg

            if sw.mh_nombre not in seen:
                legend_patches.append(
                    mpatches.Patch(color=col, label=sw.mh_nombre)
                )
                seen.add(sw.mh_nombre)

        if inst.valor_optimo > 0:
            ax.axhline(y=inst.valor_optimo, color="red", linestyle="--",
                       linewidth=1.4, label=f"Optimo ({inst.valor_optimo:.0f})")
            legend_patches.append(
                mpatches.Patch(color="red", label=f"Optimo ({inst.valor_optimo:.0f})")
            )

        ax.set_title("Pipeline Hibrido DTW - Convergencia Global", fontsize=14, fontweight="bold")
        ax.set_xlabel("Iteracion acumulada", fontsize=11)
        ax.set_ylabel("Mejor valor", fontsize=11)
        ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(output_dir, "convergencia_pipeline.png")
        fig.savefig(plot_path, dpi=150)
        print(f"  [plot] Guardado en '{plot_path}'")
        plt.close(fig)

    print("\n  PIPELINE COMPLETADO.\n")


if __name__ == "__main__":
    main()
