"""
rotating_benchmark.py
---------------------
Script maestro para el Pipeline Hibrido de Rotacion de Metaheuristicas.

Ejecuta el orquestador DTW y genera:
  - Log de consola con cada switch de MH
  - Graficos separados en plots/ (convergencia, delta DTW, Gantt de switches)
  - CSV con historial de fitness y delta DTW
  - Reporte TXT con el resumen de la ejecucion
"""

import os
import csv
import random
import datetime

import numpy as np

from mkp_core.data_loader import cargar_instancias, seleccionar_instancia
from mkp_core.problem     import MKPInstance
from dtw_stagnation       import StagnationConfig
from hybrid_mkp.orchestrator import ejecutar_pipeline, COLORES_MH
from plots import (
    grafico_convergencia, 
    grafico_instantaneo, 
    grafico_solo_instantaneo,
    grafico_dtw_delta, 
    grafico_switches
)

# ── Configuracion ──────────────────────────────────────────────────────────────

INSTANCE_URL   = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb1.txt"
INSTANCE_INDEX = 9

TIEMPO_MAX  = 120    # segundos totales de ejecucion
RANDOM_SEED = None   # None -> no fijar semilla; int -> reproducible
OUTPUT_DIR  = os.path.join("resultados", "pipeline_hibrido")

# DTW Stagnation params
STAG_WINDOW      = 30       # Tamaño de la ventana deslizante (últimas N iteraciones evaluadas)
STAG_BAND        = 0        # Banda Sakoe-Chiba para DTW. 0 = auto (10% de la ventana)
STAG_MIN_SLOPE   = 0.0      # Pendiente de la rampa ideal. 0.0 = auto (1% del progreso en la ventana)
STAG_PLATEAU_MAX = 15       # Iteraciones máximas permitidas sin mejora absoluta (fitness plano)
STAG_PATIENCE    = 3        # Alarmas consecutivas requeridas para confirmar el estancamiento (evita falsos positivos)
STAG_USE_DDTW    = False    # Usar derivadas (DDTW) en vez de valores absolutos (DTW)
STAG_ADAPT       = True     # Si es True, adapta los umbrales dinámicamente usando el historial
STAG_P_LOW       = 30.0     # Percentil bajo para umbral de línea plana (qué tan estricto es D2)
STAG_P_HIGH      = 70.0     # Percentil alto para umbral de rampa/delta (qué tan estricto es D1)


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

    # ── Exportar CSV con historial de fitness y delta DTW ─────────────────────
    csv_path = os.path.join(output_dir, "historial_dtw.csv")
    deltas   = resultado.dtw_deltas_global
    inst_hist = resultado.historial_inst_global
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteracion", "fitness", "dtw_delta", "fitness_instantaneo"])
        for i, fit in enumerate(resultado.historial_global):
            d = deltas[i] if i < len(deltas) else ""
            d_str = "" if (isinstance(d, float) and np.isnan(d)) else d
            fi = inst_hist[i] if i < len(inst_hist) else ""
            writer.writerow([i, fit, d_str, fi])
    print(f"  [csv] Guardado en '{csv_path}'")

    # ── Generar graficos separados ─────────────────────────────────────────────
    print("\n  Generando graficos...")
    grafico_convergencia(
        historial_global = resultado.historial_global,
        log_switches     = resultado.log_switches,
        colores_mh       = COLORES_MH,
        valor_optimo     = resultado.valor_optimo,
        output_dir       = output_dir,
    )
    grafico_instantaneo(
        historial_global      = resultado.historial_global,
        historial_inst_global = resultado.historial_inst_global,
        log_switches          = resultado.log_switches,
        colores_mh            = COLORES_MH,
        valor_optimo          = resultado.valor_optimo,
        output_dir            = output_dir,
    )
    grafico_solo_instantaneo(
        historial_inst_global = resultado.historial_inst_global,
        log_switches          = resultado.log_switches,
        colores_mh            = COLORES_MH,
        valor_optimo          = resultado.valor_optimo,
        output_dir            = output_dir,
    )
    grafico_dtw_delta(
        dtw_deltas_global = resultado.dtw_deltas_global,
        log_switches      = resultado.log_switches,
        colores_mh        = COLORES_MH,
        output_dir        = output_dir,
    )
    grafico_switches(
        log_switches = resultado.log_switches,
        colores_mh   = COLORES_MH,
        output_dir   = output_dir,
    )

    print("\n  PIPELINE COMPLETADO.\n")


if __name__ == "__main__":
    main()
