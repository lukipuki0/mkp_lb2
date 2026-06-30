"""
continuous_benchmark/benchmark_continuo.py
------------------------------------------
Script principal: ejecuta el Pipeline Hibrido DTW sobre TODAS las funciones
continuas definidas en test_functions.py en una sola corrida.

Cada funcion genera su propia subcarpeta con graficos, CSV y reporte.
Al final se genera un resumen global (TXT, CSV, MD) comparando todas
las funciones en la carpeta raiz del run.

Uso:
    python -m continuous_benchmark.benchmark_continuo
"""

import os
import csv
import random
import datetime
import sys

import numpy as np

# Agregar raiz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dtw_stagnation import StagnationConfig
from continuous_benchmark.funciones_cec2022 import get_test_functions, ContinuousFunction
from continuous_benchmark.orchestrator import ejecutar_pipeline, COLORES_MH
from plots import (
    grafico_convergencia,
    grafico_instantaneo,
    grafico_solo_instantaneo,
    grafico_dtw_delta,
    grafico_switches,
)


# ── Configuracion ─────────────────────────────────────────────────────────────

TIEMPO_MAX_POR_FUNCION = 10     # segundos por funcion
RANDOM_SEED            = 42     # None -> no fijar semilla
OUTPUT_BASE            = os.path.join("resultados", "benchmark_continuo")
DIMENSION              = 30     # dimensionalidad de las funciones

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


# ── Procesar una funcion ──────────────────────────────────────────────────────

def procesar_funcion(
    func       : ContinuousFunction,
    tiempo_max : float,
    stag_cfg   : StagnationConfig,
    output_dir : str,
) -> dict:
    """Ejecuta el pipeline sobre una funcion continua y guarda todos los artefactos."""
    os.makedirs(output_dir, exist_ok=True)

    resultado = ejecutar_pipeline(
        func       = func,
        tiempo_max = tiempo_max,
        stag_cfg   = stag_cfg,
        verbose    = True,
    )

    # ── Reporte consola ───────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  RESUMEN - {func.name}")
    print(sep)
    print(f"  Mejor valor global : {resultado.mejor_valor_global:.6f}")
    print(f"  Optimo conocido    : {resultado.valor_optimo:.6f}")
    print(f"  Gap relativo       : {resultado.gap_pct:.2f}%")
    print(f"  Total de switches  : {resultado.n_switches}")
    print()
    print(f"  {'#':<3} {'MH':<5} {'Mejor':>12}  {'Inicio':>7}  {'Fin':>7}  {'Iters':>6}")
    print("  " + "-" * 54)
    for i, sw in enumerate(resultado.log_switches, 1):
        print(f"  {i:<3} {sw.mh_nombre:<5} {sw.mejor_valor:>12.4f}"
              f"  {sw.t_inicio:>6.1f}s  {sw.t_fin:>6.1f}s  {sw.n_iters:>6}")

    # ── Guardar reporte TXT ───────────────────────────────────────────────
    report_path = os.path.join(output_dir, "resumen_pipeline.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Funcion            : {func.name}\n")
        f.write(f"Dimension          : {func.n_dim}\n")
        f.write(f"Limites            : [{func.lb}, {func.ub}]\n")
        f.write(f"Mejor valor global : {resultado.mejor_valor_global:.6f}\n")
        f.write(f"Optimo conocido    : {resultado.valor_optimo:.6f}\n")
        f.write(f"Gap relativo       : {resultado.gap_pct:.2f}%\n")
        f.write(f"Total switches     : {resultado.n_switches}\n\n")
        for i, sw in enumerate(resultado.log_switches, 1):
            f.write(f"{i}. {sw.mh_nombre} | mejor={sw.mejor_valor:.6f}"
                    f" | {sw.t_inicio:.1f}s-{sw.t_fin:.1f}s | iters={sw.n_iters}\n")
    print(f"\n  [txt] {report_path}")

    # ── Exportar CSV ──────────────────────────────────────────────────────
    csv_path  = os.path.join(output_dir, "historial_dtw.csv")
    deltas    = resultado.dtw_deltas_global
    inst_hist = resultado.historial_inst_global
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteracion", "fitness", "dtw_delta", "fitness_instantaneo"])
        for i, fit in enumerate(resultado.historial_global):
            d = deltas[i] if i < len(deltas) else ""
            d_str = "" if (isinstance(d, float) and np.isnan(d)) else d
            fi = inst_hist[i] if i < len(inst_hist) else ""
            writer.writerow([i, fit, d_str, fi])
    print(f"  [csv] {csv_path}")

    # ── Generar graficos ──────────────────────────────────────────────────
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

    return {
        "nombre":       func.name,
        "n_dim":        func.n_dim,
        "mejor_valor":  resultado.mejor_valor_global,
        "valor_optimo": resultado.valor_optimo,
        "gap_pct":      resultado.gap_pct,
        "n_switches":   resultado.n_switches,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    funciones = get_test_functions(DIMENSION)

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

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(OUTPUT_BASE, f"run_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    banner = "=" * 62
    print(f"\n{banner}")
    print("  CONTINUOUS BENCHMARK - Pipeline Hibrido DTW")
    print(banner)
    print(f"  Funciones a procesar  : {len(funciones)}")
    for fn in funciones:
        print(f"    - {fn.name} (Dim={fn.n_dim}, [{fn.lb}, {fn.ub}])")
    print(f"  Tiempo max / funcion  : {TIEMPO_MAX_POR_FUNCION}s")
    print(f"  Carpeta de salida     : {batch_dir}")
    print(banner)

    resumen_global: list[dict] = []

    for idx, func in enumerate(funciones, 1):
        print(f"\n{'-' * 62}")
        print(f"  [{idx}/{len(funciones)}] Optimizando {func.name} (Dim={func.n_dim})")
        print(f"{'-' * 62}")

        func_dir = os.path.join(batch_dir, func.name)

        resumen = procesar_funcion(
            func       = func,
            tiempo_max = TIEMPO_MAX_POR_FUNCION,
            stag_cfg   = stag_cfg,
            output_dir = func_dir,
        )
        resumen_global.append(resumen)

    # ── Resumen global ────────────────────────────────────────────────────
    print(f"\n\n{banner}")
    print("  RESUMEN GLOBAL DEL BATCH CONTINUO")
    print(banner)
    print(f"  {'#':<3} {'Funcion':<15} {'Dim':>5} {'Mejor':>14} {'Optimo':>10} {'Gap%':>10} {'Switches':>9}")
    print("  " + "-" * 70)
    for i, r in enumerate(resumen_global, 1):
        gap_str = f"{r['gap_pct']:.4f}"
        print(f"  {i:<3} {r['nombre']:<15} {r['n_dim']:>5} {r['mejor_valor']:>14.6f}"
              f" {r['valor_optimo']:>10.4f} {gap_str:>10} {r['n_switches']:>9}")
    print(banner)

    # TXT
    resumen_txt = os.path.join(batch_dir, "resumen_global.txt")
    with open(resumen_txt, "w", encoding="utf-8") as f:
        f.write("RESUMEN GLOBAL DEL BATCH CONTINUO\n")
        f.write(f"Fecha       : {timestamp}\n")
        f.write(f"Funciones   : {len(funciones)}\n")
        f.write(f"Tiempo/func : {TIEMPO_MAX_POR_FUNCION}s\n\n")
        f.write(f"{'#':<3} {'Funcion':<15} {'Dim':>5} {'Mejor':>14} {'Optimo':>10} {'Gap%':>10} {'Switches':>9}\n")
        f.write("-" * 72 + "\n")
        for i, r in enumerate(resumen_global, 1):
            gap_str = f"{r['gap_pct']:.4f}"
            f.write(f"{i:<3} {r['nombre']:<15} {r['n_dim']:>5} {r['mejor_valor']:>14.6f}"
                    f" {r['valor_optimo']:>10.4f} {gap_str:>10} {r['n_switches']:>9}\n")
    print(f"\n  [txt] Resumen global guardado en '{resumen_txt}'")

    # CSV
    resumen_csv = os.path.join(batch_dir, "resumen_global.csv")
    with open(resumen_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["funcion", "n_dim", "mejor_valor", "valor_optimo", "gap_pct", "n_switches"])
        for r in resumen_global:
            writer.writerow([
                r["nombre"], r["n_dim"], r["mejor_valor"],
                r["valor_optimo"], r["gap_pct"], r["n_switches"],
            ])
    print(f"  [csv] Resumen global guardado en '{resumen_csv}'")

    # Markdown
    resumen_md = os.path.join(batch_dir, "resumen_global.md")
    with open(resumen_md, "w", encoding="utf-8") as f:
        f.write(f"# Resumen Global - Benchmark Continuo ({timestamp})\n\n")
        f.write(f"- **Total funciones:** {len(funciones)}\n")
        f.write(f"- **Tiempo maximo por funcion:** {TIEMPO_MAX_POR_FUNCION} s\n\n")
        f.write("| # | Funcion | Dim | Mejor Valor | Optimo | Gap % | Switches |\n")
        f.write("|---|---------|-----|-------------|--------|-------|----------|\n")
        for i, r in enumerate(resumen_global, 1):
            gap_str = f"{r['gap_pct']:.4f}%"
            f.write(f"| {i} | `{r['nombre']}` | {r['n_dim']} | {r['mejor_valor']:.6f}"
                    f" | {r['valor_optimo']:.4f} | {gap_str} | {r['n_switches']} |\n")
    print(f"  [md]  Resumen global guardado en '{resumen_md}'")

    print(f"\n  BENCHMARK CONTINUO COMPLETADO. ({len(funciones)} funciones procesadas)\n")


if __name__ == "__main__":
    main()
