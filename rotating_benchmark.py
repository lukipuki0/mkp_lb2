"""
rotating_benchmark.py
---------------------
Script maestro para el Pipeline Hibrido de Rotacion de Metaheuristicas.

Ejecuta el orquestador DTW sobre TODAS las instancias del archivo TXT y genera
por cada una:
  - Carpeta propia:  resultados/pipeline_hibrido/run_TIMESTAMP/<nombre_inst>/
  - Log de consola con cada switch de MH
  - Graficos separados en plots/ (convergencia, delta DTW, Gantt de switches)
  - CSV con historial de fitness y delta DTW
  - Reporte TXT con el resumen de la ejecucion

Al finalizar se genera un resumen global (TXT, CSV y MD) en la carpeta raiz
del run.
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
    grafico_switches,
)

# ── Configuracion ──────────────────────────────────────────────────────────────

INSTANCE_FILE  = os.path.join("instancias", "mknapcb1.txt")   # <-- cambia el TXT aqui

TIEMPO_MAX  = 120    # segundos totales de ejecucion por instancia
RANDOM_SEED = None   # None -> no fijar semilla; int -> reproducible
OUTPUT_DIR  = os.path.join("resultados", "pipeline_hibrido")

# DTW Stagnation params
STAG_WINDOW      = 30       # Tamano de la ventana deslizante (ultimas N iteraciones evaluadas)
STAG_BAND        = 0        # Banda Sakoe-Chiba para DTW. 0 = auto (10% de la ventana)
STAG_MIN_SLOPE   = 0.5      # Pendiente de la rampa ideal. 0.0 = auto (1% del progreso en la ventana)
STAG_PLATEAU_MAX = 15       # Iteraciones maximas permitidas sin mejora absoluta (fitness plano)
STAG_PATIENCE    = 3        # Alarmas consecutivas requeridas para confirmar el estancamiento (evita falsos positivos)
STAG_USE_DDTW    = False    # Usar derivadas (DDTW) en vez de valores absolutos (DTW)
STAG_ADAPT       = True     # Si es True, adapta los umbrales dinamicamente usando el historial
STAG_P_LOW       = 30.0     # Percentil bajo para umbral de linea plana (que tan estricto es D2)
STAG_P_HIGH      = 70.0     # Percentil alto para umbral de rampa/delta (que tan estricto es D1)


# ── Procesar una instancia ─────────────────────────────────────────────────────

def procesar_instancia(
    inst: MKPInstance,
    nombre: str,
    tiempo_max: float,
    stag_cfg: StagnationConfig,
    output_dir: str,
    verbose: bool = True,
) -> dict:
    """Ejecuta el pipeline hibrido sobre una instancia y guarda todos los artefactos.

    Returns
    -------
    dict con claves: nombre, mejor_valor, valor_optimo, gap_pct, n_switches.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Ejecutar pipeline ─────────────────────────────────────────────────
    resultado = ejecutar_pipeline(
        inst       = inst,
        tiempo_max = tiempo_max,
        stag_cfg   = stag_cfg,
        verbose    = verbose,
    )

    # ── Reporte en consola ────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  RESUMEN - {nombre}")
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

    # ── Guardar reporte TXT ───────────────────────────────────────────────
    report_path = os.path.join(output_dir, "resumen_pipeline.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Instancia          : {nombre}\n")
        f.write(f"Items / Restricc.  : {inst.n} / {inst.m}\n")
        f.write(f"Mejor valor global : {resultado.mejor_valor_global:.1f}\n")
        f.write(f"Optimo conocido    : {resultado.valor_optimo:.1f}\n")
        if resultado.gap_pct is not None:
            f.write(f"Gap relativo       : {resultado.gap_pct:.2f}%\n")
        f.write(f"Total switches     : {resultado.n_switches}\n\n")
        for i, sw in enumerate(resultado.log_switches, 1):
            f.write(f"{i}. {sw.mh_nombre} ({sw.tipo}) | mejor={sw.mejor_valor:.1f}"
                    f" | {sw.t_inicio:.1f}s-{sw.t_fin:.1f}s | iters={sw.n_iters}\n")
    print(f"\n  [txt] Guardado en '{report_path}'")

    # ── Exportar CSV con historial de fitness y delta DTW ─────────────────
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
    print(f"  [csv] Guardado en '{csv_path}'")

    # ── Generar graficos separados ─────────────────────────────────────────
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
        "nombre":       nombre,
        "n":            inst.n,
        "m":            inst.m,
        "mejor_valor":  resultado.mejor_valor_global,
        "valor_optimo": resultado.valor_optimo,
        "gap_pct":      resultado.gap_pct,
        "n_switches":   resultado.n_switches,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # Nombre base del archivo (sin extension) para nombrar las carpetas
    txt_nombre = os.path.splitext(os.path.basename(INSTANCE_FILE))[0]  # e.g. "mknapcb1"

    # Cargar TODAS las instancias del archivo TXT
    todas  = cargar_instancias(INSTANCE_FILE)
    n_total = len(todas)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Configuracion DTW compartida para todas las instancias
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

    banner = "=" * 62
    print(f"\n{banner}")
    print("  ROTATING BENCHMARK - Pipeline Hibrido DTW")
    print(banner)
    print(f"  Archivo TXT           : {INSTANCE_FILE}")
    print(f"  Instancias a procesar : {n_total}")
    print(f"  Tiempo max / instancia: {TIEMPO_MAX}s")
    print(f"  Carpeta de salida     : {run_dir}")
    print(banner)

    resumen_global: list[dict] = []

    for idx, data in enumerate(todas):
        nombre = f"{txt_nombre}_inst{idx}"
        inst   = MKPInstance.from_dict(data)

        print(f"\n{'-' * 62}")
        print(f"  [{idx + 1}/{n_total}] {nombre}  |  {inst.n} items, {inst.m} restricciones")
        print(f"  Optimo    : {inst.valor_optimo}")
        print(f"{'-' * 62}")

        # Carpeta dedicada para esta instancia
        inst_dir = os.path.join(run_dir, nombre)

        resumen = procesar_instancia(
            inst       = inst,
            nombre     = nombre,
            tiempo_max = TIEMPO_MAX,
            stag_cfg   = stag_cfg,
            output_dir = inst_dir,
            verbose    = True,
        )
        resumen_global.append(resumen)

    # ── Resumen global ─────────────────────────────────────────────────────────
    print(f"\n\n{banner}")
    print("  RESUMEN GLOBAL")
    print(banner)
    print(f"  {'#':<3} {'Instancia':<22} {'N':>5} {'M':>3} {'Mejor':>10} {'Optimo':>10} {'Gap%':>8} {'Switches':>9}")
    print("  " + "-" * 74)
    for i, r in enumerate(resumen_global, 1):
        gap_str = f"{r['gap_pct']:.2f}" if r["gap_pct"] is not None else "N/A"
        print(f"  {i:<3} {r['nombre']:<22} {r['n']:>5} {r['m']:>3} {r['mejor_valor']:>10.1f}"
              f" {r['valor_optimo']:>10.1f} {gap_str:>8} {r['n_switches']:>9}")
    print(banner)

    # Guardar resumen global TXT
    resumen_txt = os.path.join(run_dir, "resumen_global.txt")
    with open(resumen_txt, "w", encoding="utf-8") as f:
        f.write("RESUMEN GLOBAL - ROTATING BENCHMARK\n")
        f.write(f"Fecha       : {timestamp}\n")
        f.write(f"Archivo     : {INSTANCE_FILE}\n")
        f.write(f"Instancias  : {n_total}\n")
        f.write(f"Tiempo/inst : {TIEMPO_MAX}s\n\n")
        f.write(f"{'#':<3} {'Instancia':<22} {'N':>5} {'M':>3} {'Mejor':>10} {'Optimo':>10} {'Gap%':>8} {'Switches':>9}\n")
        f.write("-" * 76 + "\n")
        for i, r in enumerate(resumen_global, 1):
            gap_str = f"{r['gap_pct']:.2f}" if r["gap_pct"] is not None else "N/A"
            f.write(f"{i:<3} {r['nombre']:<22} {r['n']:>5} {r['m']:>3} {r['mejor_valor']:>10.1f}"
                    f" {r['valor_optimo']:>10.1f} {gap_str:>8} {r['n_switches']:>9}\n")
    print(f"\n  [txt] Resumen global guardado en '{resumen_txt}'")

    # Guardar resumen global CSV
    resumen_csv = os.path.join(run_dir, "resumen_global.csv")
    with open(resumen_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instancia", "n", "m", "mejor_valor", "valor_optimo", "gap_pct", "n_switches"])
        for r in resumen_global:
            writer.writerow([
                r["nombre"], r["n"], r["m"],
                r["mejor_valor"], r["valor_optimo"],
                r["gap_pct"] if r["gap_pct"] is not None else "",
                r["n_switches"],
            ])
    print(f"  [csv] Resumen global guardado en '{resumen_csv}'")

    # Guardar resumen global Markdown
    resumen_md = os.path.join(run_dir, "resumen_global.md")
    with open(resumen_md, "w", encoding="utf-8") as f:
        f.write(f"# Resumen Global - {txt_nombre} ({timestamp})\n\n")
        f.write(f"- **Archivo:** `{INSTANCE_FILE}`\n")
        f.write(f"- **Total instancias:** {n_total}\n")
        f.write(f"- **Tiempo maximo por instancia:** {TIEMPO_MAX} s\n\n")
        f.write("| # | Instancia | N | M | Mejor Valor | Optimo | Gap % | Switches |\n")
        f.write("|---|-----------|---|---|-------------|--------|-------|----------|\n")
        for i, r in enumerate(resumen_global, 1):
            gap_str = f"{r['gap_pct']:.2f}%" if r["gap_pct"] is not None else "N/A"
            f.write(f"| {i} | `{r['nombre']}` | {r['n']} | {r['m']} | {r['mejor_valor']:.1f}"
                    f" | {r['valor_optimo']:.1f} | {gap_str} | {r['n_switches']} |\n")
    print(f"  [md]  Resumen global guardado en '{resumen_md}'")

    print(f"\n  PIPELINE COMPLETADO. ({n_total} instancias procesadas)\n")


if __name__ == "__main__":
    main()
