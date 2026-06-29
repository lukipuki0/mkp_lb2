"""
batch_benchmark.py
------------------
Ejecución por lotes del Pipeline Híbrido de Rotación de Metaheurísticas.

Lee la configuración de instancias desde 'config_instancias.json' y ejecuta
el orquestador DTW sobre cada una de forma secuencial. Los resultados se
organizan en:

    resultados/batch_runs/run_TIMESTAMP/
        ├── mknapcb1_inst0/
        │   ├── resumen_pipeline.txt
        │   ├── historial_dtw.csv
        │   ├── convergencia_fitness.png
        │   ├── convergencia_instantanea.png
        │   ├── solo_instantaneo.png
        │   ├── dtw_delta.png
        │   └── switches_gantt.png
        ├── mknapcb1_inst9/
        │   └── ...
        └── resumen_batch.txt          # resumen global de todas las instancias
"""

import os
import csv
import json
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


# ── Configuración y defaults ──────────────────────────────────────────────────

# Cambia esta variable para seleccionar qué instancia mknapcb ejecutar (1 a 9)
MKNAPCB_NUM = 1

# Parámetros de ejecución
TIEMPO_MAX_POR_INSTANCIA = 120
RANDOM_SEED = None
OUTPUT_BASE = os.path.join("resultados", "batch_runs")

# Parámetros de Stagnation (DTW)
STAG_WINDOW      = 30
STAG_BAND        = 0
STAG_MIN_SLOPE   = 0.1
STAG_PLATEAU_MAX = 15
STAG_PATIENCE    = 3
STAG_USE_DDTW    = False
STAG_ADAPT       = True
STAG_P_LOW       = 30.0
STAG_P_HIGH      = 70.0


def procesar_instancia(
    inst: MKPInstance,
    nombre: str,
    tiempo_max: float,
    stag_cfg: StagnationConfig,
    output_dir: str,
    verbose: bool = True,
) -> dict:
    """Ejecuta el pipeline híbrido sobre una instancia y guarda todos los artefactos.

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
    print(f"\n  [txt] {report_path}")

    # ── Exportar CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "historial_dtw.csv")
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

    # ── Generar gráficos ──────────────────────────────────────────────────
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
        "nombre":      nombre,
        "n":           inst.n,
        "m":           inst.m,
        "mejor_valor": resultado.mejor_valor_global,
        "valor_optimo": resultado.valor_optimo,
        "gap_pct":     resultado.gap_pct,
        "n_switches":  resultado.n_switches,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Usar el número de instancia configurado arriba
    mknapcb_num = MKNAPCB_NUM
    tiempo_max = TIEMPO_MAX_POR_INSTANCIA

    # Validar que esté en el rango de 1 a 9
    if mknapcb_num < 1 or mknapcb_num > 9:
        print(f"\n[!] Número de mknapcb inválido ({mknapcb_num}). Debe ser de 1 a 9. Usando 1.")
        mknapcb_num = 1

    # Generar automáticamente las 10 instancias de la mknapcb seleccionada (de la carpeta 'instancias')
    instancias = [
        {
            "url": f"instancias/mknapcb{mknapcb_num}.txt",
            "index": idx,
            "nombre": f"mknapcb{mknapcb_num}_inst{idx}"
        }
        for idx in range(10)
    ]

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # Configuración DTW compartida
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

    # Crear carpeta principal de esta sesión batch
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir  = os.path.join(OUTPUT_BASE, f"run_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    banner = "=" * 62
    print(f"\n{banner}")
    print("  BATCH BENCHMARK - Pipeline Hibrido DTW")
    print(banner)
    print(f"  Instancias a procesar : {len(instancias)}")
    print(f"  Tiempo max / instancia: {tiempo_max}s")
    print(f"  Carpeta de salida     : {batch_dir}")
    print(banner)

    # Cache de archivos descargados para no re-descargar la misma URL
    cache_urls: dict[str, list] = {}
    resumen_global: list[dict] = []

    for idx, entry in enumerate(instancias, 1):
        url    = entry["url"]
        index  = entry["index"]
        nombre = entry.get("nombre", f"inst_{idx}")

        print(f"\n{'-' * 62}")
        print(f"  [{idx}/{len(instancias)}] {nombre}  (url=...{url[-15:]}, index={index})")
        print(f"{'-' * 62}")

        # Cargar (o reutilizar) las instancias del archivo
        if url not in cache_urls:
            cache_urls[url] = cargar_instancias(url)
        data = seleccionar_instancia(cache_urls[url], index)
        inst = MKPInstance.from_dict(data)

        print(f"  Instancia : {inst.n} items, {inst.m} restricciones")
        print(f"  Optimo    : {inst.valor_optimo}")

        # Carpeta dedicada para esta instancia
        inst_dir = os.path.join(batch_dir, nombre)

        resumen = procesar_instancia(
            inst       = inst,
            nombre     = nombre,
            tiempo_max = tiempo_max,
            stag_cfg   = stag_cfg,
            output_dir = inst_dir,
            verbose    = True,
        )
        resumen_global.append(resumen)

    # ── Resumen global del batch ──────────────────────────────────────────
    print(f"\n\n{banner}")
    print("  RESUMEN GLOBAL DEL BATCH")
    print(banner)
    print(f"  {'#':<3} {'Instancia':<22} {'N':>5} {'M':>3} {'Mejor':>10} {'Optimo':>10} {'Gap%':>8} {'Switches':>9}")
    print("  " + "-" * 74)
    for i, r in enumerate(resumen_global, 1):
        gap_str = f"{r['gap_pct']:.2f}" if r["gap_pct"] is not None else "N/A"
        print(f"  {i:<3} {r['nombre']:<22} {r['n']:>5} {r['m']:>3} {r['mejor_valor']:>10.1f}"
              f" {r['valor_optimo']:>10.1f} {gap_str:>8} {r['n_switches']:>9}")
    print(banner)

    # Guardar resumen global en TXT
    resumen_path = os.path.join(batch_dir, "resumen_batch.txt")
    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write("RESUMEN GLOBAL DEL BATCH\n")
        f.write(f"Fecha       : {timestamp}\n")
        f.write(f"Instancias  : {len(instancias)}\n")
        f.write(f"Tiempo/inst : {tiempo_max}s\n\n")
        f.write(f"{'#':<3} {'Instancia':<22} {'N':>5} {'M':>3} {'Mejor':>10} {'Optimo':>10} {'Gap%':>8} {'Switches':>9}\n")
        f.write("-" * 76 + "\n")
        for i, r in enumerate(resumen_global, 1):
            gap_str = f"{r['gap_pct']:.2f}" if r["gap_pct"] is not None else "N/A"
            f.write(f"{i:<3} {r['nombre']:<22} {r['n']:>5} {r['m']:>3} {r['mejor_valor']:>10.1f}"
                    f" {r['valor_optimo']:>10.1f} {gap_str:>8} {r['n_switches']:>9}\n")
    print(f"\n  [txt] Resumen batch guardado en '{resumen_path}'")

    # Guardar resumen global en CSV
    csv_path = os.path.join(batch_dir, "resumen_batch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instancia", "n", "m", "mejor_valor", "valor_optimo", "gap_pct", "n_switches"])
        for r in resumen_global:
            writer.writerow([
                r["nombre"],
                r["n"],
                r["m"],
                r["mejor_valor"],
                r["valor_optimo"],
                r["gap_pct"] if r["gap_pct"] is not None else "",
                r["n_switches"],
            ])
    print(f"  [csv] Resumen batch guardado en '{csv_path}'")

    # Guardar resumen global en Markdown (.md)
    md_path = os.path.join(batch_dir, "resumen_batch.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Resumen de Ejecución por Lotes - {timestamp}\n\n")
        f.write(f"- **Total de Instancias:** {len(instancias)}\n")
        f.write(f"- **Tiempo Máximo por Instancia:** {tiempo_max} segundos\n\n")
        f.write("## Características de las Instancias y Resultados\n\n")
        f.write("| # | Instancia | N (Variables) | M (Restricciones) | Mejor Valor | Óptimo Conocido | Gap % | Switches |\n")
        f.write("|---|-----------|---------------|-------------------|-------------|-----------------|-------|----------|\n")
        for i, r in enumerate(resumen_global, 1):
            gap_str = f"{r['gap_pct']:.2f}%" if r["gap_pct"] is not None else "N/A"
            f.write(f"| {i} | `{r['nombre']}` | {r['n']} | {r['m']} | {r['mejor_valor']:.1f} | {r['valor_optimo']:.1f} | {gap_str} | {r['n_switches']} |\n")
    print(f"  [md] Resumen batch guardado en '{md_path}'")

    print(f"\n  BATCH COMPLETADO. ({len(instancias)} instancias procesadas)\n")


if __name__ == "__main__":
    main()
