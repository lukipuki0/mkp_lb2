"""
batch_benchmark.py
------------------
Ejecución por lotes del Pipeline Híbrido de Rotación de Metaheurísticas.

Lee la configuración de instancias y ejecuta el orquestador DTW sobre cada una de forma secuencial con 31 runs independientes por instancia.
Los resultados se organizan en:

    resultados/batch_runs/run_TIMESTAMP/
        ├── mknapcb1_inst0/
        │   ├── resumen_pipeline.txt
        │   ├── resumen_pipeline_run1.txt
        │   ├── historial_dtw_run1.csv
        │   ├── runs_resultados.csv
        │   ├── boxplot_runs.png
        │   └── ... (gráficos run1)
        ├── mknapcb1_inst9/
        │   └── ...
        └── resumen_batch.txt / csv / md
"""

import os
import csv
import json
import random
import datetime

import numpy as np
import matplotlib.pyplot as plt

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
N_RUNS                   = 31     # Repeticiones independientes por instancia
TIEMPO_MAX_POR_INSTANCIA = 120
RANDOM_SEED              = None
OUTPUT_BASE              = os.path.join("resultados", "batch_runs")

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


# ── Boxplot por instancia ─────────────────────────────────────────────────────

def grafico_boxplot_runs(
    nombre_inst: str,
    valores: list[float],
    valor_opt: float,
    output_dir: str,
) -> None:
    """Genera un boxplot de los N_RUNS valores finales para una instancia MKP."""
    fig, ax = plt.subplots(figsize=(6, 5))

    bp = ax.boxplot(
        valores,
        patch_artist=True,
        medianprops=dict(color="#FF5722", linewidth=2.5),
        boxprops=dict(facecolor="#1565C0", alpha=0.7),
        flierprops=dict(marker="o", color="#FF9800", markersize=5),
        whiskerprops=dict(color="#90CAF9", linewidth=1.5),
        capprops=dict(color="#90CAF9", linewidth=2),
    )

    if valor_opt > 0:
        ax.axhline(valor_opt, color="#4CAF50", linestyle="--", linewidth=1.5,
                   label=f"Óptimo conocido = {valor_opt:.1f}")

    mu = np.mean(valores)
    ax.scatter([1], [mu], color="#FFEB3B", zorder=5, s=60, label=f"Media = {mu:.1f}")

    ax.set_title(f"Distribución de {N_RUNS} Runs\n{nombre_inst}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Pipeline Híbrido DTW", fontsize=10)
    ax.set_ylabel("Mejor valor final (maximización)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "boxplot_runs.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [png] Boxplot guardado en '{out_path}'")


def procesar_instancia(
    inst: MKPInstance,
    nombre: str,
    tiempo_max: float,
    n_runs: int,
    stag_cfg: StagnationConfig,
    output_dir: str,
    verbose: bool = True,
) -> dict:
    """Ejecuta el pipeline híbrido n_runs veces sobre una instancia y guarda artefactos.

    Returns
    -------
    dict con estadísticas descriptivas de los runs.
    """
    os.makedirs(output_dir, exist_ok=True)

    valores_finales: list[float] = []
    n_switches_runs: list[int] = []
    gaps_runs: list[float] = []

    for run_idx in range(1, n_runs + 1):
        if verbose:
            print(f"\n  --- Run {run_idx:2d}/{n_runs} | {nombre} ---", flush=True)

        resultado = ejecutar_pipeline(
            inst       = inst,
            tiempo_max = tiempo_max,
            stag_cfg   = stag_cfg,
            verbose    = verbose,
        )
        valores_finales.append(resultado.mejor_valor_global)
        n_switches_runs.append(resultado.n_switches)

        if inst.valor_optimo > 0:
            gap = 100.0 * (inst.valor_optimo - resultado.mejor_valor_global) / inst.valor_optimo
            gaps_runs.append(gap)

        # Artefactos detallados (gráficos/CSV individual) para el Run 1
        if run_idx == 1:
            report_path = os.path.join(output_dir, "resumen_pipeline_run1.txt")
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

            csv_path = os.path.join(output_dir, "historial_dtw_run1.csv")
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

            print("\n  Generando graficos (Run 1)...")
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

    # ── Estadísticas descriptivas de N_RUNS ───────────────────────────────
    vals = np.array(valores_finales)
    media   = float(np.mean(vals))
    std     = float(np.std(vals))
    mediana = float(np.median(vals))
    mejor   = float(np.max(vals))
    peor    = float(np.min(vals))
    gap_medio = float(np.mean(gaps_runs)) if gaps_runs else None
    gap_mejor = float(100.0 * (inst.valor_optimo - mejor) / inst.valor_optimo) if inst.valor_optimo > 0 else None
    switches_medio = float(np.mean(n_switches_runs))

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  RESUMEN {n_runs} RUNS - {nombre}")
    print(sep)
    print(f"  Media           : {media:.1f}")
    print(f"  Std             : {std:.2f}")
    print(f"  Mediana         : {mediana:.1f}")
    print(f"  Mejor           : {mejor:.1f}")
    print(f"  Peor            : {peor:.1f}")
    print(f"  Óptimo          : {inst.valor_optimo:.1f}")
    if gap_medio is not None:
        print(f"  Gap medio       : {gap_medio:.2f}%")
        print(f"  Gap mejor       : {gap_mejor:.2f}%")
    print(f"  Switches medios : {switches_medio:.1f}")
    print()

    # ── Boxplot ───────────────────────────────────────────────────────────
    grafico_boxplot_runs(
        nombre_inst = nombre,
        valores     = valores_finales,
        valor_opt   = inst.valor_optimo,
        output_dir  = output_dir,
    )

    # ── CSV con todos los runs ────────────────────────────────────────────
    csv_runs_path = os.path.join(output_dir, "runs_resultados.csv")
    with open(csv_runs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "mejor_valor", "gap_pct", "n_switches"])
        for i, (v, ns) in enumerate(zip(valores_finales, n_switches_runs), 1):
            g_str = f"{(100.0 * (inst.valor_optimo - v) / inst.valor_optimo):.2f}" if inst.valor_optimo > 0 else ""
            writer.writerow([i, v, g_str, ns])
    print(f"  [csv] Resultados de runs en '{csv_runs_path}'")

    # ── Reporte TXT consolidado ───────────────────────────────────────────
    report_path = os.path.join(output_dir, "resumen_pipeline.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Instancia          : {nombre}\n")
        f.write(f"Items / Restricc.  : {inst.n} / {inst.m}\n")
        f.write(f"Runs ejecutados    : {n_runs}\n")
        f.write(f"Optimo conocido    : {inst.valor_optimo:.1f}\n")
        f.write(f"Media valor        : {media:.1f}\n")
        f.write(f"Desviacion est.    : {std:.2f}\n")
        f.write(f"Mediana valor      : {mediana:.1f}\n")
        f.write(f"Mejor valor        : {mejor:.1f}\n")
        f.write(f"Peor valor         : {peor:.1f}\n")
        if gap_medio is not None:
            f.write(f"Gap medio          : {gap_medio:.2f}%\n")
            f.write(f"Gap mejor          : {gap_mejor:.2f}%\n")
        f.write(f"Switches medios    : {switches_medio:.1f}\n")

    return {
        "nombre":         nombre,
        "n":              inst.n,
        "m":              inst.m,
        "media":          media,
        "std":            std,
        "mediana":        mediana,
        "mejor":          mejor,
        "peor":           peor,
        "valor_optimo":   inst.valor_optimo,
        "gap_medio":      gap_medio,
        "gap_mejor":      gap_mejor,
        "switches_medio": switches_medio,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    mknapcb_num = MKNAPCB_NUM
    tiempo_max = TIEMPO_MAX_POR_INSTANCIA

    if mknapcb_num < 1 or mknapcb_num > 9:
        print(f"\n[!] Número de mknapcb inválido ({mknapcb_num}). Debe ser de 1 a 9. Usando 1.")
        mknapcb_num = 1

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

    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir  = os.path.join(OUTPUT_BASE, f"run_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    banner = "=" * 62
    print(f"\n{banner}")
    print("  BATCH BENCHMARK - Pipeline Hibrido DTW (Multi-Run)")
    print(banner)
    print(f"  Instancias a procesar : {len(instancias)}")
    print(f"  Runs por instancia    : {N_RUNS}")
    print(f"  Tiempo max / run      : {tiempo_max}s")
    print(f"  Carpeta de salida     : {batch_dir}")
    print(banner)

    cache_urls: dict[str, list] = {}
    resumen_global: list[dict] = []

    for idx, entry in enumerate(instancias, 1):
        url    = entry["url"]
        index  = entry["index"]
        nombre = entry.get("nombre", f"inst_{idx}")

        print(f"\n{'-' * 62}")
        print(f"  [{idx}/{len(instancias)}] {nombre}  (url=...{url[-15:]}, index={index})")
        print(f"{'-' * 62}")

        if url not in cache_urls:
            cache_urls[url] = cargar_instancias(url)
        data = seleccionar_instancia(cache_urls[url], index)
        inst = MKPInstance.from_dict(data)

        print(f"  Instancia : {inst.n} items, {inst.m} restricciones")
        print(f"  Optimo    : {inst.valor_optimo}")

        inst_dir = os.path.join(batch_dir, nombre)

        resumen = procesar_instancia(
            inst       = inst,
            nombre     = nombre,
            tiempo_max = tiempo_max,
            n_runs     = N_RUNS,
            stag_cfg   = stag_cfg,
            output_dir = inst_dir,
            verbose    = True,
        )
        resumen_global.append(resumen)

    # ── Resumen global del batch ──────────────────────────────────────────
    banner_l = "=" * 90
    print(f"\n\n{banner_l}")
    print("  RESUMEN GLOBAL DEL BATCH")
    print(banner_l)
    print(f"  {'#':<3} {'Instancia':<18} {'N':>5} {'M':>3} {'Media':>8} {'Std':>6} {'Mejor':>8} {'Optimo':>8} {'GapMed%':>8} {'GapMej%':>8} {'Sw(med)':>7}")
    print("  " + "-" * 88)
    for i, r in enumerate(resumen_global, 1):
        g_med_str = f"{r['gap_medio']:.2f}" if r["gap_medio"] is not None else "N/A"
        g_mej_str = f"{r['gap_mejor']:.2f}" if r["gap_mejor"] is not None else "N/A"
        print(f"  {i:<3} {r['nombre']:<18} {r['n']:>5} {r['m']:>3} {r['media']:>8.1f}"
              f" {r['std']:>6.1f} {r['mejor']:>8.1f} {r['valor_optimo']:>8.1f}"
              f" {g_med_str:>8} {g_mej_str:>8} {r['switches_medio']:>7.1f}")
    print(banner_l)

    # Guardar resumen global en TXT
    resumen_path = os.path.join(batch_dir, "resumen_batch.txt")
    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write("RESUMEN GLOBAL DEL BATCH\n")
        f.write(f"Fecha       : {timestamp}\n")
        f.write(f"Instancias  : {len(instancias)}\n")
        f.write(f"Runs/inst   : {N_RUNS}\n")
        f.write(f"Tiempo/run  : {tiempo_max}s\n\n")
        f.write(f"{'#':<3} {'Instancia':<18} {'N':>5} {'M':>3} {'Media':>8} {'Std':>6} {'Mejor':>8} {'Optimo':>8} {'GapMed%':>8} {'GapMej%':>8} {'Sw(med)':>7}\n")
        f.write("-" * 90 + "\n")
        for i, r in enumerate(resumen_global, 1):
            g_med_str = f"{r['gap_medio']:.2f}" if r["gap_medio"] is not None else "N/A"
            g_mej_str = f"{r['gap_mejor']:.2f}" if r["gap_mejor"] is not None else "N/A"
            f.write(f"{i:<3} {r['nombre']:<18} {r['n']:>5} {r['m']:>3} {r['media']:>8.1f}"
                    f" {r['std']:>6.1f} {r['mejor']:>8.1f} {r['valor_optimo']:>8.1f}"
                    f" {g_med_str:>8} {g_mej_str:>8} {r['switches_medio']:>7.1f}\n")
    print(f"\n  [txt] Resumen batch guardado en '{resumen_path}'")

    # Guardar resumen global en CSV
    csv_path = os.path.join(batch_dir, "resumen_batch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instancia", "n", "m", "media", "std", "mediana", "mejor", "peor", "valor_optimo", "gap_medio_pct", "gap_mejor_pct", "switches_medio"])
        for r in resumen_global:
            writer.writerow([
                r["nombre"],
                r["n"],
                r["m"],
                r["media"],
                r["std"],
                r["mediana"],
                r["mejor"],
                r["peor"],
                r["valor_optimo"],
                r["gap_medio"] if r["gap_medio"] is not None else "",
                r["gap_mejor"] if r["gap_mejor"] is not None else "",
                r["switches_medio"],
            ])
    print(f"  [csv] Resumen batch guardado en '{csv_path}'")

    # Guardar resumen global en Markdown (.md)
    md_path = os.path.join(batch_dir, "resumen_batch.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Resumen de Ejecución por Lotes - {timestamp}\n\n")
        f.write(f"- **Total de Instancias:** {len(instancias)}\n")
        f.write(f"- **Runs por Instancia:** {N_RUNS}\n")
        f.write(f"- **Tiempo Máximo por Run:** {tiempo_max} segundos\n\n")
        f.write("## Características de las Instancias y Resultados\n\n")
        f.write("| # | Instancia | N | M | Media | Std | Mejor | Peor | Óptimo | Gap Medio % | Gap Mejor % | Switches (med) |\n")
        f.write("|---|-----------|---|---|-------|-----|-------|------|--------|-------------|-------------|----------------|\n")
        for i, r in enumerate(resumen_global, 1):
            g_med_str = f"{r['gap_medio']:.2f}%" if r["gap_medio"] is not None else "N/A"
            g_mej_str = f"{r['gap_mejor']:.2f}%" if r["gap_mejor"] is not None else "N/A"
            f.write(f"| {i} | `{r['nombre']}` | {r['n']} | {r['m']} | {r['media']:.1f} | {r['std']:.2f} | {r['mejor']:.1f} | {r['peor']:.1f} | {r['valor_optimo']:.1f} | {g_med_str} | {g_mej_str} | {r['switches_medio']:.1f} |\n")
    print(f"  [md] Resumen batch guardado en '{md_path}'")

    print(f"\n  BATCH COMPLETADO. ({len(instancias)} instancias x {N_RUNS} runs procesados)\n")


if __name__ == "__main__":
    main()

