"""
continuous_benchmark/benchmark_continuo.py
------------------------------------------
Script principal: ejecuta el Pipeline Hibrido DTW sobre TODAS las funciones
continuas CEC2022 con N_RUNS repeticiones independientes.

Por cada funcion se genera:
  - subcarpeta con artefactos del run 1 (convergencia, DTW, switches)
  - CSV con los resultados de los N_RUNS runs
  - Boxplot de los valores finales obtenidos en cada run

Al final se genera un resumen global (TXT, CSV, MD) con estadisticas
descriptivas (media, std, mediana, min, max) por funcion.

Uso:
    python -m continuous_benchmark.benchmark_continuo
"""

import os
import csv
import random
import datetime
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Agregar raiz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dtw_stagnation import StagnationConfig
from continuous_benchmark.funciones_cec2022 import get_test_functions, ContinuousFunction
from continuous_benchmark.orchestrator import ejecutar_pipeline, COLORES_MH
from analisis_estadistico import realizar_analisis_estadistico
from continuous_benchmark.plots import (
    grafico_convergencia,
    grafico_dtw_delta,
    grafico_switches,
)




# ── Configuracion ─────────────────────────────────────────────────────────────

MAX_ITERS_POR_FUNCION  = 1000    # iteraciones totales por funcion por run
N_RUNS                 = 31      # repeticiones independientes por funcion
RANDOM_SEED            = None    # None -> semilla aleatoria en cada run
OUTPUT_BASE            = os.path.join("resultados", "benchmark_continuo")
DIMENSION              = 10      # dimensionalidad de las funciones

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


# ── Boxplot por funcion ───────────────────────────────────────────────────────

def grafico_boxplot_runs(
    func_name  : str,
    valores    : list[float],
    valor_opt  : float,
    output_dir : str,
) -> None:
    """Genera un boxplot de los N_RUNS valores finales para una funcion."""
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

    ax.axhline(valor_opt, color="#4CAF50", linestyle="--", linewidth=1.5,
               label=f"Optimo conocido = {valor_opt:.4f}")

    mu  = np.mean(valores)
    med = np.median(valores)
    ax.scatter([1], [mu],  color="#FFEB3B", zorder=5, s=60, label=f"Media = {mu:.4f}")

    ax.set_title(f"Distribucion de {N_RUNS} Runs\n{func_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Pipeline Hibrido DTW", fontsize=10)
    ax.set_ylabel("Mejor valor final (minimizacion)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "boxplot_runs.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [png] Boxplot guardado en '{out_path}'")


# ── Procesar una funcion (N_RUNS veces) ───────────────────────────────────────

def procesar_funcion(
    func       : ContinuousFunction,
    max_iters  : int,
    n_runs     : int,
    stag_cfg   : StagnationConfig,
    output_dir : str,
) -> dict:
    """Ejecuta el pipeline n_runs veces para una funcion CEC y guarda artefactos."""
    os.makedirs(output_dir, exist_ok=True)

    valores_finales   : list[float] = []
    n_switches_runs   : list[int]   = []
    resultado_run1    = None

    for run_idx in range(1, n_runs + 1):
        print(f"\n  --- Run {run_idx:2d}/{n_runs} | {func.name} ---", flush=True)

        resultado = ejecutar_pipeline(
            func      = func,
            max_iters = max_iters,
            stag_cfg  = stag_cfg,
            verbose   = True,
        )
        valores_finales.append(resultado.mejor_valor_global)
        n_switches_runs.append(resultado.n_switches)

        # Guardar artefactos solo del primer run
        if run_idx == 1:
            resultado_run1 = resultado

    # ── Estadisticas descriptivas ─────────────────────────────────────────
    vals = np.array(valores_finales)
    media   = float(np.mean(vals))
    std     = float(np.std(vals))
    mediana = float(np.median(vals))
    mejor   = float(np.min(vals))
    peor    = float(np.max(vals))

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  RESUMEN {n_runs} RUNS - {func.name}")
    print(sep)
    print(f"  Media   : {media:.6f}")
    print(f"  Std     : {std:.6f}")
    print(f"  Mediana : {mediana:.6f}")
    print(f"  Mejor   : {mejor:.6f}")
    print(f"  Peor    : {peor:.6f}")
    print(f"  Optimo  : {func.optimum:.6f}")
    print()

    # ── Boxplot ───────────────────────────────────────────────────────────
    grafico_boxplot_runs(
        func_name  = func.name,
        valores    = valores_finales,
        valor_opt  = func.optimum,
        output_dir = output_dir,
    )

    # ── CSV con todos los runs ────────────────────────────────────────────
    csv_runs_path = os.path.join(output_dir, "runs_resultados.csv")
    with open(csv_runs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "mejor_valor", "n_switches"])
        for i, (v, ns) in enumerate(zip(valores_finales, n_switches_runs), 1):
            writer.writerow([i, v, ns])
    print(f"  [csv] Resultados de runs en '{csv_runs_path}'")

    # ── Reporte TXT del run 1 + estadisticas ─────────────────────────────
    report_path = os.path.join(output_dir, "resumen_pipeline.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Funcion            : {func.name}\n")
        f.write(f"Dimension          : {func.n_dim}\n")
        f.write(f"Limites            : [{func.lb}, {func.ub}]\n")
        f.write(f"N_Runs             : {n_runs}\n")
        f.write(f"Max Iters / Run    : {max_iters}\n\n")
        f.write(f"Optimo conocido    : {func.optimum:.6f}\n\n")
        f.write(f"--- Estadisticas descriptivas ({n_runs} runs) ---\n")
        f.write(f"  Media   : {media:.6f}\n")
        f.write(f"  Std     : {std:.6f}\n")
        f.write(f"  Mediana : {mediana:.6f}\n")
        f.write(f"  Mejor   : {mejor:.6f}\n")
        f.write(f"  Peor    : {peor:.6f}\n\n")
        f.write("--- Valores por run ---\n")
        for i, v in enumerate(valores_finales, 1):
            f.write(f"  Run {i:2d}: {v:.6f}\n")
    print(f"  [txt] {report_path}")

    # ── Graficos del Run 1 ────────────────────────────────────────────────
    if resultado_run1 is not None:
        run1_dir = os.path.join(output_dir, "run_01_graficos")
        os.makedirs(run1_dir, exist_ok=True)
        print("\n  Generando graficos del Run 1...")
        grafico_convergencia(
            historial_global = resultado_run1.historial_global,
            log_switches     = resultado_run1.log_switches,
            colores_mh       = COLORES_MH,
            valor_optimo     = resultado_run1.valor_optimo,
            output_dir       = run1_dir,
        )
        grafico_dtw_delta(

            dtw_deltas_global = resultado_run1.dtw_deltas_global,
            log_switches      = resultado_run1.log_switches,
            colores_mh        = COLORES_MH,
            output_dir        = run1_dir,
        )
        grafico_switches(
            log_switches = resultado_run1.log_switches,
            colores_mh   = COLORES_MH,
            output_dir   = run1_dir,
        )

    # ── Análisis Estadístico de los N_RUNS del Pipeline Híbrido ─────────────
    realizar_analisis_estadistico(
        resultados_dict      = {"Hybrid DTW": valores_finales},
        output_dir           = output_dir,
        algoritmo_referencia = "Hybrid DTW",
        metrica_label        = f"Fitness — {func.name} (Minimización)",
        titulo_benchmark     = f"CEC2022 — {func.name}",
        minimizacion         = True,
        boxplot_filename     = "boxplot_estadistico.png",
        csv_filename         = "analisis_estadistico_pvalues.csv",
        md_filename          = "analisis_estadistico_pvalues.md",
    )

    return {
        "nombre":       func.name,
        "n_dim":        func.n_dim,
        "valor_optimo": func.optimum,
        "media":        media,
        "std":          std,
        "mediana":      mediana,
        "mejor":        mejor,
        "peor":         peor,
        "n_runs":       n_runs,
        "valores_runs": valores_finales,
    }


# ── Boxplot comparativo global (todas las funciones) ─────────────────────────

def grafico_boxplot_global(
    resumen_global : list[dict],
    output_dir     : str,
) -> None:
    """Genera un boxplot multi-funcion comparando los N_RUNS runs de cada una."""
    nombres   = [r["nombre"].replace("F1_", "").replace("F2_", "").replace("F3_", "")
                 .replace("F4_", "").replace("F5_", "").replace("F6_", "")
                 .replace("F7_", "").replace("F8_", "").replace("F9_", "")
                 .replace("F10_", "").replace("F11_", "").replace("F12_", "")
                 .split("_")[0][:12]
                 for r in resumen_global]
    datos = [r["valores_runs"] for r in resumen_global]
    n = len(datos)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 6))
    colores = plt.cm.tab20.colors

    bp = ax.boxplot(
        datos,
        patch_artist=True,
        medianprops=dict(color="#FF5722", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
    )
    for patch, color in zip(bp["boxes"], colores[:n]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(nombres, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"Boxplot Global — {resumen_global[0]['n_runs']} Runs por Funcion CEC2022",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Mejor valor final (minimizacion)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "boxplot_global.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  [png] Boxplot global guardado en '{out_path}'")


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
    print("  CONTINUOUS BENCHMARK - Pipeline Hibrido DTW (Multi-Run)")
    print(banner)
    print(f"  Funciones a procesar  : {len(funciones)}")
    for fn in funciones:
        print(f"    - {fn.name} (Dim={fn.n_dim}, [{fn.lb}, {fn.ub}])")
    print(f"  Runs por funcion      : {N_RUNS}")
    print(f"  Max iters / run       : {MAX_ITERS_POR_FUNCION}")
    print(f"  Carpeta de salida     : {batch_dir}")
    print(banner)

    resumen_global: list[dict] = []

    for idx, func in enumerate(funciones, 1):
        print(f"\n{'=' * 62}")
        print(f"  [{idx}/{len(funciones)}] {func.name} (Dim={func.n_dim})")
        print(f"{'=' * 62}")

        func_dir = os.path.join(batch_dir, func.name)

        resumen = procesar_funcion(
            func      = func,
            max_iters = MAX_ITERS_POR_FUNCION,
            n_runs    = N_RUNS,
            stag_cfg  = stag_cfg,
            output_dir = func_dir,
        )
        resumen_global.append(resumen)

    # ── Boxplot comparativo global ────────────────────────────────────────
    grafico_boxplot_global(resumen_global, batch_dir)

    # ── Resumen global en consola ─────────────────────────────────────────
    print(f"\n\n{banner}")
    print("  RESUMEN GLOBAL DEL BATCH CONTINUO")
    print(banner)
    header = f"  {'#':<3} {'Funcion':<22} {'Dim':>4} {'Media':>14} {'Std':>12} {'Mediana':>12} {'Mejor':>12} {'Optimo':>10}"
    print(header)
    print("  " + "-" * 95)
    for i, r in enumerate(resumen_global, 1):
        print(f"  {i:<3} {r['nombre']:<22} {r['n_dim']:>4}"
              f" {r['media']:>14.4f} {r['std']:>12.4f} {r['mediana']:>12.4f}"
              f" {r['mejor']:>12.4f} {r['valor_optimo']:>10.4f}")
    print(banner)

    # ── TXT ───────────────────────────────────────────────────────────────
    resumen_txt = os.path.join(batch_dir, "resumen_global.txt")
    with open(resumen_txt, "w", encoding="utf-8") as f:
        f.write("RESUMEN GLOBAL DEL BENCHMARK CONTINUO\n")
        f.write(f"Fecha       : {timestamp}\n")
        f.write(f"Funciones   : {len(funciones)}\n")
        f.write(f"Runs/func   : {N_RUNS}\n")
        f.write(f"Iters/run   : {MAX_ITERS_POR_FUNCION}\n\n")
        f.write(f"{'#':<3} {'Funcion':<22} {'Dim':>4} {'Media':>14} {'Std':>12} {'Mediana':>12} {'Mejor':>12} {'Optimo':>10}\n")
        f.write("-" * 97 + "\n")
        for i, r in enumerate(resumen_global, 1):
            f.write(f"{i:<3} {r['nombre']:<22} {r['n_dim']:>4}"
                    f" {r['media']:>14.4f} {r['std']:>12.4f} {r['mediana']:>12.4f}"
                    f" {r['mejor']:>12.4f} {r['valor_optimo']:>10.4f}\n")
    print(f"\n  [txt] Resumen global guardado en '{resumen_txt}'")

    # ── CSV ───────────────────────────────────────────────────────────────
    resumen_csv = os.path.join(batch_dir, "resumen_global.csv")
    with open(resumen_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["funcion", "n_dim", "n_runs", "media", "std", "mediana", "mejor", "peor", "valor_optimo"])
        for r in resumen_global:
            writer.writerow([
                r["nombre"], r["n_dim"], r["n_runs"],
                r["media"], r["std"], r["mediana"],
                r["mejor"], r["peor"], r["valor_optimo"],
            ])
    print(f"  [csv] Resumen global guardado en '{resumen_csv}'")

    # ── CSV de todos los runs individuales ────────────────────────────────
    runs_csv = os.path.join(batch_dir, "todos_los_runs.csv")
    with open(runs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["funcion", "run", "mejor_valor", "valor_optimo"])
        for r in resumen_global:
            for run_i, v in enumerate(r["valores_runs"], 1):
                writer.writerow([r["nombre"], run_i, v, r["valor_optimo"]])
    print(f"  [csv] Todos los runs guardados en '{runs_csv}'")

    # ── Markdown ──────────────────────────────────────────────────────────
    resumen_md = os.path.join(batch_dir, "resumen_global.md")
    with open(resumen_md, "w", encoding="utf-8") as f:
        f.write(f"# Resumen Global — Benchmark Continuo CEC2022 ({timestamp})\n\n")
        f.write(f"- **Total funciones:** {len(funciones)}\n")
        f.write(f"- **Runs por funcion:** {N_RUNS}\n")
        f.write(f"- **Max iteraciones por run:** {MAX_ITERS_POR_FUNCION}\n\n")
        f.write("| # | Funcion | Dim | Media | Std | Mediana | Mejor | Optimo |\n")
        f.write("|---|---------|-----|-------|-----|---------|-------|--------|\n")
        for i, r in enumerate(resumen_global, 1):
            f.write(f"| {i} | `{r['nombre']}` | {r['n_dim']}"
                    f" | {r['media']:.4f} | {r['std']:.4f} | {r['mediana']:.4f}"
                    f" | {r['mejor']:.4f} | {r['valor_optimo']:.4f} |\n")
    print(f"  [md]  Resumen global guardado en '{resumen_md}'")

    # ── Análisis Estadístico Global (comparación entre todas las funciones CEC) ──
    if len(resumen_global) > 1:
        resultados_multi = {r["nombre"]: r["valores_runs"] for r in resumen_global}
        referencia_global = resumen_global[0]["nombre"]
        realizar_analisis_estadistico(
            resultados_dict      = resultados_multi,
            output_dir           = batch_dir,
            algoritmo_referencia = referencia_global,
            metrica_label        = "Fitness (Minimización CEC2022)",
            titulo_benchmark     = f"CEC2022 Global ({len(resumen_global)} funciones)",
            minimizacion         = True,
            boxplot_filename     = "boxplot_comparativo_funciones.png",
            csv_filename         = "analisis_estadistico_global.csv",
            md_filename          = "analisis_estadistico_global.md",
        )

    print(f"\n  BENCHMARK CONTINUO COMPLETADO. ({len(funciones)} funciones x {N_RUNS} runs)\n")


if __name__ == "__main__":
    main()
