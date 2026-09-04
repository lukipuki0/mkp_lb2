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
from continuous_benchmark.orchestrator import ejecutar_pipeline, ejecutar_mh_standalone, COLORES_MH
from continuous_benchmark.analisis_estadistico import realizar_analisis_estadistico
from continuous_benchmark.plots import (
    grafico_convergencia,
    grafico_dtw_delta,
    grafico_switches,
)




# ── Configuracion ─────────────────────────────────────────────────────────────

MAX_ITERS_POR_FUNCION  = 1000    # iteraciones totales por funcion por run
N_RUNS                 = 31      # repeticiones independientes por funcion
RANDOM_SEED            = 42      # Semilla global fijada para reproducibilidad (42)
OUTPUT_BASE            = os.path.join(os.path.dirname(__file__), "resultados")
DIMENSION              = 10      # dimensionalidad de las funciones

# Parámetros DTW
STAG_WINDOW      = 75
STAG_BAND        = 0
STAG_MIN_SLOPE   = 0.1
STAG_PLATEAU_MAX = 15
STAG_PATIENCE    = 25
STAG_USE_DDTW    = True
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
               label=f"Known Optimum = {valor_opt:.4f}")

    mu  = np.mean(valores)
    med = np.median(valores)
    ax.scatter([1], [mu],  color="#FFEB3B", zorder=5, s=60, label=f"Mean = {mu:.4f}")

    ax.set_title(f"{N_RUNS} Runs Distribution\n{func_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Hybrid DTW Pipeline", fontsize=10)
    ax.set_ylabel("Final Best Value (Minimization)", fontsize=10)
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
    cec_label  : str = "",
) -> dict:
    """Ejecuta el pipeline n_runs veces para una funcion CEC y compara vs cada MH standalone."""
    os.makedirs(output_dir, exist_ok=True)
    if not cec_label:
        cec_label = func.name

    valores_finales   : list[float] = []
    n_switches_runs   : list[int]   = []
    resultados_runs   : list        = []

    for run_idx in range(1, n_runs + 1):
        if RANDOM_SEED is not None:
            run_seed = RANDOM_SEED + run_idx
            import random
            random.seed(run_seed)
            np.random.seed(run_seed)

        print(f"\n  --- Run {run_idx:2d}/{n_runs} | {cec_label} ({func.name}) ---", flush=True)

        resultado = ejecutar_pipeline(
            func      = func,
            max_iters = max_iters,
            stag_cfg  = stag_cfg,
            verbose   = True,
        )
        valores_finales.append(resultado.mejor_valor_global)
        n_switches_runs.append(resultado.n_switches)
        resultados_runs.append(resultado)

    # ── Estadisticas descriptivas ─────────────────────────────────────────
    vals = np.array(valores_finales)
    media   = float(np.mean(vals))
    std     = float(np.std(vals))
    mediana = float(np.median(vals))
    mejor   = float(np.min(vals))
    peor    = float(np.max(vals))

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  RESUMEN {n_runs} RUNS HYBRID DTW - {cec_label} ({func.name})")
    print(sep)
    print(f"  Media   : {media:.6f}")
    print(f"  Std     : {std:.6f}")
    print(f"  Mediana : {mediana:.6f}")
    print(f"  Mejor   : {mejor:.6f}")
    print(f"  Peor    : {peor:.6f}")
    print(f"  Optimo  : {func.optimum:.6f}")
    print()

    # ── 2. Ejecutar Metaheurísticas Standalone Comparativas (N_RUNS cada una) ──
    standalone_mhs = ["PSO", "GWO", "WOA", "EHO", "ACO"]
    resultados_dict = {"Hybrid DTW": valores_finales}

    print(f"\n{sep}")
    print(f"  EJECUTANDO BENCHMARKS STANDALONE PARA ANÁLISIS ESTADÍSTICO ({n_runs} RUNS POR MH) — {cec_label}")
    print(sep)

    for mh in standalone_mhs:
        print(f"  > Ejecutando {mh} Standalone ({n_runs} runs x {max_iters} iters)...")
        vals_mh = []
        for r in range(1, n_runs + 1):
            if RANDOM_SEED is not None:
                run_seed = RANDOM_SEED + r
                import random
                random.seed(run_seed)
                np.random.seed(run_seed)
            res_std = ejecutar_mh_standalone(func, mh, max_iters=max_iters)
            vals_mh.append(res_std.mejor_valor)
        resultados_dict[mh] = vals_mh
        print(f"    {mh:<4s} | Media: {np.mean(vals_mh):.6f} | Min: {np.min(vals_mh):.6f} | Max: {np.max(vals_mh):.6f}")

    # ── Boxplot ───────────────────────────────────────────────────────────
    grafico_boxplot_runs(
        func_name  = cec_label,
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

    # ── Identificar el MEJOR RUN entre los N_RUNS (Minimización) ─────────
    best_run_idx = int(np.argmin(valores_finales))
    best_res     = resultados_runs[best_run_idx]

    best_run_dir = os.path.join(output_dir, f"best_run_{best_run_idx + 1:02d}")
    os.makedirs(best_run_dir, exist_ok=True)

    # Reporte TXT del mejor run
    report_path = os.path.join(best_run_dir, "resumen_pipeline_best_run.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Funcion            : {cec_label} ({func.name}) (Mejor Run #{best_run_idx + 1:02d})\n")
        f.write(f"Dimension          : {func.n_dim}\n")
        f.write(f"Limites            : [{func.lb}, {func.ub}]\n")
        f.write(f"Optimo conocido    : {func.optimum:.6f}\n")
        f.write(f"Mejor valor global : {best_res.mejor_valor_global:.6f}\n")
        f.write(f"Total switches     : {best_res.n_switches}\n\n")
        for i, sw in enumerate(best_res.log_switches, 1):
            f.write(f"{i}. {sw.mh_nombre} ({sw.tipo}) | mejor={sw.mejor_valor:.6f}"
                    f" | {sw.t_inicio:.1f}s-{sw.t_fin:.1f}s | iters={sw.n_iters}\n")

    # CSV y TXT Detallado DTW del Mejor Run
    csv_best_path = os.path.join(best_run_dir, "historial_dtw.csv")
    txt_best_path = os.path.join(best_run_dir, "historial_dtw_detalle.txt")

    deltas          = best_res.dtw_deltas_global
    inst_hist       = getattr(best_res, 'historial_inst_global', []) or []
    dtw_info_global = getattr(best_res, 'dtw_info_global', []) or []

    with open(csv_best_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteracion", "epoch", "mh", "tipo", "fitness_best", "fitness_instantaneo",
            "dtw_ready", "dtw_fire", "D1_vs_ramp", "D2_vs_const", "dtw_delta",
            "theta_c", "theta_r", "theta_delta", "no_improve_len", "trigger_streak",
            "window_n", "estado_dtw"
        ])
        offset = 0
        for ep_idx, sw in enumerate(best_res.log_switches, 1):
            n_seg = sw.n_iters
            for i_local in range(n_seg):
                idx = offset + i_local
                if idx >= len(best_res.historial_global):
                    break
                fit  = best_res.historial_global[idx]
                fi   = inst_hist[idx] if idx < len(inst_hist) else float("nan")
                info = dtw_info_global[idx] if idx < len(dtw_info_global) else {}

                ready = info.get("ready", False)
                fire  = info.get("fire", False)
                d1    = info.get("D1_vs_ramp", float("nan"))
                d2    = info.get("D2_vs_const", float("nan"))
                delta = info.get("delta", deltas[idx] if idx < len(deltas) else float("nan"))
                tc    = info.get("theta_c", float("nan"))
                tr    = info.get("theta_r", float("nan"))
                td    = info.get("theta_delta", float("nan"))
                no_imp= info.get("no_improve_len", 0)
                streak= info.get("trigger_streak", 0)
                win_n = info.get("n", 0)

                if not ready: estado_str = "Llenando ventana"
                elif fire:    estado_str = "ESTANCAMIENTO (Fire)"
                else:         estado_str = "Explotacion activa"

                writer.writerow([
                    idx + 1, ep_idx, sw.mh_nombre, sw.tipo, fit,
                    "" if (isinstance(fi, float) and np.isnan(fi)) else fi,
                    ready, fire,
                    "" if (isinstance(d1, float) and np.isnan(d1)) else d1,
                    "" if (isinstance(d2, float) and np.isnan(d2)) else d2,
                    "" if (isinstance(delta, float) and np.isnan(delta)) else delta,
                    "" if (isinstance(tc, float) and np.isnan(tc)) else tc,
                    "" if (isinstance(tr, float) and np.isnan(tr)) else tr,
                    "" if (isinstance(td, float) and np.isnan(td)) else td,
                    no_imp, streak, win_n, estado_str
                ])
            offset += n_seg

    with open(txt_best_path, "w", encoding="utf-8") as f_dtw:
        f_dtw.write("======================================================================================================================================================================================\n")
        f_dtw.write(f"  HISTORIAL DETALLADO DE TODAS LAS MÉTRICAS DTW POR ITERACIÓN — MEJOR RUN (#{best_run_idx + 1:02d}) — {cec_label} ({func.name})\n")
        f_dtw.write("======================================================================================================================================================================================\n\n")
        f_dtw.write(
            f"  {'Iter':>6}  {'Epoch':>5}  {'MH':<6}  {'Tipo':<12}  {'Fit Best':>14}  {'Fit Inst':>14}  "
            f"{'Ready':>6}  {'Fire':>6}  {'D1 (Ramp)':>14}  {'D2 (Const)':>14}  {'Delta DTW':>14}  "
            f"{'theta_c':>12}  {'theta_r':>12}  {'theta_delta':>14}  {'no_imp':>6}  {'streak':>6}  {'n_win':>6}  {'Estado DTW':<24}\n"
        )
        f_dtw.write("  " + "-" * 210 + "\n")

        offset = 0
        for ep_idx, sw in enumerate(best_res.log_switches, 1):
            n_seg = sw.n_iters
            for i_local in range(n_seg):
                idx = offset + i_local
                if idx >= len(best_res.historial_global):
                    break
                fit  = best_res.historial_global[idx]
                fi   = inst_hist[idx] if idx < len(inst_hist) else float("nan")
                info = dtw_info_global[idx] if idx < len(dtw_info_global) else {}

                ready = info.get("ready", False)
                fire  = info.get("fire", False)
                d1    = info.get("D1_vs_ramp", float("nan"))
                d2    = info.get("D2_vs_const", float("nan"))
                delta = info.get("delta", deltas[idx] if idx < len(deltas) else float("nan"))
                tc    = info.get("theta_c", float("nan"))
                tr    = info.get("theta_r", float("nan"))
                td    = info.get("theta_delta", float("nan"))
                no_imp= info.get("no_improve_len", 0)
                streak= info.get("trigger_streak", 0)
                win_n = info.get("n", 0)

                fi_s = f"{fi:14.6f}" if not (isinstance(fi, float) and np.isnan(fi)) else "      --      "
                d1_s = f"{d1:14.6f}" if not (isinstance(d1, float) and np.isnan(d1)) else "      --      "
                d2_s = f"{d2:14.6f}" if not (isinstance(d2, float) and np.isnan(d2)) else "      --      "
                dl_s = f"{delta:+14.6f}" if not (isinstance(delta, float) and np.isnan(delta)) else "      --      "
                tc_s = f"{tc:12.6f}" if not (isinstance(tc, float) and np.isnan(tc)) else "    --    "
                tr_s = f"{tr:12.6f}" if not (isinstance(tr, float) and np.isnan(tr)) else "    --    "
                td_s = f"{td:14.6f}" if not (isinstance(td, float) and np.isnan(td)) else "      --      "

                if not ready: estado_str = "Llenando ventana (W=30)"
                elif fire:    estado_str = "[!] ESTANCAMIENTO (Fire)"
                else:         estado_str = "[OK] Explotacion activa"

                f_dtw.write(
                    f"  {idx + 1:6d}  {ep_idx:5d}  {sw.mh_nombre:<6}  {sw.tipo:<12}  {fit:14.6f}  {fi_s}  "
                    f"{str(ready):>6}  {str(fire):>6}  {d1_s}  {d2_s}  {dl_s}  "
                    f"{tc_s}  {tr_s}  {td_s}  {no_imp:6d}  {streak:6d}  {win_n:6d}  {estado_str:<24}\n"
                )
            offset += n_seg

    print(f"\n  Generando graficos y telemetría DTW del Mejor Run (#{best_run_idx + 1:02d})...")
    grafico_convergencia(
        historial_global = best_res.historial_global,
        log_switches     = best_res.log_switches,
        colores_mh       = COLORES_MH,
        valor_optimo     = best_res.valor_optimo,
        output_dir       = best_run_dir,
    )
    grafico_dtw_delta(
        dtw_deltas_global = best_res.dtw_deltas_global,
        log_switches      = best_res.log_switches,
        colores_mh        = COLORES_MH,
        output_dir        = best_run_dir,
    )
    grafico_switches(
        log_switches = best_res.log_switches,
        colores_mh   = COLORES_MH,
        output_dir   = best_run_dir,
    )

    # ── Análisis Estadístico Inferencial Comparativo (Hybrid DTW vs cada MH) ──
    realizar_analisis_estadistico(
        resultados_dict      = resultados_dict,
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
        "cec_label":    cec_label,
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

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 6))
    colores = plt.cm.tab20.colors

    bp = ax.boxplot(
        datos,
        patch_artist=True,
        tick_labels=nombres,
        widths=0.5,
        medianprops=dict(color="#FF5722", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
    )
    for patch, color in zip(bp["boxes"], colores[:n]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(f"Global Boxplot — {resumen_global[0]['n_runs']} Runs per CEC2022 Function",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Best Value (Minimization)", fontsize=10)
    ax.set_xlabel("CEC2022 Functions", fontsize=10)
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

    dtw_mode  = "ddtw" if STAG_USE_DDTW else "dtw"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(OUTPUT_BASE, f"run_{dtw_mode}_{timestamp}")
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
        cec_label = f"CEC {idx}"
        print(f"\n{'=' * 62}")
        print(f"  [{idx}/{len(funciones)}] {cec_label}: {func.name} (Dim={func.n_dim})")
        print(f"{'=' * 62}")

        func_dir = os.path.join(batch_dir, f"CEC_{idx:02d}_{func.name}")

        resumen = procesar_funcion(
            func       = func,
            max_iters  = MAX_ITERS_POR_FUNCION,
            n_runs     = N_RUNS,
            stag_cfg   = stag_cfg,
            output_dir = func_dir,
            cec_label  = cec_label,
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
