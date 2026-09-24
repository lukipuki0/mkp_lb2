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


def resultado_frente_optimo(valor: float, optimo: float) -> str:
    """Clasifica un resultado CEC de minimización frente al óptimo conocido.

    ``isclose`` evita declarar como fallo diferencias causadas únicamente por
    precisión de punto flotante (por ejemplo, 300.00000000000006 frente a 300).
    """
    if np.isclose(valor, optimo, rtol=1e-9, atol=1e-8):
        return "Óptimo alcanzado"
    if valor > optimo:
        return "Peor que el óptimo"
    return "Mejor que el óptimo de referencia (revisar)"


def significancia_estadistica_funcion(batch_dir: str, indice_cec: int) -> str:
    """Resume los cinco Wilcoxon-Holm desde la perspectiva del híbrido."""
    prefijo = f"CEC_{indice_cec:02d}_"
    carpetas = sorted(
        nombre for nombre in os.listdir(batch_dir)
        if nombre.startswith(prefijo)
        and os.path.isdir(os.path.join(batch_dir, nombre))
    )
    if not carpetas:
        return "No disponible"

    csv_path = os.path.join(
        batch_dir, carpetas[0], "analisis_estadistico_pvalues.csv"
    )
    if not os.path.exists(csv_path):
        return "No disponible"

    with open(csv_path, newline="", encoding="utf-8") as f:
        filas = [
            fila for fila in csv.DictReader(f)
            if not fila.get("algoritmo", "").startswith("Hybrid")
        ]

    favorables = sum(
        fila.get("significancia", "").startswith("Mejor") for fila in filas
    )
    desfavorables = sum(
        fila.get("significancia", "").startswith("Peor") for fila in filas
    )
    similares = len(filas) - favorables - desfavorables

    if favorables and desfavorables:
        return (
            f"Mixta significativa (+{favorables}/={similares}/-{desfavorables})"
        )
    if favorables:
        return f"Favorable significativa (+{favorables}/={similares}/-0)"
    if desfavorables:
        return f"Desfavorable significativa (+0/={similares}/-{desfavorables})"
    return f"Sin diferencia significativa (+0/={similares}/-0)"




# ── Configuracion ─────────────────────────────────────────────────────────────

MAX_ITERS_POR_FUNCION  = int(os.environ.get("CEC_MAX_ITERS", "1000"))
N_RUNS                 = int(os.environ.get("CEC_N_RUNS", "31"))
RANDOM_SEED            = 42      # Semilla global fijada para reproducibilidad (42)
OUTPUT_BASE            = os.path.join(os.path.dirname(__file__), "resultados")
DIMENSION              = int(os.environ.get("CEC_DIMENSION", "10"))

# Parámetros DTW
STAG_WINDOW      = 75
STAG_BAND        = 0
STAG_MIN_SLOPE   = 0.1
STAG_PLATEAU_MAX = 15
STAG_PATIENCE    = 25
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
    pipeline_label: str = "Hybrid DTW",
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
    ax.set_xlabel(f"{pipeline_label} Pipeline", fontsize=10)
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
    # Desviación estándar muestral, consistente con las tablas inferenciales
    # (31 corridas independientes; ddof=1).
    std     = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    mediana = float(np.median(vals))
    mejor   = float(np.min(vals))
    peor    = float(np.max(vals))

    sep = "=" * 62
    print(f"\n{sep}")
    hybrid_label = "Hybrid DDTW" if stag_cfg.use_ddtw else "Hybrid DTW"
    print(f"  RESUMEN {n_runs} RUNS {hybrid_label.upper()} - {cec_label} ({func.name})")
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
    resultados_dict = {hybrid_label: valores_finales}

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

    # Guardar las observaciones individuales de todas las variantes. Esto
    # permite auditar y recalcular posteriormente Friedman, Wilcoxon y
    # Mann--Whitney sin depender únicamente del CSV de resúmenes.
    runs_comparativos_path = os.path.join(output_dir, "runs_comparativos.csv")
    nombres_comparativos = list(resultados_dict.keys())
    with open(runs_comparativos_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", *nombres_comparativos])
        for run_i in range(n_runs):
            writer.writerow([run_i + 1, *[resultados_dict[nombre][run_i] for nombre in nombres_comparativos]])
    print(f"  [csv] Runs comparativos guardados en '{runs_comparativos_path}'")

    # ── Boxplot ───────────────────────────────────────────────────────────
    grafico_boxplot_runs(
        func_name  = cec_label,
        valores    = valores_finales,
        valor_opt  = func.optimum,
        output_dir = output_dir,
        pipeline_label = hybrid_label,
    )

    # ── CSV con todos los runs ────────────────────────────────────────────
    csv_runs_path = os.path.join(output_dir, "runs_resultados.csv")
    with open(csv_runs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run", "mejor_valor", "n_switches", "gap_optimo_pct",
            "resultado_frente_optimo",
        ])
        optimo = float(func.optimum)
        denom_optimo = abs(optimo) if optimo != 0 else 1.0
        for i, (v, ns) in enumerate(zip(valores_finales, n_switches_runs), 1):
            writer.writerow([
                i, v, ns, 100.0 * (float(v) - optimo) / denom_optimo,
                resultado_frente_optimo(float(v), optimo),
            ])
    print(f"  [csv] Resultados de runs en '{csv_runs_path}'")

    # ── Identificar el MEJOR RUN entre los N_RUNS (Minimización) ─────────
    best_run_idx = int(np.argmin(valores_finales))
    best_res     = resultados_runs[best_run_idx]
    estado_best  = resultado_frente_optimo(best_res.mejor_valor_global, func.optimum)

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
        f.write(f"Estado del mejor   : {estado_best}\n")
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

    # ── Análisis Estadístico Inferencial Comparativo (híbrido vs cada MH) ──
    realizar_analisis_estadistico(
        resultados_dict      = resultados_dict,
        output_dir           = output_dir,
        algoritmo_referencia = hybrid_label,
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
        "mejor_run":    best_run_idx + 1,
        "peor":         peor,
        "n_runs":       n_runs,
        "valores_runs": valores_finales,
    }


# ── Boxplot comparativo global (todas las funciones) ─────────────────────────

def grafico_boxplot_global(
    resumen_global : list[dict],
    output_dir     : str,
) -> None:
    """Genera un boxplot descriptivo de error normalizado por función."""
    nombres   = [r["nombre"].replace("F1_", "").replace("F2_", "").replace("F3_", "")
                 .replace("F4_", "").replace("F5_", "").replace("F6_", "")
                 .replace("F7_", "").replace("F8_", "").replace("F9_", "")
                 .replace("F10_", "").replace("F11_", "").replace("F12_", "")
                 .split("_")[0][:12]
                 for r in resumen_global]
    datos = []
    for r in resumen_global:
        opt = float(r["valor_optimo"])
        denom = abs(opt) if opt != 0 else 1.0
        datos.append([
            100.0 * (float(v) - opt) / denom for v in r["valores_runs"]
        ])
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

    ax.set_title(f"Global Descriptive Boxplot — {resumen_global[0]['n_runs']} Runs per CEC2022 Function",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized objective error / gap (%)", fontsize=10)
    ax.set_xlabel("CEC2022 Functions", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "boxplot_global.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  [png] Boxplot global guardado en '{out_path}'")


def escribir_resumen_global_descriptivo(
    resumen_global: list[dict],
    batch_dir: str,
    timestamp: str,
) -> None:
    """Escribe un resumen global descriptivo sin comparar funciones entre sí.

    Las funciones CEC2022 tienen escalas y óptimos distintos. Por eso no se
    deben pasar como si fueran algoritmos a ``realizar_analisis_estadistico``:
    el Friedman y el Wilcoxon global resultantes no representan una
    comparación válida entre MH. Las pruebas inferenciales correctas ya se
    generan dentro de cada carpeta de función, comparando el híbrido contra
    las MH standalone.
    """
    csv_path = os.path.join(batch_dir, "analisis_estadistico_global.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "funcion", "n_dim", "n_runs", "media", "std", "mediana",
            "mejor", "peor", "valor_optimo", "gap_media_pct", "gap_mejor_pct",
            "mejor_run", "resultado_frente_optimo", "significancia_estadistica",
        ])
        for indice_cec, r in enumerate(resumen_global, 1):
            opt = float(r["valor_optimo"])
            denom = abs(opt) if opt != 0 else 1.0
            writer.writerow([
                r["nombre"], r["n_dim"], r["n_runs"], r["media"], r["std"],
                r["mediana"], r["mejor"], r["peor"], opt,
                100.0 * (float(r["media"]) - opt) / denom,
                100.0 * (float(r["mejor"]) - opt) / denom,
                r.get("mejor_run", ""),
                resultado_frente_optimo(float(r["mejor"]), opt),
                significancia_estadistica_funcion(batch_dir, indice_cec),
            ])

    md_path = os.path.join(batch_dir, "analisis_estadistico_global.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Resumen global descriptivo — CEC2022\n\n")
        f.write("> Este archivo no ejecuta pruebas Friedman/Wilcoxon entre funciones. "
                "Las funciones CEC2022 tienen escalas y óptimos diferentes, "
                "por lo que tratarlas como algoritmos produciría una tabla "
                "estadísticamente mal interpretada.\n\n")
        f.write(f"- **Fecha de la corrida:** `{timestamp}`\n")
        f.write(f"- **Funciones:** `{len(resumen_global)}`\n")
        f.write(f"- **Corridas por función:** `{resumen_global[0]['n_runs']}`\n")
        f.write("- **Pruebas inferenciales correctas:** en cada carpeta `CEC_XX_*`, "
                "contra las MH standalone.\n\n")
        f.write("- **Resultado frente al óptimo:** se determina con el mejor run, "
                "no con la media de las corridas.\n\n")
        f.write("- **Significancia estadística:** resume los cinco contrastes "
                "Wilcoxon corregidos por Holm frente a los algoritmos base. "
                "El formato es `+favorables/=similares/-desfavorables`.\n\n")
        f.write("| Función | Media | Std | Mediana | Mejor | Peor | Óptimo | "
                "Gap media (%) | Gap mejor (%) | Mejor run | Resultado frente al óptimo "
                "| Significancia estadística |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for indice_cec, r in enumerate(resumen_global, 1):
            opt = float(r["valor_optimo"])
            denom = abs(opt) if opt != 0 else 1.0
            gap_media = 100.0 * (float(r["media"]) - opt) / denom
            gap_mejor = 100.0 * (float(r["mejor"]) - opt) / denom
            estado = resultado_frente_optimo(float(r["mejor"]), opt)
            f.write(
                f"| `{r['nombre']}` | {r['media']:.6f} | {r['std']:.6f} "
                f"| {r['mediana']:.6f} | {r['mejor']:.6f} | {r['peor']:.6f} "
                f"| {opt:.6f} | {gap_media:.3f} | {gap_mejor:.3f} "
                f"| {r.get('mejor_run', '')} | {estado} "
                f"| {significancia_estadistica_funcion(batch_dir, indice_cec)} |\n"
            )
        f.write("\nPara comparar DTW/DDTW contra PSO, GWO, WOA, EHO y ACO, usar los "
                "archivos `CEC_XX_*/analisis_estadistico_pvalues.md`.\n")

    print(f"  [csv] Resumen estadístico global descriptivo guardado en '{csv_path}'")
    print(f"  [md]  Resumen estadístico global descriptivo guardado en '{md_path}'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    funciones_todas = get_test_functions(DIMENSION)

    # En ejecución Slurm array se procesa una sola función por tarea. Esto
    # evita que 12 tareas repitan todo el benchmark y permite paralelizar F1-F12
    # sin compartir archivos de resultados durante la corrida.
    tarea_cec = os.environ.get("CEC_FUNCTION_ID")
    if tarea_cec is not None:
        try:
            indice_tarea = int(tarea_cec)
        except ValueError as exc:
            raise ValueError("CEC_FUNCTION_ID debe ser un entero entre 1 y 12") from exc
        if not 1 <= indice_tarea <= len(funciones_todas):
            raise ValueError(
                f"CEC_FUNCTION_ID={indice_tarea} fuera de rango 1-{len(funciones_todas)}"
            )
        funciones = [funciones_todas[indice_tarea - 1]]
    else:
        indice_tarea = None
        funciones = funciones_todas

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
    batch_dir = os.environ.get(
        "CEC_BATCH_DIR",
        os.path.join(OUTPUT_BASE, f"run_{dtw_mode}_{timestamp}"),
    )
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

    if indice_tarea is not None:
        func = funciones[0]
        cec_label = f"CEC {indice_tarea}"
        func_dir = os.path.join(batch_dir, f"CEC_{indice_tarea:02d}_{func.name}")
        print(f"\n  Procesando únicamente {cec_label}: {func.name}")
        procesar_funcion(
            func       = func,
            max_iters  = MAX_ITERS_POR_FUNCION,
            n_runs     = N_RUNS,
            stag_cfg   = stag_cfg,
            output_dir = func_dir,
            cec_label  = cec_label,
        )
        print(f"\n  CEC {indice_tarea} completada. Resultados: {func_dir}")
        return

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
        writer.writerow([
            "funcion", "n_dim", "n_runs", "media", "std", "mediana",
            "mejor", "mejor_run", "peor", "valor_optimo",
            "resultado_mejor_frente_optimo",
        ])
        for r in resumen_global:
            writer.writerow([
                r["nombre"], r["n_dim"], r["n_runs"],
                r["media"], r["std"], r["mediana"],
                r["mejor"], r.get("mejor_run", ""), r["peor"], r["valor_optimo"],
                resultado_frente_optimo(float(r["mejor"]), float(r["valor_optimo"])),
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
        f.write("| # | Funcion | Dim | Media | Std | Mediana | Mejor | Mejor run | Optimo | Estado |\n")
        f.write("|---|---------|-----|-------|-----|---------|-------|-----------|--------|--------|\n")
        for i, r in enumerate(resumen_global, 1):
            f.write(f"| {i} | `{r['nombre']}` | {r['n_dim']}"
                    f" | {r['media']:.4f} | {r['std']:.4f} | {r['mediana']:.4f}"
                    f" | {r['mejor']:.4f} | {r.get('mejor_run', '')} | {r['valor_optimo']:.4f}"
                    f" | {resultado_frente_optimo(float(r['mejor']), float(r['valor_optimo']))} |\n")
    print(f"  [md]  Resumen global guardado en '{resumen_md}'")

    # ── Resumen global descriptivo ────────────────────────────────────────
    # No se comparan F1--F12 como si fueran algoritmos. Las pruebas
    # inferenciales válidas se generan por función dentro de procesar_funcion.
    escribir_resumen_global_descriptivo(resumen_global, batch_dir, timestamp)

    print(f"\n  BENCHMARK CONTINUO COMPLETADO. ({len(funciones)} funciones x {N_RUNS} runs)\n")


if __name__ == "__main__":
    main()
