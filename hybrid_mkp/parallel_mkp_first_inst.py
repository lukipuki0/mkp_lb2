"""
parallel_mkp_first_inst.py
--------------------------
Ejecución en paralelo para HPC del Pipeline Híbrido MKP:
- Ejecuta las 9 familias (mknapcb1 a mknapcb9).
- Procesa únicamente la primera instancia (index=0) de cada archivo .txt.
- Ejecuta las 9 instancias concurrentemente mediante hilos/procesos concurrentes (ThreadPoolExecutor / ProcessPoolExecutor).
- Genera reportes individuales por instancia y un resumen consolidado (TXT, CSV, MD) con análisis estadístico.

Uso:
    python parallel_mkp_first_inst.py
"""

import os
import csv
import time
import random
import datetime
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dtw_stagnation import StagnationConfig
from hybrid_mkp.mkp_core.data_loader import cargar_instancias, seleccionar_instancia
from hybrid_mkp.mkp_core.problem import MKPInstance
from hybrid_mkp.orchestrator import ejecutar_pipeline, ejecutar_mh_standalone, COLORES_MH
from hybrid_mkp.analisis_estadistico import realizar_analisis_estadistico
from hybrid_mkp.plots import (
    grafico_convergencia,
    grafico_dtw_delta,
    grafico_switches,
)


# ── Configuración para HPC ───────────────────────────────────────────────────

N_RUNS       = 31     # Repeticiones independientes por instancia
MAX_ITERS    = 3000   # Condición de parada por iteraciones
RANDOM_SEED  = 42     # Semilla global fijada para reproducibilidad

# Detectar workers automáticamente desde Slurm o CPUs disponibles
_slurm_cpus  = os.environ.get("SLURM_CPUS_PER_TASK")
MAX_WORKERS  = min(9, int(_slurm_cpus)) if _slurm_cpus else min(9, os.cpu_count() or 4)

OUTPUT_BASE  = os.environ.get("MKP_TMP_DIR", os.path.join("resultados", "parallel_hpc_first_inst"))

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


# ── Gráfico Boxplot por instancia ─────────────────────────────────────────────

def grafico_boxplot_runs(
    nombre_inst: str,
    valores: list[float],
    valor_opt: float,
    output_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [valores],
        tick_labels=["Hybrid DTW"],
        patch_artist=True,
        boxprops=dict(facecolor="#4C72B0", color="#2B4C7E", alpha=0.7),
        medianprops=dict(color="#C44E52", linewidth=2),
        whiskerprops=dict(color="#2B4C7E"),
        capprops=dict(color="#2B4C7E"),
        flierprops=dict(marker="o", color="#C44E52", alpha=0.6),
    )
    if valor_opt > 0:
        ax.axhline(
            y=valor_opt,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Óptimo ({valor_opt:.1f})",
        )
    ax.set_title(f"Distribución de Resultados ({len(valores)} runs) — {nombre_inst}", fontsize=11)
    ax.set_xlabel("Configuración", fontsize=10)
    ax.set_ylabel("Mejor Valor Obtenido", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "boxplot_runs.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


ORLIB_BKS_INST00 = {
    "mknapcb1": 24381.0,  # 5.100-00
    "mknapcb2": 59312.0,  # 5.250-00
    "mknapcb3": 120130.0, # 5.500-00
    "mknapcb4": 23064.0,  # 10.100-00
    "mknapcb5": 59187.0,  # 10.250-00
    "mknapcb6": 117726.0, # 10.500-00
    "mknapcb7": 21946.0,  # 30.100-00
    "mknapcb8": 56693.0,  # 30.250-00
    "mknapcb9": 115868.0, # 30.500-00
}


# ── Procesar una instancia completa (N_RUNS) ──────────────────────────────────

def procesar_tarea_instancia(
    inst_info: dict,
    max_iters: int,
    n_runs: int,
    stag_cfg: StagnationConfig,
    batch_dir: str,
) -> dict:
    url     = inst_info["url"]
    index   = inst_info["index"]
    nombre  = inst_info["nombre"]
    familia = inst_info["familia"]
    sub_nom = inst_info["sub_nom"]

    todas_inst = cargar_instancias(url)
    data = seleccionar_instancia(todas_inst, index)
    inst = MKPInstance.from_dict(data)

    if (inst.valor_optimo <= 0) and index == 0 and familia in ORLIB_BKS_INST00:
        inst.valor_optimo = ORLIB_BKS_INST00[familia]

    output_dir = os.path.join(batch_dir, familia, sub_nom)
    os.makedirs(output_dir, exist_ok=True)

    print(f"  [START] {nombre} ({inst.n} items, {inst.m} constr, opt={inst.valor_optimo})", flush=True)

    valores_finales: list[float] = []
    n_switches_runs: list[int] = []
    gaps_runs: list[float] = []
    tiempos_runs: list[float] = []
    resultados_runs: list = []

    for run_idx in range(1, n_runs + 1):
        if RANDOM_SEED is not None:
            run_seed = RANDOM_SEED + run_idx
            random.seed(run_seed)
            np.random.seed(run_seed)

        t_r_start = time.time()
        resultado = ejecutar_pipeline(
            inst      = inst,
            max_iters = max_iters,
            stag_cfg  = stag_cfg,
            verbose   = False,
        )
        t_r_end = time.time()
        dur_s = t_r_end - t_r_start
        tiempos_runs.append(dur_s)

        valores_finales.append(resultado.mejor_valor_global)
        n_switches_runs.append(resultado.n_switches)
        resultados_runs.append(resultado)

        if inst.valor_optimo > 0:
            gap = 100.0 * (inst.valor_optimo - resultado.mejor_valor_global) / inst.valor_optimo
            gaps_runs.append(gap)

    # Identificar el mejor run
    best_run_idx = int(np.argmax(valores_finales))
    best_res     = resultados_runs[best_run_idx]
    best_val     = valores_finales[best_run_idx]

    best_run_dir = os.path.join(output_dir, f"best_run_{best_run_idx + 1:02d}")
    os.makedirs(best_run_dir, exist_ok=True)

    # Reporte TXT del Mejor Run
    report_path = os.path.join(best_run_dir, "resumen_pipeline_best_run.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Instancia          : {nombre} (Mejor Run #{best_run_idx + 1:02d})\n")
        f.write(f"Items / Restricc.  : {inst.n} / {inst.m}\n")
        f.write(f"Mejor valor global : {best_res.mejor_valor_global:.1f}\n")
        f.write(f"Optimo conocido    : {best_res.valor_optimo:.1f}\n")
        if best_res.gap_pct is not None:
            f.write(f"Gap relativo       : {best_res.gap_pct:.2f}%\n")
        f.write(f"Total switches     : {best_res.n_switches}\n\n")
        for i, sw in enumerate(best_res.log_switches, 1):
            f.write(f"{i}. {sw.mh_nombre} ({sw.tipo}) | mejor={sw.mejor_valor:.1f}"
                    f" | {sw.t_inicio:.1f}s-{sw.t_fin:.1f}s | iters={sw.n_iters}\n")

    # CSV Historial DTW del Mejor Run
    csv_path = os.path.join(best_run_dir, "historial_dtw.csv")
    deltas   = best_res.dtw_deltas_global
    inst_hist = getattr(best_res, 'historial_inst_global', []) or []
    dtw_info_global = getattr(best_res, 'dtw_info_global', []) or []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
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

    # Gráficos del mejor run
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

    # Boxplot del pipeline
    grafico_boxplot_runs(
        nombre_inst = nombre,
        valores     = valores_finales,
        valor_opt   = inst.valor_optimo,
        output_dir  = output_dir,
    )

    # CSV de todos los runs
    csv_runs_path = os.path.join(output_dir, "runs_resultados.csv")
    with open(csv_runs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "mejor_valor", "n_switches", "gap_pct", "tiempo_s"])
        for i, (v, ns, t_s) in enumerate(zip(valores_finales, n_switches_runs, tiempos_runs), 1):
            g = 100.0 * (inst.valor_optimo - v) / inst.valor_optimo if inst.valor_optimo > 0 else ""
            writer.writerow([i, v, ns, g, f"{t_s:.2f}"])

    # Standalone comparativo
    standalone_mhs = ["GA", "PSO", "GWO", "WOA", "EHO", "ACO", "SA"]
    resultados_dict = {"Hybrid DTW": valores_finales}

    for mh in standalone_mhs:
        vals_mh = []
        for r in range(1, n_runs + 1):
            if RANDOM_SEED is not None:
                run_seed = RANDOM_SEED + r
                random.seed(run_seed)
                np.random.seed(run_seed)
            res_std = ejecutar_mh_standalone(inst, mh, max_iters=max_iters)
            vals_mh.append(res_std.mejor_valor)
        resultados_dict[mh] = vals_mh

    # Análisis estadístico inferencial por instancia
    realizar_analisis_estadistico(
        resultados_dict      = resultados_dict,
        output_dir           = output_dir,
        algoritmo_referencia = "Hybrid DTW",
        metrica_label        = f"Fitness (Maximization) — {nombre}",
        titulo_benchmark     = f"{nombre}",
        minimizacion         = False,
        boxplot_filename     = "mhs_comparative_boxplot.png",
        csv_filename         = "analisis_estadistico_pvalues.csv",
        md_filename          = "analisis_estadistico_pvalues.md",
    )

    vals = np.array(valores_finales)
    media   = float(np.mean(vals))
    std     = float(np.std(vals))
    mediana = float(np.median(vals))
    mejor   = float(np.max(vals))
    peor    = float(np.min(vals))
    gap_medio = float(np.mean(gaps_runs)) if gaps_runs else None
    gap_mejor = float(100.0 * (inst.valor_optimo - mejor) / inst.valor_optimo) if inst.valor_optimo > 0 else None
    switches_medio = float(np.mean(n_switches_runs))
    tiempo_medio   = float(np.mean(tiempos_runs)) if tiempos_runs else 0.0
    tiempo_total   = float(np.sum(tiempos_runs)) if tiempos_runs else 0.0

    gap_str = f"{gap_medio:.2f}%" if gap_medio is not None else "N/A"
    print(f"  [DONE] {nombre:<16} | Media: {media:9.1f} | Mejor: {mejor:9.1f} | Opt: {inst.valor_optimo:9.1f} | Gap: {gap_str} | Tiempo: {tiempo_medio:.2f}s", flush=True)

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
        "tiempo_medio_s": tiempo_medio,
        "tiempo_total_s": tiempo_total,
        "valores_runs":   valores_finales,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 9 Familias mknapcb1 a mknapcb9, instancia 0
    instancias = []
    for f_num in range(1, 10):
        instancias.append({
            "url": f"instancias/mknapcb{f_num}.txt",
            "index": 0,
            "nombre": f"mknapcb{f_num}_inst00",
            "familia": f"mknapcb{f_num}",
            "sub_nom": "inst_00",
        })

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
    batch_dir = os.path.join(OUTPUT_BASE, f"run_{dtw_mode}_3k_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    banner = "=" * 70
    print(f"\n{banner}")
    print("  PARALLEL MKP (FIRST INSTANCE OF EACH FILE) - HPC BENCHMARK")
    print(banner)
    print(f"  Instancias en paralelo : {len(instancias)} (mknapcb1 a mknapcb9, inst 0)")
    print(f"  Hilos concurrentes     : {MAX_WORKERS}")
    print(f"  Runs por instancia     : {N_RUNS}")
    print(f"  Max iters / run        : {MAX_ITERS}")
    print(f"  Carpeta de salida      : {batch_dir}")
    print(f"{banner}\n", flush=True)

    resumen_global = []

    # Ejecución paralela con ProcessPoolExecutor (multiprocesamiento real sin bloqueo de GIL)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros = {
            executor.submit(
                procesar_tarea_instancia,
                inst_info = item,
                max_iters = MAX_ITERS,
                n_runs    = N_RUNS,
                stag_cfg  = stag_cfg,
                batch_dir = batch_dir,
            ): item["nombre"]
            for item in instancias
        }

        for fut in concurrent.futures.as_completed(futuros):
            nom = futuros[fut]
            try:
                res = fut.result()
                resumen_global.append(res)
            except Exception as exc:
                print(f"  [ERROR] Excepción procesando {nom}: {exc}", flush=True)

    # Ordenar por nombre de instancia para que queden mknapcb1 a mknapcb9
    resumen_global.sort(key=lambda r: r["nombre"])

    # ── Reporte en Consola ────────────────────────────────────────────────────
    banner_l = "=" * 102
    print(f"\n\n{banner_l}")
    print("  RESUMEN GLOBAL CONSOLIDADO (9 INSTANCIAS)")
    print(banner_l)
    print(f"  {'#':<3} {'Instancia':<18} {'N':>5} {'M':>3} {'Media':>8} {'Std':>6} {'Mejor':>8} {'Optimo':>8} {'GapMed%':>8} {'GapMej%':>8} {'Sw(med)':>7} {'Tiempo(s)':>9}")
    print("  " + "-" * 100)
    for i, r in enumerate(resumen_global, 1):
        g_med_str = f"{r['gap_medio']:.2f}" if r["gap_medio"] is not None else "N/A"
        g_mej_str = f"{r['gap_mejor']:.2f}" if r["gap_mejor"] is not None else "N/A"
        t_med_str = f"{r.get('tiempo_medio_s', 0.0):.2f}"
        print(f"  {i:<3} {r['nombre']:<18} {r['n']:>5} {r['m']:>3} {r['media']:>8.1f}"
              f" {r['std']:>6.1f} {r['mejor']:>8.1f} {r['valor_optimo']:>8.1f}"
              f" {g_med_str:>8} {g_mej_str:>8} {r['switches_medio']:>7.1f} {t_med_str:>9}")
    print(banner_l)

    # ── Guardar resumen TXT ───────────────────────────────────────────────────
    resumen_path = os.path.join(batch_dir, "resumen_batch.txt")
    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write("RESUMEN GLOBAL DEL BATCH PARALELO (9 INSTANCIAS)\n")
        f.write(f"Fecha       : {timestamp}\n")
        f.write(f"Instancias  : {len(resumen_global)}\n")
        f.write(f"Runs/inst   : {N_RUNS}\n")
        f.write(f"Iters/run   : {MAX_ITERS}\n\n")
        f.write(f"{'#':<3} {'Instancia':<18} {'N':>5} {'M':>3} {'Media':>8} {'Std':>6} {'Mejor':>8} {'Optimo':>8} {'GapMed%':>8} {'GapMej%':>8} {'Sw(med)':>7} {'Tiempo(s)':>9}\n")
        f.write("-" * 100 + "\n")
        for i, r in enumerate(resumen_global, 1):
            g_med_str = f"{r['gap_medio']:.2f}" if r["gap_medio"] is not None else "N/A"
            g_mej_str = f"{r['gap_mejor']:.2f}" if r["gap_mejor"] is not None else "N/A"
            t_med_str = f"{r.get('tiempo_medio_s', 0.0):.2f}"
            f.write(f"{i:<3} {r['nombre']:<18} {r['n']:>5} {r['m']:>3} {r['media']:>8.1f}"
                    f" {r['std']:>6.1f} {r['mejor']:>8.1f} {r['valor_optimo']:>8.1f}"
                    f" {g_med_str:>8} {g_mej_str:>8} {r['switches_medio']:>7.1f} {t_med_str:>9}\n")
    print(f"\n  [txt] Resumen guardado en '{resumen_path}'")

    # ── Guardar resumen CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(batch_dir, "resumen_batch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instancia", "n", "m", "media", "std", "mediana", "mejor", "peor", "valor_optimo", "gap_medio_pct", "gap_mejor_pct", "switches_medio", "tiempo_medio_s"])
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
                f"{r.get('tiempo_medio_s', 0.0):.2f}",
            ])
    print(f"  [csv] Resumen guardado en '{csv_path}'")

    # ── Guardar resumen Markdown (.md) ────────────────────────────────────────
    md_path = os.path.join(batch_dir, "resumen_batch.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Resumen de Ejecución Paralela HPC - {timestamp}\n\n")
        f.write(f"- **Total de Instancias:** {len(resumen_global)} (primera instancia de mknapcb1 a mknapcb9)\n")
        f.write(f"- **Runs por Instancia:** {N_RUNS}\n")
        f.write(f"- **Max Iteraciones por Run:** {MAX_ITERS}\n\n")
        f.write("## Características de las Instancias y Resultados\n\n")
        f.write("| # | Instancia | N | M | Media | Std | Mejor | Peor | Óptimo | Gap Medio % | Gap Mejor % | Switches (med) | Tiempo Medio (s) |\n")
        f.write("|---|-----------|---|---|-------|-----|-------|------|--------|-------------|-------------|----------------|------------------|\n")
        for i, r in enumerate(resumen_global, 1):
            g_med_str = f"{r['gap_medio']:.2f}%" if r["gap_medio"] is not None else "N/A"
            g_mej_str = f"{r['gap_mejor']:.2f}%" if r["gap_mejor"] is not None else "N/A"
            t_med_str = f"{r.get('tiempo_medio_s', 0.0):.2f}s"
            f.write(f"| {i} | `{r['nombre']}` | {r['n']} | {r['m']} | {r['media']:.1f} | {r['std']:.2f} | {r['mejor']:.1f} | {r['peor']:.1f} | {r['valor_optimo']:.1f} | {g_med_str} | {g_mej_str} | {r['switches_medio']:.1f} | {t_med_str} |\n")
    print(f"  [md] Resumen guardado en '{md_path}'")

    # ── Análisis Estadístico Global ───────────────────────────────────────────
    if len(resumen_global) > 1:
        resultados_multi = {r["nombre"]: r["valores_runs"] for r in resumen_global}
        referencia_global = resumen_global[0]["nombre"]
        realizar_analisis_estadistico(
            resultados_dict      = resultados_multi,
            output_dir           = batch_dir,
            algoritmo_referencia = referencia_global,
            metrica_label        = "Fitness (Maximización MKP)",
            titulo_benchmark     = f"Batch MKP HPC ({len(resumen_global)} instancias)",
            minimizacion         = False,
            boxplot_filename     = "boxplot_comparativo_instancias.png",
            csv_filename         = "analisis_estadistico_global.csv",
            md_filename          = "analisis_estadistico_global.md",
        )

    print(f"\n  TODAS LAS 9 INSTANCIAS FINALIZADAS EXITOSAMENTE.\n")


if __name__ == "__main__":
    main()
