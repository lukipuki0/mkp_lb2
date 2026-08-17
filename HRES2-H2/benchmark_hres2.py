"""
HRES2-H2/benchmark_hres2.py
---------------------------
Benchmark de Ejecución Múltiple (31 Runs Independientes) para el sistema HRES2-H2
usando el Orquestador Híbrido de Metaheurísticas por Rotación DTW.
Pool Activo: PSO, GWO, WOA, EHO, ACO, ABC, ILS, SA (Excluye GA).
"""

from __future__ import annotations

import os
import csv
import random
import datetime
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Añadir primero la carpeta propia de HRES2-H2 (para cargar su módulo plots local)
sys.path.insert(0, os.path.dirname(__file__))
# Luego la raíz del proyecto
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

from dtw_stagnation import StagnationConfig
import importlib
wpeb_model = importlib.import_module("HRES2-H2.wpeb_model")
HRES2Function = wpeb_model.HRES2Function
decode_solution = wpeb_model.decode_solution

from continuous_benchmark.orchestrator import ejecutar_pipeline, ejecutar_mh_standalone, COLORES_MH
import importlib
analisis_mod = importlib.import_module("HRES2-H2.analisis_estadistico_hres2")
realizar_analisis_estadistico_completo = analisis_mod.realizar_analisis_estadistico_completo

# ── Importar módulo plots LOCAL de HRES2-H2 (sin conflicto con raíz) ──────────
import importlib.util as _ilu

def _load_hres2_plot(module_name, filename):
    _dir = os.path.dirname(__file__)
    spec = _ilu.spec_from_file_location(
        module_name, os.path.join(_dir, "plots", filename)
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_conv_mod  = _load_hres2_plot("hres2_plots_conv",  "convergencia.py")
_dtw_mod   = _load_hres2_plot("hres2_plots_dtw",   "dtw_delta.py")
_gantt_mod = _load_hres2_plot("hres2_plots_gantt", "switches_gantt.py")

grafico_convergencia_hres2 = _conv_mod.grafico_convergencia_hres2
grafico_dtw_delta           = _dtw_mod.grafico_dtw_delta
grafico_switches            = _gantt_mod.grafico_switches





# ── Configuración ─────────────────────────────────────────────────────────────

N_RUNS                 = 31     # Número de ejecuciones independientes
MAX_ITERS_POR_RUN      = 1000   # Iteraciones máximas por run
RANDOM_SEED            = 42     # Semilla global fijada para reproducibilidad (42)
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "resultados")

# Parámetros del Monitor DTW
STAG_WINDOW      = 30
STAG_BAND        = 0
STAG_MIN_SLOPE   = 0.0
STAG_PLATEAU_MAX = 15
STAG_PATIENCE    = 3
STAG_USE_DDTW    = False
STAG_ADAPT       = True
STAG_P_LOW       = 30.0
STAG_P_HIGH      = 70.0


# ── Función Auxiliar para Graficar Boxplot de HRES2-H2 ─────────────────────────

def grafico_boxplot_hres2(valores: list[float], output_dir: str):
    """Genera el gráfico de boxplot para las 31 ejecuciones de HRES2-H2."""
    plt.figure(figsize=(7, 5))
    bp = plt.boxplot(valores, patch_artist=True, widths=0.4,
                     boxprops=dict(facecolor="#2196F3", color="#1565C0", alpha=0.7),
                     medianprops=dict(color="#D32F2F", linewidth=2.0),
                     whiskerprops=dict(color="#1565C0", linewidth=1.5),
                     capprops=dict(color="#1565C0", linewidth=1.5),
                     flierprops=dict(marker="o", color="#D32F2F", alpha=0.6))

    media = np.mean(valores)
    plt.axhline(y=media, color="#388E3C", linestyle="--", linewidth=1.5, label=f"Mean: {media:.4f} CNY/kWh")

    plt.title("LCOE Distribution across 31 Runs - HRES2-H2 (WPEB)", fontsize=12, fontweight="bold")
    plt.ylabel("LCOE (CNY/kWh)", fontsize=11)
    plt.xticks([1], ["HRES2-H2 Hybrid DTW"])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "lcoe_boxplot_31runs.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [boxplot] {out_path}")


# ── Orquestación del Benchmark HRES2-H2 ────────────────────────────────────────

def ejecutar_benchmark_hres2(
    n_runs    : int = N_RUNS,
    max_iters : int = MAX_ITERS_POR_RUN,
    output_dir: str | None = None,
) -> dict:
    """Ejecuta 31 runs independientes del pipeline híbrido y standalone en HRES2-H2."""
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_BASE, f"run_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    func = HRES2Function()

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

    banner = "=" * 66
    print(f"\n{banner}")
    print(f"  BENCHMARK HRES2-H2 (WPEB Extendido) - PIPELINE HÍBRIDO DTW")
    print(banner)
    print(f"  Runs independientes : {n_runs}")
    print(f"  Max Iters / Run     : {max_iters}")
    print(f"  Pool Poblacional    : PSO, GWO, WOA, EHO, ACO, ABC")
    print(f"  Pool Trayectoria    : ILS, SA")
    print(f"  Carpeta de salida   : {output_dir}")
    print(banner)

    valores_lcoe = []
    soluciones_4d = []
    n_switches_list = []
    detalles_runs = []

    # 1. Ejecutar Pipeline Híbrido DTW (31 Runs)
    resultados_runs = []
    for r in range(1, n_runs + 1):
        if RANDOM_SEED is not None:
            run_seed = RANDOM_SEED + r
            import random
            random.seed(run_seed)
            np.random.seed(run_seed)

        print(f"  >>> RUN HYBRID DTW [{r:02d}/{n_runs:02d}] en progreso...")
        res = ejecutar_pipeline(
            func       = func,
            max_iters  = max_iters,
            stag_cfg   = stag_cfg,
            verbose    = False,
        )

        best_lcoe = res.mejor_valor_global
        sol_4d = res.mejor_solucion_global
        decoded = decode_solution(np.array(sol_4d))

        info = func.get_info(np.array(sol_4d))
        lcoh = info["lcoh_cny_per_kg"]
        agsr = info["agsr"]
        total_h2_kg = info["total_h2_kg"]
        electrolyzer_cf = info["electrolyzer_cf"]

        valores_lcoe.append(best_lcoe)
        soluciones_4d.append(sol_4d)
        n_switches_list.append(res.n_switches)
        resultados_runs.append(res)

        detalles_runs.append({
            "run": r,
            "lcoe": best_lcoe,
            "lcoh": lcoh,
            "agsr": agsr,
            "total_h2_kg": total_h2_kg,
            "electrolyzer_cf": electrolyzer_cf,
            "switches": res.n_switches,
            "wind_mw": decoded["wind_mw"],
            "pv_mw": decoded["pv_mw"],
            "n_el_units": decoded["n_el_units"],
            "electrolyzer_mw": decoded["electrolyzer_mw"],
            "battery_mw": decoded["battery_mw"],
            "battery_duration_h": decoded["battery_duration_h"],
        })

        print(f"      Run {r:02d} completado | LCOE: {best_lcoe:.6f} CNY/kWh | LCOH: {lcoh:.4f} CNY/kg | AGSR: {agsr*100:.2f}% | "
              f"Eólica: {decoded['wind_mw']:.1f}MW, PV: {decoded['pv_mw']:.1f}MW, "
              f"Elz: {decoded['electrolyzer_mw']}MW, Bat: {decoded['battery_mw']}MW ({decoded['battery_duration_h']}h) | "
              f"Switches: {res.n_switches}")

    # 2. Ejecutar Metaheurísticas Standalone Comparativas (31 Runs cada una)
    standalone_mhs = ["PSO", "GWO", "WOA", "EHO", "ACO", "ABC", "ILS", "SA"]
    resultados_dict = {"Hybrid DTW": valores_lcoe}

    print(f"\n{banner}")
    print(f"  EJECUTANDO BENCHMARKS STANDALONE PARA ANÁLISIS ESTADÍSTICO ({n_runs} RUNS POR MH)")
    print(banner)

    for mh in standalone_mhs:
        print(f"  > Ejecutando {mh} Standalone ({n_runs} runs x {max_iters} iters)...")
        vals_mh = []
        for r in range(n_runs):
            res_std = ejecutar_mh_standalone(func, mh, max_iters=max_iters)
            vals_mh.append(res_std.mejor_valor)
        resultados_dict[mh] = vals_mh
        print(f"    {mh:<4s} | Media: {np.mean(vals_mh):.6f} | Min: {np.min(vals_mh):.6f} | Max: {np.max(vals_mh):.6f}")

    # 3. Métricas Estadísticas Generales del Hybrid DTW
    media_lcoe   = float(np.mean(valores_lcoe))
    std_lcoe     = float(np.std(valores_lcoe))
    mediana_lcoe = float(np.median(valores_lcoe))
    mejor_lcoe   = float(np.min(valores_lcoe))
    peor_lcoe    = float(np.max(valores_lcoe))

    valores_lcoh = [d["lcoh"] for d in detalles_runs]
    valores_agsr = [d["agsr"] for d in detalles_runs]
    media_lcoh   = float(np.mean(valores_lcoh))
    media_agsr   = float(np.mean(valores_agsr))

    best_run_idx = int(np.argmin(valores_lcoe))
    best_config = detalles_runs[best_run_idx]
    best_res = resultados_runs[best_run_idx]

    # Guardar gráficos e información EXCLUSIVAMENTE del MEJOR RUN
    best_run_dir = os.path.join(output_dir, f"best_run_{best_run_idx + 1:02d}")
    os.makedirs(best_run_dir, exist_ok=True)

    grafico_convergencia_hres2(
        historial_global = best_res.historial_global,
        log_switches     = best_res.log_switches,
        colores_mh       = COLORES_MH,
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


    with open(os.path.join(best_run_dir, "detalle_run.txt"), "w", encoding="utf-8") as f_run:
        f_run.write(f"DETALLE MEJOR RUN (Run #{best_run_idx + 1:02d})\n")
        f_run.write(f"LCOE                : {best_config['lcoe']:.6f} CNY/kWh\n")
        f_run.write(f"LCOH                : {best_config['lcoh']:.6f} CNY/kg\n")
        f_run.write(f"AGSR (Grid Surplus) : {best_config['agsr'] * 100:.2f}%\n")
        f_run.write(f"H2 Anual Producido  : {best_config['total_h2_kg']:,.1f} kg\n")
        f_run.write(f"Factor Cap. Elz     : {best_config['electrolyzer_cf'] * 100:.2f}%\n")
        f_run.write(f"Total Switches      : {best_config['switches']}\n")
        f_run.write(f"Potencia Eólica     : {best_config['wind_mw']:.2f} MW\n")
        f_run.write(f"Potencia Solar PV   : {best_config['pv_mw']:.2f} MW\n")
        f_run.write(f"Electrolizador MW   : {best_config['electrolyzer_mw']:.1f} MW ({best_config['n_el_units']} unidades)\n")
        f_run.write(f"Batería MW          : {best_config['battery_mw']:.1f} MW ({best_config['battery_duration_h']} h)\n")

    # CSV y TXT Historial Detallado DTW del Mejor Run
    csv_best_path = os.path.join(best_run_dir, "historial_dtw.csv")
    txt_best_path = os.path.join(best_run_dir, "historial_dtw_detalle.txt")

    deltas          = best_res.dtw_deltas_global
    inst_hist       = getattr(best_res, 'historial_inst_global', []) or []
    dtw_info_global = getattr(best_res, 'dtw_info_global', []) or []

    with open(csv_best_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteracion", "epoch", "mh", "tipo", "lcoe_best", "lcoe_instantaneo",
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

                if not ready:
                    estado_str = "Llenando ventana"
                elif fire:
                    estado_str = "ESTANCAMIENTO (Fire)"
                else:
                    estado_str = "Explotacion activa"

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
        f_dtw.write(f"  HISTORIAL DETALLADO DE TODAS LAS MÉTRICAS DTW POR ITERACIÓN — MEJOR RUN (#{best_run_idx + 1:02d})\n")
        f_dtw.write("======================================================================================================================================================================================\n\n")
        f_dtw.write(
            f"  {'Iter':>6}  {'Epoch':>5}  {'MH':<6}  {'Tipo':<12}  {'LCOE Best':>14}  {'LCOE Inst':>14}  "
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

                if not ready: estado_str = f"Llenando ventana (W={win_n})"
                elif fire:    estado_str = "[!] ESTANCAMIENTO (Fire)"
                else:         estado_str = "Explotacion activa"

                d1_s  = f"{d1:14.6f}" if (isinstance(d1, float) and not np.isnan(d1)) else f"{'--':>14}"
                d2_s  = f"{d2:14.6f}" if (isinstance(d2, float) and not np.isnan(d2)) else f"{'--':>14}"
                dl_s  = f"{delta:+14.6f}" if (isinstance(delta, float) and not np.isnan(delta)) else f"{'--':>14}"
                fi_s  = f"{fi:14.6f}" if (isinstance(fi, float) and not np.isnan(fi)) else f"{'--':>14}"
                tc_s  = f"{tc:12.6f}" if (isinstance(tc, float) and not np.isnan(tc)) else f"{'--':>12}"
                tr_s  = f"{tr:12.6f}" if (isinstance(tr, float) and not np.isnan(tr)) else f"{'--':>12}"
                td_s  = f"{td:14.6f}" if (isinstance(td, float) and not np.isnan(td)) else f"{'--':>14}"

                f_dtw.write(
                    f"  {idx + 1:6d}  {ep_idx:5d}  {sw.mh_nombre:<6}  {sw.tipo:<12}  {fit:14.6f}  {fi_s}  "
                    f"{str(ready):>6}  {str(fire):>6}  {d1_s}  {d2_s}  {dl_s}  "
                    f"{tc_s}  {tr_s}  {td_s}  {no_imp:6d}  {streak:6d}  {win_n:6d}  {estado_str:<24}\n"
                )
            offset += n_seg

    print(f"\n  [info] Gráficos y archivos DTW del mejor run guardados en: '{best_run_dir}'")


    print(f"\n{banner}")
    print(f"  RESUMEN ESTADÍSTICO GENERAL ({n_runs} RUNS) - HRES2-H2")
    print(banner)
    print(f"  Media LCOE   : {media_lcoe:.6f} CNY/kWh")
    print(f"  Std LCOE     : {std_lcoe:.6f}")
    print(f"  Media LCOH   : {media_lcoh:.6f} CNY/kg")
    print(f"  Media AGSR   : {media_agsr * 100:.2f}%")
    print(f"  Mediana LCOE : {mediana_lcoe:.6f} CNY/kWh")
    print(f"  Mejor LCOE   : {mejor_lcoe:.6f} CNY/kWh (Run #{best_run_idx + 1})")
    print(f"  Peor LCOE    : {peor_lcoe:.6f} CNY/kWh")
    print(f"  Configuración Óptima:")
    print(f"    - Parque Eólico   : {best_config['wind_mw']:.2f} MW")
    print(f"    - Parque Solar PV : {best_config['pv_mw']:.2f} MW")
    print(f"    - Electrolizador  : {best_config['electrolyzer_mw']:.1f} MW ({best_config['n_el_units']} unidades)")
    print(f"    - Batería         : {best_config['battery_mw']:.1f} MW ({best_config['battery_duration_h']} horas)")
    print(f"    - Producción H2   : {best_config['total_h2_kg']:,.1f} kg/año (LCOH = {best_config['lcoh']:.4f} CNY/kg)")
    print(f"    - Ratio Excedente : AGSR = {best_config['agsr']*100:.2f}%")
    print(banner)

    # Boxplot individual del Hybrid DTW
    grafico_boxplot_hres2(valores_lcoe, output_dir)

    # CSV de Runs del Hybrid DTW
    csv_runs_path = os.path.join(output_dir, "runs_resultados.csv")
    with open(csv_runs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "lcoe_cny_kwh", "lcoh_cny_kg", "agsr_pct", "total_h2_kg", "electrolyzer_cf_pct",
                         "n_switches", "wind_mw", "pv_mw", "n_el_units", "electrolyzer_mw", "battery_mw", "battery_duration_h"])
        for d in detalles_runs:
            writer.writerow([d["run"], f"{d['lcoe']:.6f}", f"{d['lcoh']:.6f}", f"{d['agsr']*100:.4f}",
                             f"{d['total_h2_kg']:.2f}", f"{d['electrolyzer_cf']*100:.4f}",
                             d["switches"], d["wind_mw"], d["pv_mw"], d["n_el_units"],
                             d["electrolyzer_mw"], d["battery_mw"], d["battery_duration_h"]])
    print(f"  [csv] {csv_runs_path}")

    # Reporte TXT Resumen
    txt_path = os.path.join(output_dir, "resumen_hres2.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("RESUMEN BENCHMARK HRES2-H2 (WPEB EXTENDIDO)\n")
        f.write(f"Runs independientes : {n_runs}\n")
        f.write(f"Max Iters / Run     : {max_iters}\n\n")
        f.write("Estadísticas Principales:\n")
        f.write(f"  Media LCOE       : {media_lcoe:.6f} CNY/kWh (± {std_lcoe:.6f})\n")
        f.write(f"  Media LCOH       : {media_lcoh:.6f} CNY/kg\n")
        f.write(f"  Media AGSR       : {media_agsr * 100:.2f}%\n")
        f.write(f"  Mejor LCOE       : {mejor_lcoe:.6f} CNY/kWh (Run #{best_run_idx + 1})\n")
        f.write(f"  Peor LCOE        : {peor_lcoe:.6f} CNY/kWh\n\n")
        f.write("Configuración Global Óptima:\n")
        f.write(f"  Parque Eólico    : {best_config['wind_mw']:.2f} MW\n")
        f.write(f"  Parque Solar PV  : {best_config['pv_mw']:.2f} MW\n")
        f.write(f"  Electrolizador   : {best_config['electrolyzer_mw']:.1f} MW ({best_config['n_el_units']} unidades)\n")
        f.write(f"  Batería          : {best_config['battery_mw']:.1f} MW ({best_config['battery_duration_h']} h)\n")
        f.write(f"  Producción H2    : {best_config['total_h2_kg']:,.1f} kg/año\n")
        f.write(f"  LCOH Óptimo      : {best_config['lcoh']:.6f} CNY/kg\n")
        f.write(f"  AGSR Óptimo      : {best_config['agsr'] * 100:.2f}%\n\n")
        f.write("Detalle por Run:\n")
        for d in detalles_runs:
            f.write(f"  Run {d['run']:02d}: LCOE={d['lcoe']:.6f} | LCOH={d['lcoh']:.4f} | AGSR={d['agsr']*100:.2f}% | "
                    f"Wind={d['wind_mw']:.1f}MW, PV={d['pv_mw']:.1f}MW, Elz={d['electrolyzer_mw']}MW, Bat={d['battery_mw']}MW ({d['battery_duration_h']}h) | "
                    f"Switches={d['switches']}\n")
    print(f"  [txt] {txt_path}")

    # Reporte Markdown Resumen
    md_path = os.path.join(output_dir, "resumen_hres2.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Resumen Benchmark HRES2-H2 - Pipeline Híbrido DTW ({n_runs} Runs)\n\n")
        f.write(f"- **Runs totales:** {n_runs}\n")
        f.write(f"- **Iteraciones por Run:** {max_iters}\n")
        f.write(f"- **LCOE Medio:** `{media_lcoe:.6f} CNY/kWh` (± {std_lcoe:.6f})\n")
        f.write(f"- **LCOH Medio:** `{media_lcoh:.6f} CNY/kg`\n")
        f.write(f"- **AGSR Medio (Excedente Red):** `{media_agsr * 100:.2f}%` (Límite max: 20.00%)\n")
        f.write(f"- **Mejor LCOE:** **`{mejor_lcoe:.6f} CNY/kWh`**\n\n")
        f.write("## Configuración del Sistema Óptimo Encontrado\n\n")
        f.write("| Componente / Métrica | Valor Óptimo |\n")
        f.write("|---|---|\n")
        f.write(f"| **LCOE (Coste Nivelado de Energía)** | `{best_config['lcoe']:.6f} CNY/kWh` |\n")
        f.write(f"| **LCOH (Coste Nivelado del Hidrógeno)** | `{best_config['lcoh']:.6f} CNY/kg` |\n")
        f.write(f"| **AGSR (Annual Green Hydrogen Surplus Ratio)** | `{best_config['agsr'] * 100:.2f}%` |\n")
        f.write(f"| **Producción de H2 Verde** | `{best_config['total_h2_kg']:,.1f} kg/año` |\n")
        f.write(f"| **Potencia Eólica** | `{best_config['wind_mw']:.2f} MW` |\n")
        f.write(f"| **Potencia Solar PV** | `{best_config['pv_mw']:.2f} MW` |\n")
        f.write(f"| **Electrolizador H2** | `{best_config['electrolyzer_mw']:.1f} MW` ({best_config['n_el_units']} módulos de 5 MW) |\n")
        f.write(f"| **Almacenamiento Baterías** | `{best_config['battery_mw']:.1f} MW` ({best_config['battery_duration_h']} h) |\n\n")
    print(f"  [md]  {md_path}")



    # 4. ANÁLISIS ESTADÍSTICO INFERENCIAL (p-valores, Wilcoxon, Shapiro, Friedman)
    stat_results = realizar_analisis_estadistico_completo(
        resultados_dict = resultados_dict,
        output_dir      = output_dir,
        algoritmo_referencia = "Hybrid DTW",
    )

    return {
        "media_lcoe": media_lcoe,
        "std_lcoe": std_lcoe,
        "mejor_lcoe": mejor_lcoe,
        "best_config": best_config,
        "stat_results": stat_results,
    }


def main():

    ejecutar_benchmark_hres2(
        n_runs    = N_RUNS,
        max_iters = MAX_ITERS_POR_RUN,
    )



if __name__ == "__main__":
    main()
