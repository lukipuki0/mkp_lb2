"""
HRES2-H2/run_hres2.py
---------------------
Script de ejecución de una sola corrida del Pipeline Híbrido DTW
sobre el sistema HRES2-H2 (WPEB Extendido).

Genera en resultados/run_TIMESTAMP/:
  - fitness_convergence.png
  - instantaneous_convergence.png
  - instantaneous_fitness_only.png
  - dtw_delta.png
  - switches_gantt.png
  - historial_dtw.csv
  - resumen_run.txt

Uso:
    python HRES2-H2/run_hres2.py
"""

import os
import csv
import sys
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")

# ── Path ──────────────────────────────────────────────────────────────────────
# Primero la carpeta propia de HRES2-H2 (para cargar su módulo plots local)
sys.path.insert(0, os.path.dirname(__file__))
# Luego la raíz del proyecto
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

from dtw_stagnation import StagnationConfig
import importlib
wpeb_model = importlib.import_module("HRES2-H2.wpeb_model")
HRES2Function   = wpeb_model.HRES2Function
decode_solution = wpeb_model.decode_solution

import importlib.util as _ilu_orch
_orch_spec = _ilu_orch.spec_from_file_location(
    "hres2_orchestrator",
    os.path.join(os.path.dirname(__file__), "orchestrator.py")
)
_orch_mod = _ilu_orch.module_from_spec(_orch_spec)
_orch_spec.loader.exec_module(_orch_mod)
ejecutar_pipeline_hres2 = _orch_mod.ejecutar_pipeline_hres2
COLORES_MH             = _orch_mod.COLORES_MH

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

_conv_mod = _load_hres2_plot("hres2_plots_conv",    "convergencia.py")
_dtw_mod  = _load_hres2_plot("hres2_plots_dtw",     "dtw_delta.py")
_gantt_mod= _load_hres2_plot("hres2_plots_gantt",   "switches_gantt.py")

grafico_convergencia_hres2 = _conv_mod.grafico_convergencia_hres2
grafico_dtw_delta           = _dtw_mod.grafico_dtw_delta
grafico_switches            = _gantt_mod.grafico_switches






# ── Configuración ─────────────────────────────────────────────────────────────

MAX_ITERS   = 1000   # iteraciones totales del pipeline
RANDOM_SEED = 42     # Semilla global fijada para reproducibilidad (42)

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "resultados")

# Parámetros del Monitor DTW
STAG_WINDOW      = 40
STAG_BAND        = 0
STAG_MIN_SLOPE   = 0.0
STAG_PLATEAU_MAX = 15
STAG_PATIENCE    = 8
STAG_USE_DDTW    = False
STAG_ADAPT       = True
STAG_P_LOW       = 30.0
STAG_P_HIGH      = 70.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if RANDOM_SEED is not None:
        import random
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    dtw_mode   = "ddtw" if STAG_USE_DDTW else "dtw"
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE, f"run_{dtw_mode}_{timestamp}")
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

    sep = "=" * 62
    print(f"\n{sep}")
    print("  HRES2-H2 — Una Corrida del Pipeline Híbrido DTW")
    print(f"  Pool Poblacional : PSO, GWO, WOA, EHO, ACO, ABC")
    print(f"  Pool Trayectoria : ILS, SA, TS, VNS")
    print(f"  Max Iteraciones  : {MAX_ITERS}")
    print(f"  Carpeta de salida: {output_dir}")
    print(sep)

    # ── Callback: imprime métricas HRES2 después de cada switch de MH ────────
    sep_cb = "-" * 90
    print(f"\n{'Epoch':>5} {'MH':<5} {'Tipo':<14} {'Iters':>6} {'LCOE CNY/kWh':>14} {'LCOH CNY/kg':>13} {'AGSR%':>7} {'H2 kg/año':>14} {'Válida':>7}")
    print(sep_cb)

    def epoch_callback(epoch, mh, tipo, iters_total, mejor_valor, mejor_solucion):
        if mejor_solucion is None:
            print(f"{epoch:>5} {mh:<5} {tipo:<14} {iters_total:>6}  {'---':>14}  {'---':>13}  {'---':>7}  {'---':>14}  {'---':>7}")
            return
        try:
            info    = func.get_info(mejor_solucion)
            lcoe    = mejor_valor
            lcoh    = info["lcoh_cny_per_kg"]
            agsr    = info["agsr"] * 100.0
            h2_kg   = info["total_h2_kg"]
            valida  = info.get("feasible", True)
            valida_str = " SI" if valida else " NO"
        except Exception:
            lcoe, lcoh, agsr, h2_kg, valida_str = mejor_valor, float("nan"), float("nan"), float("nan"), "ERR"
        print(f"{epoch:>5} {mh:<5} {tipo:<14} {iters_total:>6}  {lcoe:>14.6f}  {lcoh:>13.4f}  {agsr:>7.2f}  {h2_kg:>14,.0f}  {valida_str:>7}")

    # ── Ejecutar pipeline ────────────────────────────────────────────────────
    resultado = ejecutar_pipeline_hres2(
        func               = func,
        max_iters          = MAX_ITERS,
        stag_cfg           = stag_cfg,
        verbose            = False,
        on_epoch_callback  = epoch_callback,
    )
    print(sep_cb)


    # ── Decodificar la mejor solución ─────────────────────────────────────
    sol_4d  = np.array(resultado.mejor_solucion_global)
    decoded = decode_solution(sol_4d)
    info    = func.get_info(sol_4d)
    lcoe    = resultado.mejor_valor_global
    lcoh    = info["lcoh_cny_per_kg"]
    agsr    = info["agsr"]
    h2_kg   = info["total_h2_kg"]
    elz_cf  = info["electrolyzer_cf"]

    # ── Resumen en consola ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  RESULTADO FINAL")
    print(sep)
    print(f"  LCOE                : {lcoe:.6f} CNY/kWh")
    print(f"  LCOH                : {lcoh:.6f} CNY/kg")
    print(f"  AGSR (Grid Surplus) : {agsr * 100:.2f}%")
    print(f"  H2 Anual Producido  : {h2_kg:,.1f} kg/año")
    print(f"  Factor Cap. Elz     : {elz_cf * 100:.2f}%")
    print(f"  Total Switches      : {resultado.n_switches}")
    print(f"  Parque Eólico       : {decoded['wind_mw']:.2f} MW")
    print(f"  Parque Solar PV     : {decoded['pv_mw']:.2f} MW")
    print(f"  Electrolizador      : {decoded['electrolyzer_mw']:.1f} MW ({decoded['n_el_units']} unidades)")
    print(f"  Batería             : {decoded['battery_mw']:.1f} MW ({decoded['battery_duration_h']} h)")
    print(sep)
    print(f"\n  {'#':<3} {'MH':<5} {'Tipo':<14} {'Mejor LCOE':>12}  {'Inicio':>7}  {'Fin':>7}  {'Iters':>6}")
    print("  " + "-" * 58)
    for i, sw in enumerate(resultado.log_switches, 1):
        print(f"  {i:<3} {sw.mh_nombre:<5} {sw.tipo:<14} {sw.mejor_valor:>12.6f}"
              f"  {sw.t_inicio:>6.1f}s  {sw.t_fin:>6.1f}s  {sw.n_iters:>6}")

    # ── Muestra de resultados DTW por iteración en consola ──────────────────
    print(f"\n{sep}")
    print("  RESULTADOS DTW POR ITERACIÓN")
    print(sep)
    print(f"  {'Iter':>6}  {'MH':<6}  {'Tipo':<12}  {'LCOE (Best)':>12}  {'Delta DTW':>14}  {'Estado DTW':<24}")
    print("  " + "-" * 80)

    offset = 0
    for sw in resultado.log_switches:
        n_seg = sw.n_iters
        mh_nombre = sw.mh_nombre
        tipo = sw.tipo

        for i_local in range(n_seg):
            idx = offset + i_local
            if idx >= len(resultado.historial_global):
                break
            fit = resultado.historial_global[idx]
            d = resultado.dtw_deltas_global[idx] if idx < len(resultado.dtw_deltas_global) else float("nan")

            if isinstance(d, float) and np.isnan(d):
                delta_str = "      --      "
                estado_str = "Llenando ventana (W=30)"
            elif d > 0:
                delta_str = f"{d:+14.6f}"
                estado_str = "[!] ESTANCAMIENTO (Fire)"
            else:
                delta_str = f"{d:+14.6f}"
                estado_str = "[OK] Explotacion activa"


            print(f"  {idx + 1:6d}  {mh_nombre:<6}  {tipo:<12}  {fit:12.6f}  {delta_str:>14}  {estado_str:<24}")

        offset += n_seg


    # ── TXT Resumen ───────────────────────────────────────────────────────
    txt_path = os.path.join(output_dir, "resumen_run.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("HRES2-H2 — UNA CORRIDA DEL PIPELINE HÍBRIDO DTW\n")
        f.write(f"Timestamp           : {timestamp}\n")
        f.write(f"Max Iteraciones     : {MAX_ITERS}\n\n")
        f.write("Resultado:\n")
        f.write(f"  LCOE                : {lcoe:.6f} CNY/kWh\n")
        f.write(f"  LCOH                : {lcoh:.6f} CNY/kg\n")
        f.write(f"  AGSR (Grid Surplus) : {agsr * 100:.2f}%\n")
        f.write(f"  H2 Anual Producido  : {h2_kg:,.1f} kg/año\n")
        f.write(f"  Factor Cap. Elz     : {elz_cf * 100:.2f}%\n")
        f.write(f"  Total Switches      : {resultado.n_switches}\n\n")
        f.write("Configuración Óptima:\n")
        f.write(f"  Parque Eólico   : {decoded['wind_mw']:.2f} MW\n")
        f.write(f"  Parque Solar PV : {decoded['pv_mw']:.2f} MW\n")
        f.write(f"  Electrolizador  : {decoded['electrolyzer_mw']:.1f} MW ({decoded['n_el_units']} unidades)\n")
        f.write(f"  Batería         : {decoded['battery_mw']:.1f} MW ({decoded['battery_duration_h']} h)\n\n")
        f.write("Log de Switches:\n")
        for i, sw in enumerate(resultado.log_switches, 1):
            f.write(f"  {i:2d}. {sw.mh_nombre} ({sw.tipo}) | LCOE={sw.mejor_valor:.6f}"
                    f" | {sw.t_inicio:.1f}s-{sw.t_fin:.1f}s | iters={sw.n_iters}\n")
    print(f"\n  [txt] {txt_path}")

    # ── CSV y TXT Historial Detallado DTW (TODAS LAS MÉTRICAS) ─────────────
    csv_path = os.path.join(output_dir, "historial_dtw.csv")
    dtw_txt_path = os.path.join(output_dir, "historial_dtw_detalle.txt")

    deltas          = resultado.dtw_deltas_global
    inst_hist       = resultado.historial_inst_global
    dtw_info_global = getattr(resultado, 'dtw_info_global', []) or []

    # 1. Escribir CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteracion", "epoch", "mh", "tipo", "lcoe_best", "lcoe_instantaneo",
            "dtw_ready", "dtw_fire", "D1_vs_ramp", "D2_vs_const", "dtw_delta",
            "theta_c", "theta_r", "theta_delta", "no_improve_len", "trigger_streak",
            "window_n", "estado_dtw"
        ])

        offset = 0
        for ep_idx, sw in enumerate(resultado.log_switches, 1):
            n_seg = sw.n_iters
            for i_local in range(n_seg):
                idx = offset + i_local
                if idx >= len(resultado.historial_global):
                    break
                fit = resultado.historial_global[idx]
                fi  = inst_hist[idx] if idx < len(inst_hist) else float("nan")
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

    print(f"  [csv] {csv_path}")

    # 2. Escribir TXT Detallado
    with open(dtw_txt_path, "w", encoding="utf-8") as f_dtw:
        f_dtw.write("======================================================================================================================================================================================\n")
        f_dtw.write("  HISTORIAL DETALLADO DE TODAS LAS MÉTRICAS DTW POR ITERACIÓN — HRES2-H2 PIPELINE\n")
        f_dtw.write(f"  Timestamp: {timestamp}\n")
        f_dtw.write("======================================================================================================================================================================================\n\n")
        f_dtw.write(
            f"  {'Iter':>6}  {'Epoch':>5}  {'MH':<6}  {'Tipo':<12}  {'LCOE Best':>14}  {'LCOE Inst':>14}  "
            f"{'Ready':>6}  {'Fire':>6}  {'D1 (Ramp)':>14}  {'D2 (Const)':>14}  {'Delta DTW':>14}  "
            f"{'theta_c':>12}  {'theta_r':>12}  {'theta_delta':>14}  {'no_imp':>6}  {'streak':>6}  {'n_win':>6}  {'Estado DTW':<24}\n"
        )
        f_dtw.write("  " + "-" * 210 + "\n")

        offset = 0
        for ep_idx, sw in enumerate(resultado.log_switches, 1):
            n_seg = sw.n_iters
            for i_local in range(n_seg):
                idx = offset + i_local
                if idx >= len(resultado.historial_global):
                    break
                fit  = resultado.historial_global[idx]
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

                fi_str    = f"{fi:14.6f}" if not (isinstance(fi, float) and np.isnan(fi)) else "      --      "
                d1_str    = f"{d1:14.6f}" if not (isinstance(d1, float) and np.isnan(d1)) else "      --      "
                d2_str    = f"{d2:14.6f}" if not (isinstance(d2, float) and np.isnan(d2)) else "      --      "
                delta_str = f"{delta:+14.6f}" if not (isinstance(delta, float) and np.isnan(delta)) else "      --      "
                tc_str    = f"{tc:12.6f}" if not (isinstance(tc, float) and np.isnan(tc)) else "    --    "
                tr_str    = f"{tr:12.6f}" if not (isinstance(tr, float) and np.isnan(tr)) else "    --    "
                td_str    = f"{td:14.6f}" if not (isinstance(td, float) and np.isnan(td)) else "      --      "

                if not ready:
                    estado_str = "Llenando ventana (W=30)"
                elif fire:
                    estado_str = "[!] ESTANCAMIENTO (Fire)"
                else:
                    estado_str = "[OK] Explotacion activa"

                f_dtw.write(
                    f"  {idx + 1:6d}  {ep_idx:5d}  {sw.mh_nombre:<6}  {sw.tipo:<12}  {fit:14.6f}  {fi_str:>14}  "
                    f"{str(ready):>6}  {str(fire):>6}  {d1_str:>14}  {d2_str:>14}  {delta_str:>14}  "
                    f"{tc_str:>12}  {tr_str:>12}  {td_str:>14}  {no_imp:6d}  {streak:6d}  {win_n:6d}  {estado_str:<24}\n"
                )
            offset += n_seg

    print(f"  [txt] {dtw_txt_path}")



    # ── Gráficos ──────────────────────────────────────────────────────────
    print("\n  Generando gráficos...")
    grafico_convergencia_hres2(
        historial_global = resultado.historial_global,
        log_switches     = resultado.log_switches,
        colores_mh       = COLORES_MH,
        output_dir       = output_dir,
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


    print(f"\n  Todos los artefactos guardados en: '{output_dir}'")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
