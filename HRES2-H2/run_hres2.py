"""
HRES2-H2/run_hres2.py
---------------------
Script de ejecución de una sola corrida del Pipeline Híbrido DTW
sobre el sistema HRES2-H2 (WPEB Extendido).

Genera en resultados/run_TIMESTAMP/:
  - convergencia_fitness.png
  - convergencia_instantanea.png
  - solo_instantaneo.png
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dtw_stagnation import StagnationConfig
import importlib
wpeb_model = importlib.import_module("HRES2-H2.wpeb_model")
HRES2Function   = wpeb_model.HRES2Function
decode_solution = wpeb_model.decode_solution

from continuous_benchmark.orchestrator import ejecutar_pipeline, COLORES_MH
from plots import (
    grafico_convergencia,
    grafico_instantaneo,
    grafico_solo_instantaneo,
    grafico_dtw_delta,
    grafico_switches,
)


# ── Configuración ─────────────────────────────────────────────────────────────

MAX_ITERS   = 1000   # iteraciones totales del pipeline
RANDOM_SEED = None   # None → estocástico; int → reproducible

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if RANDOM_SEED is not None:
        import random
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

    sep = "=" * 62
    print(f"\n{sep}")
    print("  HRES2-H2 — Una Corrida del Pipeline Híbrido DTW")
    print(f"  Pool Poblacional : PSO, GWO, WOA, EHO, ACO, ABC")
    print(f"  Pool Trayectoria : ILS, SA")
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

    # ── Ejecutar pipeline ─────────────────────────────────────────────────
    resultado = ejecutar_pipeline(
        func              = func,
        max_iters         = MAX_ITERS,
        stag_cfg          = stag_cfg,
        verbose           = False,
        on_epoch_callback = epoch_callback,
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

    # ── CSV historial ─────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "historial_dtw.csv")
    deltas    = resultado.dtw_deltas_global
    inst_hist = resultado.historial_inst_global
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteracion", "lcoe_best", "dtw_delta", "lcoe_instantaneo"])
        for i, fit in enumerate(resultado.historial_global):
            d   = deltas[i]   if i < len(deltas)    else ""
            d_s = "" if (isinstance(d, float) and np.isnan(d)) else d
            fi  = inst_hist[i] if i < len(inst_hist) else ""
            writer.writerow([i, fit, d_s, fi])
    print(f"  [csv] {csv_path}")

    # ── Gráficos ──────────────────────────────────────────────────────────
    print("\n  Generando gráficos...")
    grafico_convergencia(
        historial_global = resultado.historial_global,
        log_switches     = resultado.log_switches,
        colores_mh       = COLORES_MH,
        valor_optimo     = 0.30,
        output_dir       = output_dir,
    )
    grafico_instantaneo(
        historial_global      = resultado.historial_global,
        historial_inst_global = resultado.historial_inst_global,
        log_switches          = resultado.log_switches,
        colores_mh            = COLORES_MH,
        valor_optimo          = 0.30,
        output_dir            = output_dir,
    )
    grafico_solo_instantaneo(
        historial_inst_global = resultado.historial_inst_global,
        log_switches          = resultado.log_switches,
        colores_mh            = COLORES_MH,
        valor_optimo          = 0.30,
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

    print(f"\n  Todos los artefactos guardados en: '{output_dir}'")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
