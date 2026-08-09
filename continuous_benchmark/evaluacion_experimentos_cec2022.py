"""
================================================================================
Evaluación de Experimentos Benchmark CEC 2022
================================================================================
Ejecuta la suite experimental completa de las 12 funciones de benchmark CEC 2022
siguiendo las especificaciones de los papers de referencia:
  - 30 ejecuciones independientes por función.
  - Tamaño de población: 30.
  - Iteraciones máximas: 1000.
  - Algoritmos evaluados: ABC (Artificial Bee Colony) y PSO (Particle Swarm Optimization).
  - Reporta: Best, Mean, SD (desviación estándar) y Tiempo promedio de ejecución.
  - Tests estadísticos:
      * Wilcoxon signed-rank test (comparación par a par).
      * Friedman rank test (ranking no paramétrico global).
  - Generación de curvas de convergencia (Fitness vs Iteración):
      * Escala LOGARÍTMICA para F1, F6 y F8.
      * Escala LINEAL para F2-F5, F7, F9-F12.
================================================================================
"""

import sys
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare

# Agregar directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from continuous_benchmark.funciones_cec2022 import (
    get_test_functions,
    OFFICIAL_BIASES,
    ContinuousFunction,
)
from continuous_benchmark.mh.abc import ejecutar_abc, ABCParams
from continuous_benchmark.mh.pso import ejecutar_epoch as ejecutar_pso, PSOParams

# ── Configuración Global ──
NUM_RUNS = 30
POP_SIZE = 30
MAX_ITERS = 1000
DIMENSION = 20
OUTPUT_DIR = os.path.join("resultados", "cec2022_experiments")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def ejecutar_experimentos():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    funciones = get_test_functions(n_dim=DIMENSION)

    print("=" * 80)
    print("  INICIANDO EVALUACIÓN DE EXPERIMENTOS CEC 2022")
    print(f"  Población: {POP_SIZE} | Iteraciones Máx: {MAX_ITERS} | Corridas: {NUM_RUNS} | Dim: {DIMENSION}")
    print("=" * 80)

    # Estructura de almacenamiento de resultados:
    # resultados[func_name][alg_name] = {"runs_best": [], "runs_time": [], "histories": []}
    resultados = {}

    for func in funciones:
        print(f"\n>>> Procesando {func.name} (Bias = {func.optimum})...")
        resultados[func.name] = {
            "ABC": {"runs_best": [], "runs_time": [], "histories": []},
            "PSO": {"runs_best": [], "runs_time": [], "histories": []},
        }

        for run in range(NUM_RUNS):
            seed = 42 + run * 100

            # 1. ABC
            t0 = time.perf_counter()
            params_abc = ABCParams(pop_size=POP_SIZE, iterations=MAX_ITERS)
            res_abc = ejecutar_abc(func, params_abc, seed=seed)
            t_abc = time.perf_counter() - t0

            resultados[func.name]["ABC"]["runs_best"].append(res_abc.mejor_valor)
            resultados[func.name]["ABC"]["runs_time"].append(t_abc)
            resultados[func.name]["ABC"]["histories"].append(res_abc.historial)

            # 2. PSO
            t0 = time.perf_counter()
            params_pso = PSOParams(pop_size=POP_SIZE, iterations=MAX_ITERS, use_stagnation=False)
            res_pso = ejecutar_pso(func, params_pso, verbose=False)
            t_pso = time.perf_counter() - t0

            resultados[func.name]["PSO"]["runs_best"].append(res_pso.mejor_valor)
            resultados[func.name]["PSO"]["runs_time"].append(t_pso)
            resultados[func.name]["PSO"]["histories"].append(res_pso.historial)

        # Imprimir progreso rápido por función
        abc_b = np.min(resultados[func.name]["ABC"]["runs_best"])
        abc_m = np.mean(resultados[func.name]["ABC"]["runs_best"])
        pso_b = np.min(resultados[func.name]["PSO"]["runs_best"])
        pso_m = np.mean(resultados[func.name]["PSO"]["runs_best"])
        print(f"    ABC -> Best: {abc_b:.4f}, Mean: {abc_m:.4f}")
        print(f"    PSO -> Best: {pso_b:.4f}, Mean: {pso_m:.4f}")

    # ── Reporte Estadístico Tabla ──
    reporte_csv = os.path.join(OUTPUT_DIR, "resumen_estadistico_cec2022.csv")
    reporte_txt = os.path.join(OUTPUT_DIR, "resumen_estadistico_cec2022.txt")

    print("\n" + "=" * 95)
    print("  RESUMEN ESTADÍSTICO CEC 2022 (30 CORRIDAS INDEPENDIENTES)")
    print("=" * 95)
    print(f"{'Función':<25} {'Alg.':<5} {'Best':>14} {'Mean':>14} {'SD':>14} {'Tiempo (s)':>12}")
    print("-" * 95)

    rows_csv = [["Funcion", "Algoritmo", "Best", "Mean", "SD", "Tiempo_Medio"]]

    for func_idx, func in enumerate(funciones, 1):
        for alg in ["ABC", "PSO"]:
            bests = resultados[func.name][alg]["runs_best"]
            times = resultados[func.name][alg]["runs_time"]

            best_val = float(np.min(bests))
            mean_val = float(np.mean(bests))
            sd_val = float(np.std(bests))
            mean_t = float(np.mean(times))

            print(f"{func.name:<25} {alg:<5} {best_val:>14.4f} {mean_val:>14.4f} {sd_val:>14.4f} {mean_t:>12.3f}")
            rows_csv.append([func.name, alg, f"{best_val:.6f}", f"{mean_val:.6f}", f"{sd_val:.6f}", f"{mean_t:.4f}"])

    with open(reporte_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_csv)
        
    with open(reporte_txt, "w", encoding="utf-8") as f:
        f.write("RESUMEN ESTADISTICO CEC 2022\n")
        f.write("=" * 95 + "\n")
        for row in rows_csv:
            f.write(f"{row[0]:<30} | {row[1]:<6} | Best: {row[2]:<12} | Mean: {row[3]:<12} | SD: {row[4]:<12} | Time: {row[5]}s\n")

    print("=" * 95)
    print(f" [CSV] Guardado en: {reporte_csv}")

    # ── Tests Estadísticos ──
    print("\n" + "=" * 80)
    print("  PRUEBAS ESTADÍSTICAS NO PARAMÉTRICAS (Wilcoxon & Friedman)")
    print("=" * 80)

    # 1. Wilcoxon Signed-Rank Test por función
    print("\n--- Wilcoxon Signed-Rank Test (ABC vs PSO) ---")
    print(f"{'Función':<25} {'Estadístico W':>15} {'p-value':>15} {'Significativo (alpha=0.05)':<25}")
    print("-" * 80)

    wilcoxon_rows = [["Funcion", "Wilcoxon_W", "p_value", "Significativo"]]

    # Datos agregados para test de Friedman (medias por función)
    means_abc = []
    means_pso = []

    for func in funciones:
        b_abc = np.array(resultados[func.name]["ABC"]["runs_best"])
        b_pso = np.array(resultados[func.name]["PSO"]["runs_best"])

        means_abc.append(np.mean(b_abc))
        means_pso.append(np.mean(b_pso))

        diffs = b_abc - b_pso
        if np.all(diffs == 0):
            stat, p_val = 0.0, 1.0
        else:
            try:
                stat, p_val = wilcoxon(b_abc, b_pso)
            except Exception:
                stat, p_val = 0.0, 1.0

        sig = "Sí (p < 0.05)" if p_val < 0.05 else "No (p >= 0.05)"
        print(f"{func.name:<25} {stat:>15.2f} {p_val:>15.4e} {sig:<25}")
        wilcoxon_rows.append([func.name, f"{stat:.4f}", f"{p_val:.4e}", sig])

    with open(os.path.join(OUTPUT_DIR, "test_wilcoxon.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(wilcoxon_rows)

    # 2. Friedman Rank Test global
    stat_f, p_val_f = friedmanchisquare(means_abc, means_pso)
    print("\n--- Friedman Rank Test Global ---")
    print(f"Estadístico Q de Friedman : {stat_f:.4f}")
    print(f"p-value de Friedman      : {p_val_f:.4e}")
    if p_val_f < 0.05:
        print("Resultado: Hay una diferencia estadísticamente significativa entre los algoritmos (p < 0.05).")
    else:
        print("Resultado: No se detectó diferencia estadísticamente significativa global (p >= 0.05).")

    with open(os.path.join(OUTPUT_DIR, "test_friedman.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Friedman_Q", "p_value", "Significativo"])
        writer.writerow([f"{stat_f:.4f}", f"{p_val_f:.4e}", "Sí" if p_val_f < 0.05 else "No"])

    # ── Curvas de Convergencia ──
    print("\n" + "=" * 80)
    print("  GENERANDO CURVAS DE CONVERGENCIA")
    print("=" * 80)

    # F1, F6 y F8 requieren escala logarítmica
    log_scale_funcs = {"F1", "F6", "F8"}

    for idx, func in enumerate(funciones, 1):
        f_key = f"F{idx}"
        is_log = f_key in log_scale_funcs

        plt.figure(figsize=(8, 5))

        for alg, color in zip(["ABC", "PSO"], ["#1f77b4", "#ff7f0e"]):
            histories = np.array(resultados[func.name][alg]["histories"])
            mean_history = np.mean(histories, axis=0)
            
            # Si el gráfico usa escala log, restar el bias para ver la convergencia hacia 0
            if is_log:
                plot_data = np.maximum(mean_history - func.optimum, 1e-12)
            else:
                plot_data = mean_history

            plt.plot(range(1, MAX_ITERS + 1), plot_data, label=alg, color=color, linewidth=2)

        title_scale = " (Escala Logarítmica f(x)-bias)" if is_log else " (Escala Lineal)"
        plt.title(f"Convergencia {func.name}{title_scale}", fontsize=12, fontweight="bold")
        plt.xlabel("Iteración", fontsize=10)
        plt.ylabel("Fitness (Valor f(x))" if not is_log else "Error Log (f(x) - f*)", fontsize=10)
        
        if is_log:
            plt.yscale("log")
            
        plt.axhline(y=func.optimum if not is_log else 1e-12, color="red", linestyle="--", alpha=0.6, label="Óptimo/Bias")
        plt.grid(True, which="both", linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()

        plot_path = os.path.join(PLOTS_DIR, f"convergencia_{f_key}_{func.name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        scale_str = "LOGARÍTMICA" if is_log else "LINEAL"
        print(f"  [{scale_str:<11}] {plot_path}")

    print("\n>>> EXPERIMENTOS COMPLETADOS EXITOSAMENTE. RESULTADOS GUARDADOS EN 'resultados/cec2022_experiments/' <<<")


if __name__ == "__main__":
    ejecutar_experimentos()
