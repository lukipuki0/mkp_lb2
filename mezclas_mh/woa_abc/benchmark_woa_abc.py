"""
mezclas_mh/woa_abc/benchmark_woa_abc.py
----------------------------------------
Benchmark completo y unificado para la mezcla WOA-ABC (Variantes A, B, C, MDG-WABC, WOA y ABC).

Cada ejecución del benchmark crea una carpeta por corrida en:
  resultados/mezclas_mh/woa_abc/run_<TIMESTAMP>/

Con 2 carpetas principales en su interior:
  1. mkp/     : Resultados detallados de cada instancia del Multidimensional Knapsack Problem.
                Contiene subcarpetas por instancia (gráficos PNG, CSV, TXT) y resúmenes globales en la raíz de mkp/.
  2. cec2022/  : Resultados detallados de las 12 funciones del Benchmark Continuo CEC2022.
                Contiene subcarpetas por función F1..F12 (gráficos PNG, CSV, TXT) y resúmenes globales en la raíz de cec2022/.

Uso:
    python -m mezclas_mh.woa_abc.benchmark_woa_abc
"""

from __future__ import annotations

import csv
import datetime
import os
import random
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from continuous_benchmark.funciones_cec2022 import get_test_functions, ContinuousFunction
from mkp_core.data_loader import cargar_instancias
from mkp_core.problem import MKPInstance

from mh.woa import WOAParams as WOAMKPParams, ejecutar_epoch as woa_mkp_epoch
from mh.abc import ABCParams as ABCMKPParams, ejecutar_epoch as abc_mkp_epoch
from continuous_benchmark.mh.woa import WOAParams as WOAContParams, ejecutar_epoch as woa_cont_epoch
from continuous_benchmark.mh.abc import ABCParams as ABCContParams, ejecutar_epoch as abc_cont_epoch

from mezclas_mh.woa_abc import (
    VariantAParams, variant_a_epoch, variant_a_epoch_continuo,
    VariantBParams, variant_b_epoch, variant_b_epoch_continuo,
    VariantCParams, variant_c_epoch, variant_c_epoch_continuo,
    MDGWABCParams, mdg_wabc_epoch, mdg_wabc_epoch_continuo,
    DTWWOAABCParams, variant_d_epoch, variant_d_epoch_continuo,
)


# ── Configuración General ──────────────────────────────────────────────────────
DIMENSION_CEC = 10
ITERACIONES = 100
POP_SIZE = 30
NUM_INSTANCIAS_MKP = 5   # Primeras N instancias de MKP a evaluar por corrida
OUTPUT_BASE = os.path.join("resultados", "mezclas_mh", "woa_abc")
RANDOM_SEED = 42

COLORES_ALGO = {
    "WOA Puro": "#2196F3",               # Azul
    "ABC Puro": "#FFC107",               # Amarillo
    "Variante A": "#9C27B0",             # Morado
    "Variante B": "#FF5722",             # Naranja
    "Variante C": "#00BCD4",             # Cian
    "MDG-WABC (B+C)": "#4CAF50",         # Verde
    "Variante D (DTW)": "#E91E63",       # Rosado/Fucsia
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. EVALUACIÓN Y GUARDADO: MKP (Discreto)
# ─────────────────────────────────────────────────────────────────────────────

def procesar_instancia_mkp(
    inst: MKPInstance,
    inst_name: str,
    iterations: int,
    output_dir: str,
) -> dict:
    """Ejecuta todos los algoritmos sobre una instancia MKP y guarda gráficos, CSV y TXT."""
    os.makedirs(output_dir, exist_ok=True)

    p_woa = WOAMKPParams(pop_size=POP_SIZE, iterations=iterations)
    p_abc = ABCMKPParams(pop_size=POP_SIZE, iterations=iterations)
    p_a = VariantAParams(pop_size=POP_SIZE, iterations=iterations)
    p_b = VariantBParams(pop_size=POP_SIZE, iterations=iterations)
    p_c = VariantCParams(pop_size=POP_SIZE, iterations=iterations)
    p_mdg = MDGWABCParams(pop_size=POP_SIZE, iterations=iterations)
    p_d = DTWWOAABCParams(pop_size=POP_SIZE, iterations=iterations)

    dict_res = {}
    algos = [
        ("WOA Puro", woa_mkp_epoch, p_woa),
        ("ABC Puro", abc_mkp_epoch, p_abc),
        ("Variante A", variant_a_epoch, p_a),
        ("Variante B", variant_b_epoch, p_b),
        ("Variante C", variant_c_epoch, p_c),
        ("MDG-WABC (B+C)", mdg_wabc_epoch, p_mdg),
        ("Variante D (DTW)", variant_d_epoch, p_d),
    ]

    for name, fn_epoch, params in algos:
        print(f"\n    ---> Ejecutando {name} en {inst_name} ({iterations} iters)...", flush=True)
        dict_res[name] = fn_epoch(inst, params, verbose=True)

    # TXT Individual
    txt_path = os.path.join(output_dir, "resumen_instancia.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Instancia          : {inst_name}\n")
        f.write(f"Dimensiones (n x m): {inst.n} x {inst.m}\n")
        f.write(f"Óptimo Conocido    : {inst.valor_optimo if inst.valor_optimo else 'Desconocido'}\n\n")
        f.write(f"{'Algoritmo':<20} {'Mejor Valor':>14} {'Iteraciones':>12}\n")
        f.write("-" * 50 + "\n")
        for name, r in dict_res.items():
            f.write(f"{name:<20} {r.mejor_valor:>14.1f} {r.iteraciones:>12}\n")

    # CSV Historial
    csv_path = os.path.join(output_dir, "historial_convergencia.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteracion"] + list(dict_res.keys()))
        max_len = max(len(r.historial) for r in dict_res.values())
        for i in range(max_len):
            row = [i]
            for r in dict_res.values():
                val = r.historial[i] if i < len(r.historial) else r.historial[-1]
                row.append(val)
            writer.writerow(row)

    # Gráfico 1: Convergencia MKP (Maximización)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in dict_res.items():
        color = COLORES_ALGO.get(name, "#333333")
        ax.plot(r.historial, label=name, color=color, linewidth=2)
    if inst.valor_optimo and inst.valor_optimo > 0:
        ax.axhline(inst.valor_optimo, color="red", linestyle="--", alpha=0.7, label=f"Óptimo ({inst.valor_optimo})")
    ax.set_title(f"Convergencia MKP - {inst_name} (n={inst.n}, m={inst.m})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Fitness / Valor Ganancia (Maximización)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_convergencia_comparativa.png"), dpi=150)
    plt.close(fig)

    # Gráfico 2: Instantáneo
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in dict_res.items():
        color = COLORES_ALGO.get(name, "#333333")
        if hasattr(r, "historial_inst") and r.historial_inst:
            ax.plot(r.historial_inst, label=name, color=color, alpha=0.5, linewidth=1)
    ax.set_title(f"Fitness Instantáneo MKP - {inst_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Fitness de la Iteración")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_instantaneo_comparativo.png"), dpi=150)
    plt.close(fig)

    mejores_valores = {k: v.mejor_valor for k, v in dict_res.items()}
    ganador = max(mejores_valores, key=mejores_valores.get)

    return {
        "nombre": inst_name,
        "n": inst.n,
        "m": inst.m,
        "optimo": inst.valor_optimo,
        "ganador": ganador,
        "mejor_valor": mejores_valores[ganador],
        "detalles": mejores_valores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. EVALUACIÓN Y GUARDADO: BENCHMARK CONTINUO (CEC2022)
# ─────────────────────────────────────────────────────────────────────────────

def procesar_funcion_continua(
    func: ContinuousFunction,
    iterations: int,
    output_dir: str,
) -> dict:
    """Ejecuta todos los algoritmos sobre una función CEC2022 y guarda gráficos, CSV y TXT."""
    os.makedirs(output_dir, exist_ok=True)

    p_woa = WOAContParams(pop_size=POP_SIZE, iterations=iterations)
    p_abc = ABCContParams(pop_size=POP_SIZE, iterations=iterations)
    p_a = VariantAParams(pop_size=POP_SIZE, iterations=iterations)
    p_b = VariantBParams(pop_size=POP_SIZE, iterations=iterations)
    p_c = VariantCParams(pop_size=POP_SIZE, iterations=iterations)
    p_mdg = MDGWABCParams(pop_size=POP_SIZE, iterations=iterations)
    p_d = DTWWOAABCParams(pop_size=POP_SIZE, iterations=iterations)

    dict_res = {}
    algos = [
        ("WOA Puro", woa_cont_epoch, p_woa),
        ("ABC Puro", abc_cont_epoch, p_abc),
        ("Variante A", variant_a_epoch_continuo, p_a),
        ("Variante B", variant_b_epoch_continuo, p_b),
        ("Variante C", variant_c_epoch_continuo, p_c),
        ("MDG-WABC (B+C)", mdg_wabc_epoch_continuo, p_mdg),
        ("Variante D (DTW)", variant_d_epoch_continuo, p_d),
    ]

    for name, fn_epoch, params in algos:
        print(f"\n    ---> Ejecutando {name} en {func.name} ({iterations} iters)...")
        dict_res[name] = fn_epoch(func, params, verbose=True)

    # TXT Individual
    txt_path = os.path.join(output_dir, "resumen_funcion.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Función            : {func.name}\n")
        f.write(f"Dimensión          : {func.n_dim}\n")
        f.write(f"Rango              : [{func.lb}, {func.ub}]\n")
        f.write(f"Óptimo conocido    : {func.optimum:.4f}\n\n")
        f.write(f"{'Algoritmo':<20} {'Mejor Valor':>14} {'Gap c/ Óptimo':>15}\n")
        f.write("-" * 52 + "\n")
        for name, r in dict_res.items():
            gap = abs(r.mejor_valor - func.optimum)
            f.write(f"{name:<20} {r.mejor_valor:>14.4f} {gap:>15.4f}\n")

    # CSV Historial
    csv_path = os.path.join(output_dir, "historial_convergencia.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteracion"] + list(dict_res.keys()))
        max_len = max(len(r.historial) for r in dict_res.values())
        for i in range(max_len):
            row = [i]
            for r in dict_res.values():
                val = r.historial[i] if i < len(r.historial) else r.historial[-1]
                row.append(val)
            writer.writerow(row)

    # Gráfico 1: Convergencia Continuo (Minimización)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in dict_res.items():
        color = COLORES_ALGO.get(name, "#333333")
        ax.plot(r.historial, label=name, color=color, linewidth=2)
    ax.axhline(func.optimum, color="red", linestyle="--", alpha=0.7, label=f"Óptimo ({func.optimum:.1f})")
    ax.set_title(f"Convergencia CEC2022 - {func.name} (Dim={func.n_dim})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Fitness (Minimización)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_convergencia_comparativa.png"), dpi=150)
    plt.close(fig)

    # Gráfico 2: Instantáneo
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in dict_res.items():
        color = COLORES_ALGO.get(name, "#333333")
        if hasattr(r, "historial_inst") and r.historial_inst:
            ax.plot(r.historial_inst, label=name, color=color, alpha=0.5, linewidth=1)
    ax.set_title(f"Fitness Instantáneo CEC2022 - {func.name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Fitness de la Iteración")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_instantaneo_comparativo.png"), dpi=150)
    plt.close(fig)

    mejores_valores = {k: v.mejor_valor for k, v in dict_res.items()}
    ganador = min(mejores_valores, key=mejores_valores.get)

    return {
        "nombre": func.name,
        "n_dim": func.n_dim,
        "optimo": func.optimum,
        "ganador": ganador,
        "mejor_valor": mejores_valores[ganador],
        "detalles": mejores_valores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. EJECUTOR PRINCIPAL DEL BENCHMARK UNIFICADO
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_benchmark_completo():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_BASE, f"run_{timestamp}")

    # Carpetas separadas solicitadas por el usuario
    mkp_root_dir = os.path.join(run_dir, "mkp")
    cec_root_dir = os.path.join(run_dir, "cec2022")

    os.makedirs(mkp_root_dir, exist_ok=True)
    os.makedirs(cec_root_dir, exist_ok=True)

    banner = "=" * 80
    print(f"\n{banner}")
    print(" BENCHMARK UNIFICADO MEZCLAS WOA-ABC (MKP + CEC2022)")
    print(banner)
    print(f"  Carpeta de la corrida : {run_dir}")
    print(f"    - Subcarpeta MKP    : {mkp_root_dir}")
    print(f"    - Subcarpeta CEC22  : {cec_root_dir}")
    print(f"  Iteraciones / exp    : {ITERACIONES}")
    print(banner)

    # -------------------------------------------------------------------------
    # PARTE A: PROBLEMA DISCRETO MKP
    # -------------------------------------------------------------------------
    raw_instancias = cargar_instancias("instancias/mknapcb1.txt")
    instancias_eval = raw_instancias[:NUM_INSTANCIAS_MKP]

    print(f"\n[1/2] PROCESANDO {len(instancias_eval)} INSTANCIAS MKP...")
    resumen_mkp: list[dict] = []

    for idx, inst_dict in enumerate(instancias_eval, 1):
        inst = MKPInstance.from_dict(inst_dict)
        name = f"Instancia_{idx:02d}_n{inst.n}_m{inst.m}"
        print(f"  -> [{idx}/{len(instancias_eval)}] {name}...")
        inst_dir = os.path.join(mkp_root_dir, name)
        res = procesar_instancia_mkp(inst, name, ITERACIONES, inst_dir)
        resumen_mkp.append(res)
        print(f"         Ganador: {res['ganador']} (Valor: {res['mejor_valor']:.1f})")

    # Resúmenes Raíz MKP
    # 1. TXT MKP
    txt_mkp = os.path.join(mkp_root_dir, "resumen_mkp.txt")
    with open(txt_mkp, "w", encoding="utf-8") as f:
        f.write("RESUMEN GLOBAL MKP - WOA-ABC\n")
        f.write(f"{'Instancia':<25} {'WOA':>10} {'ABC':>10} {'Var A':>10} {'Var B':>10} {'Var C':>10} {'MDG-WABC':>10} {'Var D':>10} {'Ganador':>15}\n")
        f.write("-" * 118 + "\n")
        for r in resumen_mkp:
            d = r["detalles"]
            f.write(f"{r['nombre']:<25} {d['WOA Puro']:>10.1f} {d['ABC Puro']:>10.1f} {d['Variante A']:>10.1f} "
                    f"{d['Variante B']:>10.1f} {d['Variante C']:>10.1f} {d['MDG-WABC (B+C)']:>10.1f} {d['Variante D (DTW)']:>10.1f} {r['ganador']:>15}\n")

    # 2. CSV MKP
    csv_mkp = os.path.join(mkp_root_dir, "resumen_mkp.csv")
    with open(csv_mkp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instancia", "n", "m", "WOA_Puro", "ABC_Puro", "Variante_A", "Variante_B", "Variante_C", "MDG_WABC", "Variante_D_DTW", "Ganador"])
        for r in resumen_mkp:
            d = r["detalles"]
            writer.writerow([
                r["nombre"], r["n"], r["m"],
                d["WOA Puro"], d["ABC Puro"], d["Variante A"],
                d["Variante B"], d["Variante C"], d["MDG-WABC (B+C)"],
                d["Variante D (DTW)"], r["ganador"]
            ])

    # 3. MD MKP
    md_mkp = os.path.join(mkp_root_dir, "resumen_mkp.md")
    with open(md_mkp, "w", encoding="utf-8") as f:
        f.write(f"# Resumen MKP - Benchmark WOA-ABC ({timestamp})\n\n")
        f.write("| Instancia | WOA Puro | ABC Puro | Var A | Var B | Var C | MDG-WABC | Var D (DTW) | Ganador |\n")
        f.write("|-----------|----------|----------|-------|-------|-------|----------|-------------|---------|\n")
        for r in resumen_mkp:
            d = r["detalles"]
            f.write(f"| `{r['nombre']}` | {d['WOA Puro']:.1f} | {d['ABC Puro']:.1f} | {d['Variante A']:.1f} | "
                    f"{d['Variante B']:.1f} | {d['Variante C']:.1f} | {d['MDG-WABC (B+C)']:.1f} | **{d['Variante D (DTW)']:.1f}** | `{r['ganador']}` |\n")

    print(f"  [mkp] Carpetas e informes creados en '{mkp_root_dir}'")

    # -------------------------------------------------------------------------
    # PARTE B: BENCHMARK CONTINUO (CEC2022)
    # -------------------------------------------------------------------------
    funciones = get_test_functions(DIMENSION_CEC)
    print(f"\n[2/2] PROCESANDO {len(funciones)} FUNCIONES CONTINUAS CEC2022...")
    resumen_cec: list[dict] = []

    for idx, func in enumerate(funciones, 1):
        print(f"  -> [{idx:2d}/{len(funciones)}] {func.name}...")
        func_dir = os.path.join(cec_root_dir, func.name)
        res = procesar_funcion_continua(func, ITERACIONES, func_dir)
        resumen_cec.append(res)
        print(f"         Ganador: {res['ganador']} (Valor: {res['mejor_valor']:.4f})")

    # Resúmenes Raíz CEC2022
    # 1. TXT CEC
    txt_cec = os.path.join(cec_root_dir, "resumen_global.txt")
    with open(txt_cec, "w", encoding="utf-8") as f:
        f.write("RESUMEN GLOBAL CEC2022 - WOA-ABC\n")
        f.write(f"{'Función':<30} {'Óptimo':>10} {'WOA':>10} {'ABC':>10} {'Var A':>10} {'Var B':>10} {'Var C':>10} {'MDG-WABC':>10} {'Var D':>10} {'Ganador':>15}\n")
        f.write("-" * 128 + "\n")
        for r in resumen_cec:
            d = r["detalles"]
            f.write(f"{r['nombre']:<30} {r['optimo']:>10.1f} "
                    f"{d['WOA Puro']:>10.2f} {d['ABC Puro']:>10.2f} {d['Variante A']:>10.2f} "
                    f"{d['Variante B']:>10.2f} {d['Variante C']:>10.2f} {d['MDG-WABC (B+C)']:>10.2f} {d['Variante D (DTW)']:>10.2f} {r['ganador']:>15}\n")

    # 2. CSV CEC
    csv_cec = os.path.join(cec_root_dir, "resumen_global.csv")
    with open(csv_cec, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["funcion", "dim", "optimo", "WOA_Puro", "ABC_Puro", "Variante_A", "Variante_B", "Variante_C", "MDG_WABC", "Variante_D_DTW", "Ganador"])
        for r in resumen_cec:
            d = r["detalles"]
            writer.writerow([
                r["nombre"], r["n_dim"], r["optimo"],
                d["WOA Puro"], d["ABC Puro"], d["Variante A"],
                d["Variante B"], d["Variante C"], d["MDG-WABC (B+C)"],
                d["Variante D (DTW)"], r["ganador"]
            ])

    # 3. MD CEC
    md_cec = os.path.join(cec_root_dir, "resumen_global.md")
    with open(md_cec, "w", encoding="utf-8") as f:
        f.write(f"# Resumen CEC2022 - Benchmark WOA-ABC ({timestamp})\n\n")
        f.write("| # | Función | Óptimo | WOA Puro | ABC Puro | Var A | Var B | Var C | MDG-WABC | Var D (DTW) | Ganador |\n")
        f.write("|---|---------|--------|----------|----------|-------|-------|-------|----------|-------------|---------|\n")
        for i, r in enumerate(resumen_cec, 1):
            d = r["detalles"]
            f.write(f"| {i} | `{r['nombre']}` | {r['optimo']:.1f} | {d['WOA Puro']:.2f} | {d['ABC Puro']:.2f} | "
                    f"{d['Variante A']:.2f} | {d['Variante B']:.2f} | {d['Variante C']:.2f} | {d['MDG-WABC (B+C)']:.2f} | {d['Variante D (DTW)']:.2f} | `{r['ganador']}` |\n")

    print(f"  [cec2022] Carpetas e informes creados en '{cec_root_dir}'")

    # -------------------------------------------------------------------------
    # INFORME UNIFICADO EN LA RAÍZ DEL RUN
    # -------------------------------------------------------------------------
    md_run = os.path.join(run_dir, "resumen_general.md")
    with open(md_run, "w", encoding="utf-8") as f:
        f.write(f"# Resumen General de Corrida: WOA-ABC ({timestamp})\n\n")
        f.write(f"Se completó la evaluación unificada conteniendo 2 módulos independientes:\n")
        f.write(f"1. **Módulo MKP Discreto**: [`mkp/`](file:///{os.path.abspath(mkp_root_dir)})\n")
        f.write(f"2. **Módulo Continuo CEC2022**: [`cec2022/`](file:///{os.path.abspath(cec_root_dir)})\n\n")
        f.write("### Resumen Rápido MKP\n")
        f.write("| Instancia | Ganador | Mejor Valor |\n")
        f.write("|---|---|---|\n")
        for r in resumen_mkp:
            f.write(f"| `{r['nombre']}` | `{r['ganador']}` | {r['mejor_valor']:.1f} |\n")
        f.write("\n### Resumen Rápido CEC2022\n")
        f.write("| Función | Ganador | Mejor Valor |\n")
        f.write("|---|---|---|\n")
        for r in resumen_cec:
            f.write(f"| `{r['nombre']}` | `{r['ganador']}` | {r['mejor_valor']:.4f} |\n")

    print(f"\n{banner}")
    print(" BENCHMARK UNIFICADO COMPLETADO EXITOSAMENTE")
    print(f" Resumen general guardado en: '{md_run}'")
    print(banner + "\n")


if __name__ == "__main__":
    ejecutar_benchmark_completo()
