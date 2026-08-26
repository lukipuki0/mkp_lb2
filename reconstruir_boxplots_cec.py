"""
reconstruir_boxplots_cec.py
───────────────────────────
Regenera los boxplots globales y análisis estadísticos de benchmarks continuos CEC2022
existentes usando nombres limpios ('CEC 1', 'CEC 2', ..., 'CEC 12').
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from continuous_benchmark.analisis_estadistico import realizar_analisis_estadistico


def get_cec_index_from_name(name: str) -> int:
    """Extrae el número de función 1..12 a partir de nombres como 'F1_F1_...', 'F12_...', 'CEC_01_...'."""
    import re
    m = re.search(r'(?:CEC_0?|F0?)([0-9]{1,2})', name)
    if m:
        return int(m.group(1))
    return 99


def regenerar_benchmark_continuo(run_dir: str) -> None:
    todos_runs_path = os.path.join(run_dir, "todos_los_runs.csv")
    resumen_path = os.path.join(run_dir, "resumen_global.csv")

    if not os.path.exists(todos_runs_path):
        print(f"  [SKIP] No se encontro {todos_runs_path}")
        return

    print(f"\n=======================================================")
    print(f"  Procesando carpeta: {run_dir}")
    print(f"=======================================================")

    df = pd.read_csv(todos_runs_path)
    
    # Extraer funciones únicas y ordenarlas por su índice CEC
    funciones_raw = list(df["funcion"].unique())
    funciones_sorted = sorted(funciones_raw, key=get_cec_index_from_name)

    # Armar diccionario con etiquetas limpias: "CEC 1", "CEC 2", ...
    resultados_multi = {}
    resumen_global_dict = []

    for fn_raw in funciones_sorted:
        idx = get_cec_index_from_name(fn_raw)
        label = f"CEC {idx}"
        vals = df[df["funcion"] == fn_raw]["mejor_valor"].tolist()
        v_opt = float(df[df["funcion"] == fn_raw]["valor_optimo"].iloc[0])
        resultados_multi[label] = vals

        resumen_global_dict.append({
            "nombre": label,
            "nombre_original": fn_raw,
            "n_runs": len(vals),
            "valores_runs": vals,
            "media": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "mediana": float(np.median(vals)),
            "mejor": float(np.min(vals)),
            "peor": float(np.max(vals)),
            "valor_optimo": v_opt,
        })

    # 1. Regenerar boxplot_global.png (en orden numérico CEC 1 .. CEC 12)
    n = len(resumen_global_dict)
    nombres_global = [r["nombre"] for r in resumen_global_dict]
    datos_global = [r["valores_runs"] for r in resumen_global_dict]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 6))
    colores = plt.cm.tab20.colors

    bp = ax.boxplot(
        datos_global,
        patch_artist=True,
        tick_labels=nombres_global,
        widths=0.55,
        medianprops=dict(color="#FF5722", linewidth=2.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
    )
    for patch, color in zip(bp["boxes"], colores[:n]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(f"Global Boxplot — {len(datos_global[0])} Runs per CEC2022 Function",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Best Value (Minimization)", fontsize=10)
    ax.set_xlabel("CEC2022 Benchmark Functions", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_global_png = os.path.join(run_dir, "boxplot_global.png")
    plt.savefig(out_global_png, dpi=300)
    plt.close(fig)
    print(f"  [OK] Regenerado: {out_global_png}")

    # 2. Regenerar análisis estadístico y boxplot_comparativo_funciones.png (orden Friedman)
    referencia_global = "CEC 1"
    realizar_analisis_estadistico(
        resultados_dict      = resultados_multi,
        output_dir           = run_dir,
        algoritmo_referencia = referencia_global,
        metrica_label        = "Fitness (Minimización CEC2022)",
        titulo_benchmark     = f"CEC2022 Global ({len(resultados_multi)} funciones)",
        minimizacion         = True,
        boxplot_filename     = "boxplot_comparativo_funciones.png",
        csv_filename         = "analisis_estadistico_global.csv",
        md_filename          = "analisis_estadistico_global.md",
    )
    print(f"  [OK] Regenerado analisis_estadistico_global y boxplot_comparativo_funciones.png")


def main():
    base_dir = os.path.join("resultados", "benchmark_continuo")
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if not run_dirs:
        print(f"No se encontraron carpetas run_* en {base_dir}")
        return

    for d in run_dirs:
        regenerar_benchmark_continuo(d)


if __name__ == "__main__":
    main()
