"""
reconstruir_boxplots_cec.py
───────────────────────────
Regenera los resúmenes y boxplots descriptivos de benchmarks CEC2022.

Las funciones CEC tienen escalas y óptimos diferentes. Este script no las
trata como algoritmos ni ejecuta Friedman/Wilcoxon entre funciones; las
pruebas inferenciales se generan por función durante el benchmark principal.
"""

import os
import glob
import csv
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from continuous_benchmark.analisis_estadistico import ajustar_pvalues_holm


def get_cec_index_from_name(name: str) -> int:
    """Extrae el número de función 1..12 a partir de nombres como 'F1_F1_...', 'F12_...', 'CEC_01_...'."""
    import re
    m = re.search(r'(?:CEC_0?|F0?)([0-9]{1,2})', name)
    if m:
        return int(m.group(1))
    return 99


def actualizar_reporte_holm(func_dir: str, expected_reference: str) -> None:
    """Migra un reporte antiguo para mostrar el p bruto y el p Holm.

    Los reportes históricos ya contienen los p-valores de Wilcoxon, por lo que
    no es necesario repetir las 31 ejecuciones para aplicar Holm. Los nuevos
    reportes producidos por ``realizar_analisis_estadistico`` ya incluyen estas
    columnas y se dejan intactos.
    """
    csv_path = os.path.join(func_dir, "analisis_estadistico_pvalues.csv")
    md_path = os.path.join(func_dir, "analisis_estadistico_pvalues.md")
    if not (os.path.exists(csv_path) and os.path.exists(md_path)):
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    already_adjusted = (
        "wilcoxon_pvalue_raw" in fieldnames
        and "wilcoxon_pvalue_holm" in fieldnames
    )
    # Algunos reportes DDTW antiguos quedaron etiquetados como Hybrid DTW.
    # La carpeta de la corrida es la fuente inequívoca para corregirlo.
    if expected_reference == "Hybrid DDTW":
        for row in rows:
            if row.get("algoritmo") == "Hybrid DTW":
                row["algoritmo"] = expected_reference

    raw_key = "wilcoxon_pvalue_raw" if "wilcoxon_pvalue_raw" in fieldnames else "wilcoxon_pvalue"
    if "wilcoxon_pvalue_holm" not in fieldnames:
        raw_pos = fieldnames.index(raw_key)
        fieldnames[raw_pos] = "wilcoxon_pvalue_raw"
        fieldnames.insert(raw_pos + 1, "wilcoxon_pvalue_holm")
        for row in rows:
            row["wilcoxon_pvalue_raw"] = row.pop(raw_key)
            row["wilcoxon_pvalue_holm"] = "1.000000e+00"

    reference = next(
        (row for row in rows if row.get("algoritmo", "").startswith("Hybrid ")),
        None,
    )
    if reference is None:
        return
    reference_name = reference["algoritmo"]
    comparables = [row for row in rows if row["algoritmo"] != reference_name]
    adjusted = ajustar_pvalues_holm([
        float(row["wilcoxon_pvalue_raw"]) for row in comparables
    ])
    for row, p_holm in zip(comparables, adjusted):
        row["wilcoxon_pvalue_holm"] = f"{p_holm:.6e}"
    reference["wilcoxon_pvalue_holm"] = "1.000000e+00"

    ref_mean = float(reference["media"])
    for row in rows:
        if row["algoritmo"] == reference_name:
            row["significancia"] = "="
            continue
        p_holm = float(row["wilcoxon_pvalue_holm"])
        if p_holm < 0.001:
            symbol = "***"
        elif p_holm < 0.01:
            symbol = "**"
        elif p_holm < 0.05:
            symbol = "*"
        else:
            symbol = "ns"
        # La etiqueta se expresa desde la perspectiva del híbrido de
        # referencia: en minimización, una media mayor del comparador implica
        # que el híbrido es mejor.
        direction = "Mejor (+)" if float(row["media"]) > ref_mean else "Peor (-)"
        row["significancia"] = f"{direction} {symbol}" if p_holm < 0.05 else f"Similar (=) {symbol}"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Actualizar también el Markdown histórico sin cambiar sus estadísticas.
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    updated = []
    for line in lines:
        if expected_reference == "Hybrid DDTW":
            line = line.replace("Hybrid DTW", "Hybrid DDTW")
        if line.startswith("- **Corrección por comparaciones múltiples:**"):
            line = (f"- **Corrección por comparaciones múltiples:** Holm sobre "
                    f"{len(comparables)} comparaciones pareadas contra la referencia.")
        elif line.startswith("| Rank | Algoritmo |") and not already_adjusted:
            line = ("| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | "
                    "Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |")
        elif line.startswith("|------|-----------|") and not already_adjusted:
            line = ("|------|-----------|-----------|-------|-----|---------|--------|"
                    "-----------|------------------|-----------------|---------------|")
        elif line.startswith("| ") and not line.startswith("|---"):
            parts = line.split("|")
            if len(parts) >= 12 and parts[1].strip().isdigit():
                algorithm = re.sub(r"[*`]", "", parts[2]).strip()
                row = next((r for r in rows if r["algoritmo"] == algorithm), None)
                if row is not None:
                    if not already_adjusted:
                        parts[9] = f" `{float(row['wilcoxon_pvalue_raw']):.4e}` "
                        parts.insert(10, f" `{float(row['wilcoxon_pvalue_holm']):.4e}` ")
                    parts[11] = f" **{row['significancia']}** "
                    line = "|".join(parts)
        updated.append(line)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated) + "\n")


def regenerar_benchmark_continuo(run_dir: str) -> None:
    todos_runs_path = os.path.join(run_dir, "todos_los_runs.csv")

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

    # Armar un resumen descriptivo con etiquetas limpias: "CEC 1", ...
    resumen_global_dict = []

    for fn_raw in funciones_sorted:
        idx = get_cec_index_from_name(fn_raw)
        label = f"CEC {idx}"
        vals = df[df["funcion"] == fn_raw]["mejor_valor"].tolist()
        v_opt = float(df[df["funcion"] == fn_raw]["valor_optimo"].iloc[0])
        denom = abs(v_opt) if v_opt != 0 else 1.0
        gaps = [100.0 * (float(v) - v_opt) / denom for v in vals]

        resumen_global_dict.append({
            "nombre": label,
            "nombre_original": fn_raw,
            "n_runs": len(vals),
            "valores_runs": vals,
            "gaps_runs": gaps,
            "media": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "mediana": float(np.median(vals)),
            "mejor": float(np.min(vals)),
            "peor": float(np.max(vals)),
            "valor_optimo": v_opt,
            "gap_media_pct": float(np.mean(gaps)),
            "gap_mejor_pct": float(np.min(gaps)),
        })

    # Migrar los reportes por función que fueron generados antes de incorporar
    # Holm. Esto solo transforma los p-valores ya calculados; no reejecuta el
    # benchmark ni altera las observaciones experimentales.
    expected_reference = (
        "Hybrid DDTW" if "ddtw" in os.path.basename(run_dir).lower()
        else "Hybrid DTW"
    )
    for func_dir in glob.glob(os.path.join(run_dir, "CEC_*")):
        actualizar_reporte_holm(func_dir, expected_reference)

    # Boxplot global descriptivo en error normalizado. Así la visualización no
    # induce a comparar directamente escalas incompatibles entre funciones.
    n = len(resumen_global_dict)
    nombres_global = [r["nombre"] for r in resumen_global_dict]
    datos_global = [r["gaps_runs"] for r in resumen_global_dict]

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

    ax.set_title(f"Global Descriptive Boxplot — {len(datos_global[0])} Runs per CEC2022 Function",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized objective error / gap (%)", fontsize=10)
    ax.set_xlabel("CEC2022 Benchmark Functions", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_global_png = os.path.join(run_dir, "boxplot_global.png")
    plt.savefig(out_global_png, dpi=300)
    plt.close(fig)
    print(f"  [OK] Regenerado: {out_global_png}")

    # Resumen global exclusivamente descriptivo. El nombre histórico de los
    # archivos se conserva para no romper referencias existentes.
    csv_path = os.path.join(run_dir, "analisis_estadistico_global.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "funcion", "n_runs", "media", "std", "mediana", "mejor", "peor",
            "valor_optimo", "gap_media_pct", "gap_mejor_pct",
            "resultado_frente_optimo",
        ])
        for r in resumen_global_dict:
            writer.writerow([
                r["nombre"], r["n_runs"], r["media"], r["std"], r["mediana"],
                r["mejor"], r["peor"], r["valor_optimo"],
                r["gap_media_pct"], r["gap_mejor_pct"],
                "Óptimo alcanzado" if r["gap_media_pct"] <= 0.001
                else "Peor que el óptimo",
            ])

    md_path = os.path.join(run_dir, "analisis_estadistico_global.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Resumen global descriptivo — CEC2022\n\n")
        f.write("> No se ejecutan pruebas Friedman/Wilcoxon entre F1–F12. "
                "Las funciones tienen escalas y óptimos diferentes; las "
                "pruebas inferenciales válidas están en cada carpeta CEC.\n\n")
        f.write("| Función | Runs | Media | Std | Mediana | Mejor | Peor | Óptimo | Gap media (%) | Gap mejor (%) | Resultado frente al óptimo |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in resumen_global_dict:
            f.write(
                f"| `{r['nombre']}` | {r['n_runs']} | {r['media']:.6f} | "
                f"{r['std']:.6f} | {r['mediana']:.6f} | {r['mejor']:.6f} | "
                f"{r['peor']:.6f} | {r['valor_optimo']:.6f} | "
                f"{r['gap_media_pct']:.3f} | {r['gap_mejor_pct']:.3f} | "
                f"{'Óptimo alcanzado' if r['gap_media_pct'] <= 0.001 else 'Peor que el óptimo'} |\n"
            )
        f.write("\nUsar `CEC_XX_*/analisis_estadistico_pvalues.md` para las "
                "comparaciones híbrido vs. MH standalone.\n")
    print(f"  [OK] Regenerado resumen global descriptivo: {md_path}")


def main():
    # El benchmark principal escribe en continuous_benchmark/resultados.
    # Resolver desde este archivo permite ejecutar el script desde cualquier
    # directorio del proyecto.
    project_dir = os.path.dirname(os.path.abspath(__file__))
    base_dirs = [
        os.path.join(project_dir, "continuous_benchmark", "resultados"),
        os.path.join(project_dir, "resultados finales"),
    ]
    run_dirs = []
    for base_dir in base_dirs:
        run_dirs.extend(glob.glob(os.path.join(base_dir, "run_*")))
    if not run_dirs:
        print(f"No se encontraron carpetas run_* en {base_dir}")
        return

    for d in run_dirs:
        regenerar_benchmark_continuo(d)


if __name__ == "__main__":
    main()
