"""
HRES2-H2/analisis_estadistico_hres2.py
--------------------------------------
Módulo de Análisis Estadístico Inferencial y Pruebas de Hipótesis para HRES2-H2.
Calcula p-valores (Wilcoxon, Mann-Whitney U, Friedman), Normalidad (Shapiro-Wilk)
y métricas comparativas entre el Pipeline Híbrido DTW y las MHs individuales (31 runs).
"""

from __future__ import annotations

import os
import csv
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def realizar_analisis_estadistico_completo(
    resultados_dict: dict[str, list[float]],
    output_dir     : str,
    algoritmo_referencia: str = "Hybrid DTW",
) -> dict:
    """
    Realiza pruebas estadísticas completas sobre los 31 runs de cada algoritmo.

    Parameters
    ----------
    resultados_dict : dict {nombre_algoritmo: lista_31_valores_lcoe}
    output_dir      : carpeta de destino para artefactos y tablas
    algoritmo_referencia : nombre de la MH o pipeline base contra el que comparar p-valores

    Returns
    -------
    dict con resúmenes de p-valores, rrankings y tablas estadísticas.
    """
    os.makedirs(output_dir, exist_ok=True)
    nombres_algs = list(resultados_dict.keys())
    ref_vals = resultados_dict[algoritmo_referencia]
    n_runs = len(ref_vals)

    print("\n" + "=" * 85)
    print(f"  ANÁLISIS ESTADÍSTICO Y INFERENCIAL DE RESULTADOS ({n_runs} RUNS)")
    print(f"  Algoritmo de Referencia (Control): {algoritmo_referencia}")
    print("=" * 85)

    # ── 1. Pruebas Descriptivas y de Normalidad (Shapiro-Wilk) ───────────────
    tabla_resumen = []
    data_matrix = []

    for alg in nombres_algs:
        vals = np.array(resultados_dict[alg])
        data_matrix.append(vals)

        mean_v   = float(np.mean(vals))
        std_v    = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        se_v     = std_v / np.sqrt(len(vals))
        median_v = float(np.median(vals))
        iqr_v    = float(stats.iqr(vals))
        min_v    = float(np.min(vals))
        max_v    = float(np.max(vals))
        ci95_low = mean_v - 1.96 * se_v
        ci95_high= mean_v + 1.96 * se_v

        # Test de Normalidad Shapiro-Wilk
        if len(np.unique(vals)) > 1:
            stat_sw, p_sw = stats.shapiro(vals)
        else:
            stat_sw, p_sw = 1.0, 1.0

        # Comparación vs Algoritmo de Referencia (Wilcoxon & Mann-Whitney U)
        if alg == algoritmo_referencia:
            stat_wilc, p_wilc = np.nan, 1.0
            stat_mwu, p_mwu   = np.nan, 1.0
            sig_label = "="
        else:
            # Wilcoxon Signed-Rank Test (Muestras pareadas de 31 runs)
            if np.array_equal(vals, ref_vals):
                stat_wilc, p_wilc = 0.0, 1.0
            else:
                try:
                    stat_wilc, p_wilc = stats.wilcoxon(ref_vals, vals)
                except Exception:
                    stat_wilc, p_wilc = np.nan, 1.0

            # Mann-Whitney U Test (Muestras independientes)
            try:
                stat_mwu, p_mwu = stats.mannwhitneyu(ref_vals, vals, alternative="two-sided")
            except Exception:
                stat_mwu, p_mwu = np.nan, 1.0

            # Etiqueta de significancia (alpha = 0.05)
            if p_wilc < 0.001:
                sig_symbol = "***"
            elif p_wilc < 0.01:
                sig_symbol = "**"
            elif p_wilc < 0.05:
                sig_symbol = "*"
            else:
                sig_symbol = "ns"

            if mean_v > float(np.mean(ref_vals)) and p_wilc < 0.05:
                sig_label = f"Peor (-) {sig_symbol}"
            elif mean_v < float(np.mean(ref_vals)) and p_wilc < 0.05:
                sig_label = f"Mejor (+) {sig_symbol}"
            else:
                sig_label = f"Similar (=) {sig_symbol}"


        tabla_resumen.append({
            "algoritmo": alg,
            "mean": mean_v,
            "std": std_v,
            "median": median_v,
            "iqr": iqr_v,
            "min": min_v,
            "max": max_v,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "shapiro_p": p_sw,
            "wilcoxon_p": p_wilc,
            "mwu_p": p_mwu,
            "significancia": sig_label,
        })

    # ── 2. Test de Friedman (Ranking Global No Paramétrico) ──────────────────
    try:
        stat_fried, p_fried = stats.friedmanchisquare(*data_matrix)
    except Exception:
        stat_fried, p_fried = np.nan, 1.0

    # Calcular Ranks Promedio (Menor LCOE -> Rank 1)
    all_runs_matrix = np.array(data_matrix) # (N_algs, 31)
    ranks_matrix = np.zeros_like(all_runs_matrix)
    for r in range(n_runs):
        col_vals = all_runs_matrix[:, r]
        ranks_matrix[:, r] = stats.rankdata(col_vals)

    mean_ranks = np.mean(ranks_matrix, axis=1)
    for idx, alg_dict in enumerate(tabla_resumen):
        alg_dict["mean_rank"] = float(mean_ranks[idx])

    # Ordenar por Mean Rank
    tabla_resumen.sort(key=lambda x: x["mean_rank"])

    # ── 3. Impresión por Consola ──────────────────────────────────────────────
    print(f"\n--- TEST DE FRIEDMAN GLOBAL ---")
    print(f"  Estadístico Chi2 : {stat_fried:.4f}")
    print(f"  p-value          : {p_fried:.6e} " + ("(Diferencia Global Significativa p < 0.05)" if p_fried < 0.05 else "(Sin Diferencia Significativa)"))
    print("\n--- RANKING NO PARAMÉTRICO Y PRUEBAS PAR A PAR VS CONTROL ---")
    header = f"{'Rank':<5} {'Algoritmo':<15} {'Mean LCOE':>12} {'Std':>10} {'Median':>12} {'Shapiro p':>12} {'Wilcoxon p':>12} {'Significancia':<16}"
    print(header)
    print("-" * len(header))
    for r_idx, d in enumerate(tabla_resumen, 1):
        print(f"{r_idx:<5d} {d['algoritmo']:<15s} {d['mean']:>12.6f} {d['std']:>10.6f} {d['median']:>12.6f} {d['shapiro_p']:>12.4e} {d['wilcoxon_p']:>12.4e} {d['significancia']:<16s}")

    # ── 4. Generar Gráfico de Boxplot Comparativo Multi-Algoritmo ────────────
    plt.figure(figsize=(12, 6))
    algs_sorted = [d["algoritmo"] for d in tabla_resumen]
    vals_sorted = [resultados_dict[alg] for alg in algs_sorted]

    bp = plt.boxplot(vals_sorted, patch_artist=True, tick_labels=algs_sorted, widths=0.5)
    colors = plt.cm.tab10(np.linspace(0, 1, len(algs_sorted)))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    plt.title(f"LCOE Statistical Comparison across {n_runs} Runs (Friedman Ranking)", fontsize=13, fontweight="bold")
    plt.ylabel("LCOE (CNY/kWh)", fontsize=11)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    boxplot_comp_path = os.path.join(output_dir, "mhs_comparative_boxplot.png")
    plt.savefig(boxplot_comp_path, dpi=300)
    plt.close()
    print(f"\n  [plot] {boxplot_comp_path}")

    # ── 5. Guardar CSV Estadístico ───────────────────────────────────────────
    csv_stat_path = os.path.join(output_dir, "analisis_estadistico_pvalues.csv")
    with open(csv_stat_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "algoritmo", "mean_rank", "mean_lcoe", "std_lcoe",
            "median_lcoe", "iqr_lcoe", "ci95_low", "ci95_high",
            "shapiro_pvalue", "wilcoxon_pvalue", "mannwhitney_pvalue", "significancia"
        ])
        for r_idx, d in enumerate(tabla_resumen, 1):
            writer.writerow([
                r_idx, d["algoritmo"], f"{d['mean_rank']:.2f}",
                f"{d['mean']:.6f}", f"{d['std']:.6f}", f"{d['median']:.6f}", f"{d['iqr']:.6f}",
                f"{d['ci95_low']:.6f}", f"{d['ci95_high']:.6f}",
                f"{d['shapiro_p']:.6e}", f"{d['wilcoxon_p']:.6e}", f"{d['mwu_p']:.6e}",
                d["significancia"]
            ])
    print(f"  [csv]  {csv_stat_path}")

    # ── 6. Guardar Reporte Markdown ──────────────────────────────────────────
    md_stat_path = os.path.join(output_dir, "analisis_estadistico_pvalues.md")
    with open(md_stat_path, "w", encoding="utf-8") as f:
        f.write("# Reporte de Análisis Estadístico Inferencial y p-valores (HRES2-H2)\n\n")
        f.write(f"- **Número de Corridas Independientes:** {n_runs}\n")
        f.write(f"- **Algoritmo de Control (Referencia):** `{algoritmo_referencia}`\n")
        f.write(f"- **Test de Friedman Chi2:** `{stat_fried:.4f}` (p-value = `{p_fried:.6e}`)\n\n")
        f.write("## Tabla de Pruebas de Hipótesis y p-valores (Wilcoxon Signed-Rank Test)\n\n")
        f.write("| Rank | Algoritmo | Mean Rank | Mean LCOE | Std | Mediana | Shapiro p-val | Wilcoxon p-val | Significancia vs Control |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r_idx, d in enumerate(tabla_resumen, 1):
            f.write(f"| {r_idx} | **{d['algoritmo']}** | {d['mean_rank']:.2f} | `{d['mean']:.6f}` | `{d['std']:.6f}` | `{d['median']:.6f}` | `{d['shapiro_p']:.4e}` | `{d['wilcoxon_p']:.4e}` | **{d['significancia']}** |\n")
        f.write("\n\n*Leyenda de Significancia:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: No significativo (p >= 0.05)`.\n")
    print(f"  [md]   {md_stat_path}")

    return {
        "friedman_stat": stat_fried,
        "friedman_p": p_fried,
        "tabla_resumen": tabla_resumen,
    }
