"""
continuous_benchmark/analisis_estadistico.py
--------------------------------------------
Módulo de Análisis Estadístico Inferencial para benchmarks continuos CEC2022.

Pruebas implementadas:
  - Shapiro-Wilk     : Normalidad de las distribuciones de cada algoritmo
  - Wilcoxon Signed-Rank : Comparación pareada vs algoritmo de referencia
  - Holm step-down       : Corrección de las comparaciones pareadas
  - Mann-Whitney U   : Comparación independiente vs algoritmo de referencia
  - Friedman         : Ranking global no paramétrico entre todos los algoritmos
  - IC 95%           : Intervalo de confianza para la media de cada algoritmo

Salida:
  - boxplot_comparativo.png   (Boxplots ordenados por Mean Rank Friedman)
  - analisis_estadistico_pvalues.csv
  - analisis_estadistico_pvalues.md
"""

from __future__ import annotations

import os
import csv
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ajustar_pvalues_holm(pvalues: list[float]) -> list[float]:
    """Aplica Holm step-down y devuelve los p-valores en el orden original."""
    adjusted = [float("nan")] * len(pvalues)
    valid = [(i, float(p)) for i, p in enumerate(pvalues) if np.isfinite(p)]
    if not valid:
        return adjusted

    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    running_max = 0.0
    for rank, (original_index, pvalue) in enumerate(ordered, start=1):
        corrected = min(1.0, (m - rank + 1) * pvalue)
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max
    return adjusted


def realizar_analisis_estadistico(
    resultados_dict     : dict[str, list[float]],
    output_dir          : str,
    algoritmo_referencia: str,
    metrica_label       : str  = "Valor",
    titulo_benchmark    : str  = "Benchmark",
    minimizacion        : bool = True,
    boxplot_filename    : str  = "boxplot_comparativo.png",
    csv_filename        : str  = "analisis_estadistico_pvalues.csv",
    md_filename         : str  = "analisis_estadistico_pvalues.md",
) -> dict:
    """
    Realiza análisis estadístico inferencial completo sobre N runs de cada algoritmo.

    Parameters
    ----------
    resultados_dict      : {nombre_algoritmo: lista_N_valores}
    output_dir           : carpeta de destino para artefactos
    algoritmo_referencia : nombre del algoritmo control para comparar p-valores
    metrica_label        : etiqueta de la métrica en gráficos (ej. "LCOE (CNY/kWh)")
    titulo_benchmark     : título del benchmark en reportes (ej. "HRES2-H2")
    minimizacion         : True si menor = mejor (LCOE), False si mayor = mejor (MKP)
    boxplot_filename     : nombre del archivo de boxplot
    csv_filename         : nombre del CSV de resultados
    md_filename          : nombre del Markdown de resultados

    Returns
    -------
    dict con friedman_stat, friedman_p, tabla_resumen ordenada por Mean Rank
    """
    os.makedirs(output_dir, exist_ok=True)

    nombres_algs = list(resultados_dict.keys())
    if algoritmo_referencia not in resultados_dict:
        raise ValueError(f"algoritmo_referencia='{algoritmo_referencia}' no está en resultados_dict.")

    ref_vals = np.array(resultados_dict[algoritmo_referencia])
    n_runs   = len(ref_vals)
    if n_runs < 2:
        raise ValueError("Se requieren al menos 2 runs por algoritmo.")
    if any(len(resultados_dict[alg]) != n_runs for alg in nombres_algs):
        raise ValueError("Todos los algoritmos deben tener el mismo número de runs.")

    print("\n" + "=" * 85)
    print(f"  ANÁLISIS ESTADÍSTICO INFERENCIAL — {titulo_benchmark} ({n_runs} RUNS)")
    print(f"  Referencia: '{algoritmo_referencia}' | Métrica: {metrica_label}")
    print("=" * 85)

    # ── 1. Estadísticas descriptivas y pruebas por algoritmo ────────────────
    tabla_resumen: list[dict] = []
    data_matrix:   list[np.ndarray] = []

    for alg in nombres_algs:
        vals = np.array(resultados_dict[alg])
        data_matrix.append(vals)

        mean_v   = float(np.mean(vals))
        std_v    = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        se_v     = std_v / np.sqrt(len(vals)) if len(vals) > 0 else 0.0
        median_v = float(np.median(vals))
        iqr_v    = float(stats.iqr(vals))
        min_v    = float(np.min(vals))
        max_v    = float(np.max(vals))
        ci95_low  = mean_v - 1.96 * se_v
        ci95_high = mean_v + 1.96 * se_v

        # Shapiro-Wilk
        if len(np.unique(vals)) > 1:
            _, p_sw = stats.shapiro(vals)
        else:
            p_sw = 1.0

        if alg == algoritmo_referencia:
            p_wilc   = 1.0
            p_mwu    = 1.0
            sig_label = "="
        else:
            # Wilcoxon Signed-Rank
            try:
                if np.array_equal(vals, ref_vals):
                    p_wilc = 1.0
                else:
                    _, p_wilc = stats.wilcoxon(ref_vals, vals)
            except Exception:
                p_wilc = 1.0

            # Mann-Whitney U
            try:
                _, p_mwu = stats.mannwhitneyu(ref_vals, vals, alternative="two-sided")
            except Exception:
                p_mwu = 1.0

            # La significancia se determina después de ajustar conjuntamente
            # todas las comparaciones pareadas contra la referencia.
            sig_label = "Pendiente de Holm"

        tabla_resumen.append({
            "algoritmo":  alg,
            "mean":       mean_v,
            "std":        std_v,
            "median":     median_v,
            "iqr":        iqr_v,
            "min":        min_v,
            "max":        max_v,
            "ci95_low":   ci95_low,
            "ci95_high":  ci95_high,
            "shapiro_p":  p_sw,
            "wilcoxon_p": p_wilc,
            "wilcoxon_p_holm": float("nan"),
            "mwu_p":      p_mwu,
            "significancia": sig_label,
        })

    # ── 2. Corrección de Holm dentro de la familia de comparaciones ─────────
    comparables = [
        d for d in tabla_resumen if d["algoritmo"] != algoritmo_referencia
    ]
    pvalues_holm = ajustar_pvalues_holm([d["wilcoxon_p"] for d in comparables])
    for d, p_holm in zip(comparables, pvalues_holm):
        d["wilcoxon_p_holm"] = p_holm

    ref_mean = float(np.mean(ref_vals))
    for d in tabla_resumen:
        if d["algoritmo"] == algoritmo_referencia:
            d["wilcoxon_p_holm"] = 1.0
            d["significancia"] = "="
            continue

        p_holm = d["wilcoxon_p_holm"]
        if p_holm < 0.001:
            sym = "***"
        elif p_holm < 0.01:
            sym = "**"
        elif p_holm < 0.05:
            sym = "*"
        else:
            sym = "ns"

        if minimizacion:
            # La etiqueta se expresa desde la perspectiva de la referencia
            # (Hybrid DTW/DDTW): menor fitness significa que la referencia es
            # mejor cuando el comparador tiene una media mayor.
            referencia_es_mejor = d["mean"] > ref_mean
        else:
            referencia_es_mejor = d["mean"] < ref_mean
        if p_holm < 0.05:
            d["significancia"] = (
                f"Mejor (+) {sym}" if referencia_es_mejor else f"Peor (-) {sym}"
            )
        else:
            d["significancia"] = f"Similar (=) {sym}"

    # ── 3. Test de Friedman ───────────────────────────────────────────────────
    try:
        stat_fried, p_fried = stats.friedmanchisquare(*data_matrix)
    except Exception:
        stat_fried, p_fried = float("nan"), 1.0

    # Mean Ranks por run (menor rank = mejor para minimización)
    all_matrix = np.array(data_matrix)  # (n_algs, n_runs)
    ranks_matrix = np.zeros_like(all_matrix)
    for r in range(n_runs):
        col = all_matrix[:, r]
        if not minimizacion:
            col = -col  # invertir para que menor rank = mayor valor (maximización)
        ranks_matrix[:, r] = stats.rankdata(col)

    mean_ranks = np.mean(ranks_matrix, axis=1)
    for idx, d in enumerate(tabla_resumen):
        d["mean_rank"] = float(mean_ranks[idx])

    # Ordenar por mean_rank ascendente
    tabla_resumen.sort(key=lambda x: x["mean_rank"])

    # ── 4. Consola ────────────────────────────────────────────────────────────
    print(f"\n--- TEST DE FRIEDMAN GLOBAL ---")
    print(f"  Chi2 = {stat_fried:.4f}  |  p-value = {p_fried:.6e}  "
          + ("(Significativo p < 0.05)" if p_fried < 0.05 else "(Sin diferencia global significativa)"))
    print("\n--- RANKING NO PARAMÉTRICO (VS REFERENCIA) ---")
    header = f"{'Rank':<5} {'Algoritmo':<20} {'Media':>14} {'Std':>10} {'Mediana':>12} {'Shapiro p':>12} {'Wilcoxon p(Holm)':>17} {'Significancia':<18}"
    print(header)
    print("-" * len(header))
    for r_idx, d in enumerate(tabla_resumen, 1):
        print(f"{r_idx:<5d} {d['algoritmo']:<20s} {d['mean']:>14.6f} {d['std']:>10.6f} "
              f"{d['median']:>12.6f} {d['shapiro_p']:>12.4e} {d['wilcoxon_p_holm']:>17.4e} {d['significancia']:<18s}")

    # ── 5. Boxplot Comparativo ────────────────────────────────────────────────
    algs_sorted = [d["algoritmo"] for d in tabla_resumen]
    vals_sorted = [resultados_dict[alg] for alg in algs_sorted]

    fig, ax = plt.subplots(figsize=(max(10, len(algs_sorted) * 1.5), 6))
    bp = ax.boxplot(vals_sorted, patch_artist=True, tick_labels=algs_sorted, widths=0.5,
                    medianprops=dict(color="#FF5722", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(algs_sorted)))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    # Marcar el algoritmo de referencia con un borde negro
    for i, alg in enumerate(algs_sorted):
        if alg == algoritmo_referencia:
            bp["boxes"][i].set_linewidth(2.5)
            bp["boxes"][i].set_edgecolor("black")

    ax.set_title(f"Statistical Comparison — {titulo_benchmark} ({n_runs} Runs, Friedman Ranking)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel(metrica_label, fontsize=11)
    ax.set_xticks(range(1, len(algs_sorted) + 1))
    ax.set_xticklabels(algs_sorted, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    boxplot_path = os.path.join(output_dir, boxplot_filename)
    plt.savefig(boxplot_path, dpi=300)
    plt.close(fig)
    print(f"\n  [plot] {boxplot_path}")

    # ── 6. CSV ────────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "algoritmo", "mean_rank", "media", "std",
            "mediana", "iqr", "ci95_low", "ci95_high", "min", "max",
            "shapiro_pvalue", "wilcoxon_pvalue_raw", "wilcoxon_pvalue_holm",
            "mannwhitney_pvalue_raw", "significancia"
        ])
        for r_idx, d in enumerate(tabla_resumen, 1):
            writer.writerow([
                r_idx, d["algoritmo"], f"{d['mean_rank']:.2f}",
                f"{d['mean']:.6f}", f"{d['std']:.6f}", f"{d['median']:.6f}",
                f"{d['iqr']:.6f}", f"{d['ci95_low']:.6f}", f"{d['ci95_high']:.6f}",
                f"{d['min']:.6f}", f"{d['max']:.6f}",
                f"{d['shapiro_p']:.6e}", f"{d['wilcoxon_p']:.6e}",
                f"{d['wilcoxon_p_holm']:.6e}", f"{d['mwu_p']:.6e}",
                d["significancia"]
            ])
    print(f"  [csv]  {csv_path}")

    # ── 7. Markdown ───────────────────────────────────────────────────────────
    md_path = os.path.join(output_dir, md_filename)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Análisis Estadístico Inferencial — {titulo_benchmark}\n\n")
        f.write(f"- **Runs independientes:** {n_runs}\n")
        f.write(f"- **Referencia (control):** `{algoritmo_referencia}`\n")
        f.write(f"- **Métrica:** {metrica_label}\n")
        f.write("- **Desviación estándar:** muestral (`ddof=1`)\n")
        f.write("- **IC 95%:** aproximación normal `media ± 1.96·SE`\n")
        f.write(f"- **Corrección por comparaciones múltiples:** Holm sobre "
                f"{len(comparables)} comparaciones pareadas contra la referencia.\n")
        f.write(f"- **Friedman χ²:** `{stat_fried:.4f}`  |  p-value = `{p_fried:.6e}`")
        f.write("  ✅ Diferencia significativa\n\n" if p_fried < 0.05 else "  ❌ Sin diferencia global significativa\n\n")
        f.write("## Tabla de Resultados y p-valores\n\n")
        f.write("| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |\n")
        f.write("|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|\n")
        for r_idx, d in enumerate(tabla_resumen, 1):
            ic = f"[{d['ci95_low']:.4f}, {d['ci95_high']:.4f}]"
            bold = "**" if d["algoritmo"] == algoritmo_referencia else ""
            f.write(f"| {r_idx} | {bold}`{d['algoritmo']}`{bold} | {d['mean_rank']:.2f} | "
                    f"`{d['mean']:.6f}` | `{d['std']:.6f}` | `{d['median']:.6f}` | {ic} | "
                    f"`{d['shapiro_p']:.4e}` | `{d['wilcoxon_p']:.4e}` | "
                    f"`{d['wilcoxon_p_holm']:.4e}` | **{d['significancia']}** |\n")
        f.write("\n\n*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.\n")
    print(f"  [md]   {md_path}")

    return {
        "friedman_stat":   stat_fried,
        "friedman_p":      p_fried,
        "tabla_resumen":   tabla_resumen,
        "n_runs":          n_runs,
    }
