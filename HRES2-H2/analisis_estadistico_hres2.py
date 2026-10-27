"""
HRES2-H2/analisis_estadistico_hres2.py
--------------------------------------
Módulo de Análisis Estadístico Inferencial y Pruebas de Hipótesis para HRES2-H2.
La comparación promovida es Mann-Whitney U bilateral con Holm y W/T/L
(menor LCOE físico es mejor y hace falta p ajustada). Wilcoxon y Friedman
no se calculan: no hay pareo justificado.
"""

from __future__ import annotations

import os
import csv
import math
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_ALPHA = 0.05
_PAIRED_NOT_VALID = (
    "Paired Wilcoxon and Friedman are not valid without proof of paired initial conditions "
    "and are not used for inference."
)
_NO_PHYSICAL = (
    "skipped incompatible LCOE inference: no finite feasible physical LCOE observations; "
    "records and counts were preserved; values were not imputed or dropped"
)


def _holm_adjust(p_values):
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (float(p_values[index]), index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def _finite_numbers(values):
    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return finite


def _inference_gate(n_records, n_finite, expected_n):
    """Infer only when every expected observation is finite and physically feasible."""
    expected = n_records if expected_n is None else int(expected_n)
    if n_records != expected:
        return expected, (
            "skipped incomplete sample: missing runs "
            f"(n_records={n_records}, expected={expected}); "
            "comparison not computed; no p-value or W/T/L; values were not imputed"
        )
    if n_finite != expected:
        detail = (
            "no finite feasible physical LCOE"
            if n_finite == 0
            else "not every expected record is finite physical feasible"
        )
        return expected, (
            f"skipped incomplete sample: {detail}; "
            "comparison not computed; no p-value or W/T/L; values were not imputed"
        )
    return expected, ""


def realizar_analisis_estadistico_completo(
    resultados_dict: dict[str, list[float]],
    output_dir     : str,
    algoritmo_referencia: str = "Hybrid DTW",
    records: list[dict] | None = None,
    expected_n: int | None = None,
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
    groups = None
    if records is not None:
        groups = {}
        for record in records:
            groups.setdefault(str(record.get("method")), []).append(record)
    nombres_algs = list(dict.fromkeys([*resultados_dict.keys(), *(groups or {})]))
    if algoritmo_referencia not in nombres_algs:
        raise KeyError(algoritmo_referencia)
    samples = {}
    for alg in nombres_algs:
        if groups is None:
            raw = list(resultados_dict.get(alg, []))
            finite = _finite_numbers(raw)
            samples[alg] = {
                "n_records": len(raw), "n_finite_feasible": len(finite), "sample": finite, "reason": "",
            }
            continue
        group = groups.get(alg, [])
        finite = []
        for record in group:
            if record.get("feasible") is not True:
                continue
            try:
                number = float(record.get("physical_lcoe"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                finite.append(number)
        samples[alg] = {
            "n_records": len(group), "n_finite_feasible": len(finite), "sample": finite, "reason": "",
        }
    for info in samples.values():
        expected, gate = _inference_gate(info["n_records"], info["n_finite_feasible"], expected_n)
        info["expected_n"] = expected
        info["inference_ready"] = gate == ""
        if gate:
            info["reason"] = gate
    ref_ready = samples[algoritmo_referencia]["inference_ready"]
    ref_vals = samples[algoritmo_referencia]["sample"]
    n_runs = samples[algoritmo_referencia]["n_records"]

    print("\n" + "=" * 85)
    print(f"  ANÁLISIS ESTADÍSTICO Y INFERENCIAL DE RESULTADOS ({n_runs} RUNS)")
    print(f"  Algoritmo de Referencia (Control): {algoritmo_referencia}")
    print("  Comparación promovida: Mann-Whitney U bilateral + Holm. Menor LCOE es mejor.")
    print("  " + _PAIRED_NOT_VALID)
    print("=" * 85)

    # ── 1. Pruebas Descriptivas y de Normalidad (Shapiro-Wilk) ───────────────
    tabla_resumen = []

    for alg in nombres_algs:
        info = samples[alg]
        vals = np.array(info["sample"], dtype=float)
        if vals.size == 0:
            if alg == algoritmo_referencia and info["inference_ready"]:
                empty_label, empty_status = "reference", "reference"
                empty_reason = info["reason"]
            else:
                empty_label = "skipped"
                empty_status = "reference_incomplete" if alg == algoritmo_referencia else "skipped"
                empty_reason = info["reason"] or _NO_PHYSICAL
                if alg != algoritmo_referencia and not ref_ready:
                    empty_reason = (
                        "skipped incomplete reference: all comparisons skipped; no p-value or W/T/L; "
                        "values were not imputed; " + samples[algoritmo_referencia]["reason"]
                    )
                    if info["reason"] and info["reason"] not in empty_reason:
                        empty_reason += "; " + info["reason"]
            tabla_resumen.append({
                "algoritmo": alg,
                "mean": float("nan"), "std": float("nan"), "median": float("nan"), "best": None,
                "iqr": float("nan"), "min": float("nan"), "max": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"), "shapiro_p": float("nan"),
                "wilcoxon_p": float("nan"), "mwu_p": float("nan"), "holm_p": None, "mean_rank": None,
                "significancia": empty_label,
                "wtl": empty_label,
                "n_records": info["n_records"], "n_finite_feasible": 0,
                "inference_reason": empty_reason,
                "inference_status": empty_status,
            })
            continue

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
        if len(vals) >= 3 and len(np.unique(vals)) > 1:
            stat_sw, p_sw = stats.shapiro(vals)
        elif len(vals) >= 3:
            stat_sw, p_sw = 1.0, 1.0
        else:
            stat_sw, p_sw = np.nan, np.nan

        # Mann-Whitney U bilateral. Wilcoxon no se usa: el pareo no está justificado.
        p_wilc = np.nan
        if alg == algoritmo_referencia:
            stat_mwu, p_mwu = np.nan, np.nan
            sig_label = "reference" if info["inference_ready"] else "skipped"
            status = "reference" if info["inference_ready"] else "reference_incomplete"
        elif not ref_ready or not info["inference_ready"]:
            stat_mwu, p_mwu = np.nan, np.nan
            sig_label = "skipped"
            status = "skipped"
            if not ref_ready:
                own = info["reason"]
                info["reason"] = (
                    "skipped incomplete reference: all comparisons skipped; no p-value or W/T/L; "
                    "values were not imputed; " + samples[algoritmo_referencia]["reason"]
                )
                if own and own not in info["reason"]:
                    info["reason"] += "; " + own
        else:
            try:
                stat_mwu, p_mwu = stats.mannwhitneyu(ref_vals, vals, alternative="two-sided")
                sig_label = "pending"
                status = "pending"
            except Exception as exc:
                stat_mwu, p_mwu = np.nan, np.nan
                sig_label = "skipped"
                status = "skipped"
                info["reason"] = (
                    "skipped incompatible LCOE inference: " + str(exc) +
                    "; comparison not computed; no p-value or W/T/L; values were not imputed"
                )

        tabla_resumen.append({
            "algoritmo": alg,
            "mean": mean_v,
            "std": std_v,
            "median": median_v,
            "best": min_v,
            "iqr": iqr_v,
            "min": min_v,
            "max": max_v,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "shapiro_p": p_sw,
            "wilcoxon_p": p_wilc,
            "mwu_p": p_mwu,
            "holm_p": None,
            "mean_rank": None,
            "significancia": sig_label,
            "wtl": "T" if sig_label == "pending" else sig_label,
            "n_records": info["n_records"],
            "n_finite_feasible": info["n_finite_feasible"],
            "inference_reason": info["reason"],
            "inference_status": status,
        })

    # Holm sobre las comparaciones de esta variante. W/T/L exige p ajustada, no dirección sola.
    pending = [d for d in tabla_resumen if d["inference_status"] == "pending"]
    if pending:
        adjusted = _holm_adjust([float(d["mwu_p"]) for d in pending])
        ref_mean = float(np.mean(ref_vals))
        for d, holm_p in zip(pending, adjusted):
            d["holm_p"] = float(holm_p)
            if holm_p < _ALPHA and d["mean"] > ref_mean:
                label = "W"
            elif holm_p < _ALPHA and d["mean"] < ref_mean:
                label = "L"
            else:
                label = "T"
            d["wtl"] = d["significancia"] = label
            d["inference_status"] = "mannwhitney_holm"
    wtl_counts = {"W": 0, "T": 0, "L": 0}
    for d in tabla_resumen:
        if d["wtl"] in wtl_counts:
            wtl_counts[d["wtl"]] += 1
    stat_fried, p_fried = float("nan"), float("nan")

    # ── 3. Impresión por Consola ──────────────────────────────────────────────
    print("\n--- MANN-WHITNEY U + HOLM (W/T/L de la referencia; menor LCOE es mejor) ---")
    print("  " + _PAIRED_NOT_VALID)
    header = f"{'Rank':<5} {'Algoritmo':<15} {'Mean LCOE':>12} {'Std':>10} {'Median':>12} {'Best':>12} {'Holm p':>12} {'WTL':<10}"
    print(header)
    print("-" * len(header))
    for r_idx, d in enumerate(tabla_resumen, 1):
        best_s = "null" if d["best"] is None else f"{d['best']:.6f}"
        holm_s = "" if d["holm_p"] is None else f"{d['holm_p']:.4e}"
        print(f"{r_idx:<5d} {d['algoritmo']:<15s} {d['mean']:>12.6f} {d['std']:>10.6f} {d['median']:>12.6f} {best_s:>12} {holm_s:>12} {d['wtl']:<10s}")

    # ── 4. Generar Gráfico de Boxplot Comparativo Multi-Algoritmo ────────────
    algs_sorted = [d["algoritmo"] for d in tabla_resumen if samples[d["algoritmo"]]["sample"]]
    vals_sorted = [samples[alg]["sample"] for alg in algs_sorted]
    boxplot_comp_path = os.path.join(output_dir, "mhs_comparative_boxplot.png")
    if algs_sorted:
        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(vals_sorted, patch_artist=True, tick_labels=algs_sorted, widths=0.5)
        colors = plt.cm.tab10(np.linspace(0, 1, len(algs_sorted)))
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        plt.title("Finite feasible physical LCOE (Mann-Whitney + Holm)", fontsize=13, fontweight="bold")
        plt.ylabel("LCOE (CNY/kWh)", fontsize=11)
        plt.xticks(rotation=25, ha="right", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(boxplot_comp_path, dpi=300)
        plt.close()
        print(f"\n  [plot] {boxplot_comp_path}")

    # ── 5. Guardar CSV Estadístico ───────────────────────────────────────────
    csv_stat_path = os.path.join(output_dir, "analisis_estadistico_pvalues.csv")
    with open(csv_stat_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "algoritmo", "mean_rank", "mean_lcoe", "std_lcoe",
            "median_lcoe", "best_lcoe", "iqr_lcoe", "ci95_low", "ci95_high",
            "shapiro_pvalue", "wilcoxon_pvalue", "mannwhitney_pvalue", "holm_pvalue",
            "n_records", "n_finite_feasible", "wtl", "inference_reason", "significancia"
        ])
        for r_idx, d in enumerate(tabla_resumen, 1):
            writer.writerow([
                r_idx, d["algoritmo"], "" if d["mean_rank"] is None else f"{d['mean_rank']:.2f}",
                f"{d['mean']:.6f}", f"{d['std']:.6f}", f"{d['median']:.6f}",
                "" if d["best"] is None else f"{d['best']:.6f}", f"{d['iqr']:.6f}",
                f"{d['ci95_low']:.6f}", f"{d['ci95_high']:.6f}",
                f"{d['shapiro_p']:.6e}", f"{d['wilcoxon_p']:.6e}", f"{d['mwu_p']:.6e}",
                "" if d["holm_p"] is None else f"{d['holm_p']:.6e}",
                d["n_records"], d["n_finite_feasible"], d["wtl"], d["inference_reason"],
                d["significancia"]
            ])
    print(f"  [csv]  {csv_stat_path}")

    # ── 6. Guardar Reporte Markdown ──────────────────────────────────────────
    md_stat_path = os.path.join(output_dir, "analisis_estadistico_pvalues.md")
    with open(md_stat_path, "w", encoding="utf-8") as f:
        f.write("# Inferencia HRES2-H2\n\n")
        f.write("Promoted comparison: independent-sample Mann-Whitney U (two-sided) with Holm ")
        f.write("across comparisons against this reference variant. W/T/L requires lower physical ")
        f.write(f"LCOE and adjusted p < {_ALPHA:.2f}. Every expected record of the reference and ")
        f.write("of that comparator must be finite physical feasible. An incomplete reference skips ")
        f.write("every comparison. A feasible subset is descriptive only: no p-value, W/T/L, imputation, ")
        f.write("or feasible-only superiority is produced. Records stay in the counts.\n\n")
        f.write(_PAIRED_NOT_VALID + "\n")
        f.write("Paired diagnostics are not valid.\n\n")
        f.write(f"- **Número de registros de la referencia:** {n_runs}\n")
        f.write(f"- **Algoritmo de Control (Referencia):** `{algoritmo_referencia}`\n\n")
        f.write("| Algoritmo | best (min) | median | n_records | n_finite_feasible | Holm p | WTL |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for d in tabla_resumen:
            best_s = "null" if d["best"] is None else f"{d['best']:.6f}"
            median_s = "null" if not math.isfinite(d["median"]) else f"{d['median']:.6f}"
            holm_s = "null" if d["holm_p"] is None else f"{d['holm_p']:.6e}"
            f.write(
                f"| {d['algoritmo']} | {best_s} | {median_s} | {d['n_records']} | "
                f"{d['n_finite_feasible']} | {holm_s} | {d['wtl']} |\n"
            )
            if d["inference_reason"]:
                f.write(f"\n{d['algoritmo']}: {d['inference_reason']}\n\n")
    print(f"  [md]   {md_stat_path}")

    return {
        "friedman_stat": stat_fried,
        "friedman_p": p_fried,
        "tabla_resumen": tabla_resumen,
        "paired_inference_valid": False,
        "paired_inference_reason": _PAIRED_NOT_VALID,
        "wtl_counts": wtl_counts,
    }
