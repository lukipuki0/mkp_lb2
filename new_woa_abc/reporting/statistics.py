"""Análisis estadístico pareado para las variantes de ``new_woa_abc``.

Replica el protocolo usado por los otros benchmarks del proyecto: estadística
descriptiva, Shapiro--Wilk, Wilcoxon pareado contra un control, corrección de
Holm, Mann--Whitney U, Friedman y ranking medio. Las parejas siempre se forman
por semilla; nunca por la posición accidental de una fila en un CSV.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import warnings
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from new_woa_abc.dtw.strategies import VARIANT_NAMES

from .io import write_csv, write_json


def adjust_pvalues_holm(pvalues: Sequence[float]) -> list[float]:
    """Aplica Holm step-down conservando el orden original."""

    adjusted = [float("nan")] * len(pvalues)
    valid = [
        (index, float(value))
        for index, value in enumerate(pvalues)
        if math.isfinite(float(value))
    ]
    ordered = sorted(valid, key=lambda item: item[1])
    running_max = 0.0
    total = len(ordered)
    for rank, (original_index, pvalue) in enumerate(ordered):
        corrected = min(1.0, (total - rank) * pvalue)
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max
    return adjusted


def _stars(pvalue: float, alpha: float) -> str:
    if not math.isfinite(pvalue) or pvalue >= alpha:
        return "ns"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    return "*"


def _shapiro(values: np.ndarray) -> tuple[float, float, str]:
    if values.size < 3:
        return float("nan"), float("nan"), "requiere al menos 3 valores"
    if np.allclose(values, values[0], rtol=0.0, atol=0.0):
        return 1.0, 1.0, "distribución degenerada"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.shapiro(values)
    return float(result.statistic), float(result.pvalue), ""


def _wilcoxon(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    if np.array_equal(reference, candidate):
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = stats.wilcoxon(
                candidate,
                reference,
                alternative="two-sided",
                zero_method="pratt",
                method="auto",
            )
        except ValueError:
            return 0.0, 1.0
    return float(result.statistic), float(result.pvalue)


def _rank_biserial(oriented_differences: np.ndarray) -> float:
    """Correlación biserial; positivo significa que la variante es mejor."""

    differences = oriented_differences[
        ~np.isclose(oriented_differences, 0.0, rtol=0.0, atol=1e-15)
    ]
    if differences.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    positive = float(np.sum(ranks[differences > 0]))
    negative = float(np.sum(ranks[differences < 0]))
    denominator = positive + negative
    return 0.0 if denominator == 0 else (positive - negative) / denominator


def _variant_order(names: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(str(name) for name in names))
    official = [name for name in VARIANT_NAMES if name in present]
    return official + [name for name in present if name not in official]


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "sí", "si"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"valor booleano inválido: {value!r}")


def _paired_values(
    rows: Sequence[dict[str, Any]],
    metric: str,
) -> tuple[list[str], list[int], dict[str, np.ndarray]]:
    by_variant: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        variant = str(row.get("variant", ""))
        if not variant:
            raise ValueError("cada fila estadística debe incluir 'variant'")
        try:
            seed = int(row["seed"])
            value = float(row[metric])
        except KeyError as exc:
            raise ValueError(f"falta la columna estadística {exc.args[0]!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"valor no finito para {variant}, seed={seed}: {value}")
        if seed in by_variant[variant]:
            raise ValueError(f"seed duplicada para {variant}: {seed}")
        by_variant[variant][seed] = value

    variants = _variant_order(by_variant)
    if len(variants) < 2:
        raise ValueError("se requieren al menos dos variantes para comparar")
    seed_sets = {variant: set(by_variant[variant]) for variant in variants}
    reference_seeds = seed_sets[variants[0]]
    for variant in variants[1:]:
        if seed_sets[variant] != reference_seeds:
            missing = sorted(reference_seeds - seed_sets[variant])
            extra = sorted(seed_sets[variant] - reference_seeds)
            raise ValueError(
                f"las semillas no están emparejadas para {variant}; "
                f"faltan={missing}, sobran={extra}"
            )
    seeds = sorted(reference_seeds)
    if len(seeds) < 2:
        raise ValueError("se requieren al menos dos semillas emparejadas")
    values = {
        variant: np.asarray([by_variant[variant][seed] for seed in seeds], dtype=float)
        for variant in variants
    }
    return variants, seeds, values


def _save_boxplot(
    directory: Path,
    table: Sequence[dict[str, Any]],
    values: dict[str, np.ndarray],
    title: str,
    metric_label: str,
    reference_variant: str,
) -> None:
    ordered = [str(row["variant"]) for row in table]
    data = [values[variant] for variant in ordered]
    fig, axis = plt.subplots(figsize=(max(10.0, 1.45 * len(ordered)), 6.0))
    plot = axis.boxplot(
        data,
        patch_artist=True,
        tick_labels=ordered,
        widths=0.55,
        medianprops={"color": "#c0392b", "linewidth": 2.0},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.55},
    )
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(ordered)))
    for patch, color, variant in zip(plot["boxes"], colors, ordered):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        if variant == reference_variant:
            patch.set_edgecolor("black")
            patch.set_linewidth(2.5)
    axis.set_title(f"Comparación estadística — {title}")
    axis.set_ylabel(metric_label)
    axis.tick_params(axis="x", rotation=28)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"boxplot_estadistico.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def save_paired_statistical_analysis(
    directory: Path,
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    metric_label: str,
    title: str,
    reference_variant: str = "M0_no_dtw",
    minimize: bool = True,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Analiza variantes con semillas pareadas y escribe todos los artefactos."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe estar entre 0 y 1")
    variants, seeds, values = _paired_values(rows, metric)
    if reference_variant not in values:
        raise ValueError(
            f"la variante de control {reference_variant!r} no está en los resultados"
        )
    directory.mkdir(parents=True, exist_ok=True)
    reference = values[reference_variant]
    matrix = np.asarray([values[variant] for variant in variants], dtype=float)
    success_by_variant: dict[str, list[bool]] | None = None
    if rows and all("optimum_reached" in row for row in rows):
        indexed_success: dict[str, dict[int, bool]] = defaultdict(dict)
        for row in rows:
            indexed_success[str(row["variant"])][int(row["seed"])] = _boolean(
                row["optimum_reached"]
            )
        success_by_variant = {
            variant: [indexed_success[variant][seed] for seed in seeds]
            for variant in variants
        }

    ranks = np.empty_like(matrix)
    for run_index in range(len(seeds)):
        column = matrix[:, run_index]
        ranks[:, run_index] = stats.rankdata(column if minimize else -column)
    mean_ranks = np.mean(ranks, axis=1)

    if len(variants) >= 3 and np.allclose(matrix, matrix[0], rtol=0.0, atol=0.0):
        friedman_statistic = 0.0
        friedman_pvalue = 1.0
    elif len(variants) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                friedman = stats.friedmanchisquare(
                    *(values[variant] for variant in variants)
                )
                friedman_statistic = float(friedman.statistic)
                friedman_pvalue = float(friedman.pvalue)
            except ValueError:
                friedman_statistic = float("nan")
                friedman_pvalue = 1.0
    else:
        friedman_statistic = float("nan")
        friedman_pvalue = float("nan")

    table: list[dict[str, Any]] = []
    comparison_indices: list[int] = []
    raw_pvalues: list[float] = []
    for variant_index, variant in enumerate(variants):
        candidate = values[variant]
        mean = float(np.mean(candidate))
        std = float(np.std(candidate, ddof=1))
        standard_error = std / math.sqrt(candidate.size)
        shapiro_statistic, shapiro_pvalue, shapiro_note = _shapiro(candidate)
        oriented = reference - candidate if minimize else candidate - reference
        diff_statistic, diff_pvalue, diff_note = _shapiro(oriented)
        if variant == reference_variant:
            wilcoxon_statistic, wilcoxon_pvalue = float("nan"), 1.0
            mannwhitney_statistic, mannwhitney_pvalue = float("nan"), 1.0
            effect_size = 0.0
            relation = "control"
        else:
            wilcoxon_statistic, wilcoxon_pvalue = _wilcoxon(reference, candidate)
            mannwhitney = stats.mannwhitneyu(
                candidate,
                reference,
                alternative="two-sided",
            )
            mannwhitney_statistic = float(mannwhitney.statistic)
            mannwhitney_pvalue = float(mannwhitney.pvalue)
            effect_size = _rank_biserial(oriented)
            relation = "pending"
            comparison_indices.append(len(table))
            raw_pvalues.append(wilcoxon_pvalue)

        table.append({
            "variant": variant,
            "runs": int(candidate.size),
            "mean_rank": float(mean_ranks[variant_index]),
            "mean": mean,
            "std": std,
            "median": float(np.median(candidate)),
            "iqr": float(stats.iqr(candidate)),
            "minimum": float(np.min(candidate)),
            "maximum": float(np.max(candidate)),
            "ci95_low": mean - 1.96 * standard_error,
            "ci95_high": mean + 1.96 * standard_error,
            "shapiro_statistic": shapiro_statistic,
            "shapiro_pvalue": shapiro_pvalue,
            "shapiro_note": shapiro_note,
            "paired_difference_shapiro_statistic": diff_statistic,
            "paired_difference_shapiro_pvalue": diff_pvalue,
            "paired_difference_shapiro_note": diff_note,
            "wilcoxon_statistic": wilcoxon_statistic,
            "wilcoxon_pvalue_raw": wilcoxon_pvalue,
            "wilcoxon_pvalue_holm": 1.0 if variant == reference_variant else float("nan"),
            "mannwhitney_statistic": mannwhitney_statistic,
            "mannwhitney_pvalue_raw": mannwhitney_pvalue,
            "rank_biserial": effect_size,
            "optimum_hits": (
                sum(success_by_variant[variant])
                if success_by_variant is not None
                else None
            ),
            "optimum_rate": (
                float(np.mean(success_by_variant[variant]))
                if success_by_variant is not None
                else None
            ),
            "relation_vs_control": relation,
            "significance": "Control" if variant == reference_variant else "Pendiente",
        })

    adjusted = adjust_pvalues_holm(raw_pvalues)
    reference_mean = float(np.mean(reference))
    for table_index, pvalue in zip(comparison_indices, adjusted):
        row = table[table_index]
        row["wilcoxon_pvalue_holm"] = pvalue
        candidate_better = (
            float(row["mean"]) < reference_mean
            if minimize
            else float(row["mean"]) > reference_mean
        )
        if not math.isfinite(pvalue) or pvalue >= alpha:
            relation = "similar"
            text = f"Similar a {reference_variant} (=) {_stars(pvalue, alpha)}"
        elif candidate_better:
            relation = "better"
            text = f"Mejor que {reference_variant} (+) {_stars(pvalue, alpha)}"
        else:
            relation = "worse"
            text = f"Peor que {reference_variant} (-) {_stars(pvalue, alpha)}"
        row["relation_vs_control"] = relation
        row["significance"] = text

    table.sort(key=lambda row: (float(row["mean_rank"]), str(row["variant"])))
    for rank, row in enumerate(table, start=1):
        row["rank"] = rank

    analysis: dict[str, Any] = {
        "title": title,
        "metric": metric,
        "metric_label": metric_label,
        "objective": "minimize" if minimize else "maximize",
        "reference_variant": reference_variant,
        "alpha": alpha,
        "paired_by": "seed",
        "seeds": seeds,
        "n_runs": len(seeds),
        "n_variants": len(variants),
        "friedman_statistic": friedman_statistic,
        "friedman_pvalue": friedman_pvalue,
        "holm_comparisons": len(raw_pvalues),
        "table": table,
    }
    write_csv(directory / "analisis_estadistico_pvalues.csv", table)
    write_json(directory / "analisis_estadistico_pvalues.json", analysis)

    friedman_text = (
        "no aplicable (requiere al menos tres variantes)"
        if not math.isfinite(friedman_pvalue)
        else f"χ²={friedman_statistic:.6g}, p={friedman_pvalue:.6e}"
    )
    lines = [
        f"# Análisis estadístico inferencial — {title}",
        "",
        f"- Corridas emparejadas por semilla: {len(seeds)}.",
        f"- Control: `{reference_variant}`.",
        f"- Métrica: {metric_label} ({'menor' if minimize else 'mayor'} es mejor).",
        f"- Wilcoxon bilateral pareado; Holm controla {len(raw_pvalues)} comparaciones contra el control.",
        "- Mann–Whitney U se informa para compatibilidad con los benchmarks anteriores, pero la inferencia principal es pareada.",
        f"- Friedman global: {friedman_text}.",
        "- IC 95 %: media ± 1.96·error estándar.",
        "",
        "| Rank | Variante | Mean rank | Media | Std | Mediana | IQR | IC 95 % | Shapiro p | Wilcoxon p | p Holm | RBC | Óptimos | Resultado |",
        "|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in table:
        interval = f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}]"
        success = (
            "N/D"
            if row["optimum_hits"] is None
            else f"{row['optimum_hits']}/{row['runs']} ({100.0 * row['optimum_rate']:.1f} %)"
        )
        lines.append(
            f"| {row['rank']} | `{row['variant']}` | {row['mean_rank']:.4f} | "
            f"{row['mean']:.8g} | {row['std']:.8g} | {row['median']:.8g} | "
            f"{row['iqr']:.8g} | {interval} | {row['shapiro_pvalue']:.4e} | "
            f"{row['wilcoxon_pvalue_raw']:.4e} | {row['wilcoxon_pvalue_holm']:.4e} | "
            f"{row['rank_biserial']:.4f} | {success} | {row['significance']} |"
        )
    lines.extend([
        "",
        "RBC es la correlación biserial de rangos: un valor positivo favorece a la variante frente al control.",
        "Significancia: `*** p<0.001`, `** p<0.01`, `* p<0.05`, `ns p≥0.05`, usando el p ajustado por Holm.",
    ])
    (directory / "analisis_estadistico_pvalues.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    _save_boxplot(
        directory,
        table,
        values,
        title,
        metric_label,
        reference_variant,
    )
    return analysis


def save_global_statistical_summary(
    directory: Path,
    analyses: Sequence[tuple[str, dict[str, Any]]],
    *,
    title: str,
) -> dict[str, Any]:
    """Agrega rankings y victorias sin mezclar objetivos crudos."""

    if not analyses:
        raise ValueError("no hay análisis por problema para resumir")
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    for problem, analysis in analyses:
        for row in analysis["table"]:
            entry = {"problem": problem, **row}
            details.append(entry)
            by_variant[str(row["variant"])].append(entry)

    summary: list[dict[str, Any]] = []
    for variant in _variant_order(by_variant):
        entries = by_variant[variant]
        relations = [str(entry["relation_vs_control"]) for entry in entries]
        summary.append({
            "variant": variant,
            "problems": len(entries),
            "average_mean_rank": float(np.mean([
                float(entry["mean_rank"]) for entry in entries
            ])),
            "rank_first_places": sum(float(entry["mean_rank"]) == 1.0 for entry in entries),
            "significant_wins_vs_control": relations.count("better"),
            "statistical_ties_vs_control": relations.count("similar"),
            "significant_losses_vs_control": relations.count("worse"),
            "control_rows": relations.count("control"),
            "optimum_hits": (
                sum(int(entry["optimum_hits"]) for entry in entries)
                if all(entry.get("optimum_hits") is not None for entry in entries)
                else None
            ),
            "optimum_trials": (
                sum(int(entry["runs"]) for entry in entries)
                if all(entry.get("optimum_hits") is not None for entry in entries)
                else None
            ),
        })
    summary.sort(key=lambda row: (float(row["average_mean_rank"]), str(row["variant"])))
    for rank, row in enumerate(summary, start=1):
        row["global_rank"] = rank

    result = {
        "title": title,
        "aggregation": "mean of within-problem paired mean ranks",
        "raw_objectives_combined": False,
        "problems": [problem for problem, _ in analyses],
        "summary": summary,
        "details": details,
    }
    write_csv(directory / "analisis_estadistico_global.csv", summary)
    write_csv(directory / "analisis_estadistico_por_problema.csv", details)
    write_json(directory / "analisis_estadistico_global.json", result)

    lines = [
        f"# Análisis estadístico global — {title}",
        "",
        "> No se mezclan valores objetivos crudos de problemas diferentes. El orden global es el promedio de los mean ranks calculados dentro de cada problema y semilla.",
        "",
        "| Rank | Variante | Problemas | Mean rank global | Primeros lugares | + vs control | = vs control | - vs control | Óptimos |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        optimum = (
            "N/D"
            if row["optimum_hits"] is None
            else f"{row['optimum_hits']}/{row['optimum_trials']}"
        )
        lines.append(
            f"| {row['global_rank']} | `{row['variant']}` | {row['problems']} | "
            f"{row['average_mean_rank']:.4f} | {row['rank_first_places']} | "
            f"{row['significant_wins_vs_control']} | {row['statistical_ties_vs_control']} | "
            f"{row['significant_losses_vs_control']} | {optimum} |"
        )
    lines.extend([
        "",
        "`+`, `=` y `-` cuentan resultados Wilcoxon significativos o similares después de Holm dentro de cada problema.",
    ])
    (directory / "analisis_estadistico_global.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    fig, axis = plt.subplots(figsize=(max(9.0, 1.3 * len(summary)), 5.5))
    labels = [str(row["variant"]) for row in summary]
    values = [float(row["average_mean_rank"]) for row in summary]
    bars = axis.bar(labels, values, color=plt.cm.tab10(np.linspace(0, 1, len(summary))))
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_title(f"Ranking global — {title}")
    axis.set_ylabel("Mean rank promedio (menor es mejor)")
    axis.tick_params(axis="x", rotation=28)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"ranking_estadistico_global.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    return result


__all__ = [
    "adjust_pvalues_holm",
    "save_global_statistical_summary",
    "save_paired_statistical_analysis",
]
