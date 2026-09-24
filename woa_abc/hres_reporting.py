"""Reportes descriptivos y gráficos del experimento WOA--ABC HRES2."""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from woa_abc.cec_reporting import save_epoch_plots
from woa_abc.plots.dtw import save_dtw_comparison_plot
from woa_abc.result_io import write_csv, write_json


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value)


def save_run_summary(
    run_dir: Path,
    row: dict[str, Any],
    decoded_solution: dict[str, Any],
    hres2_metrics: dict[str, Any],
) -> None:
    lines = [
        "RESULTADO WOA--ABC HRES2-H2/WPEB",
        f"Problema         : {row['problem']}",
        f"Variante         : {row['variant']}",
        f"Run              : {row['run']}",
        f"Semilla          : {row['seed']}",
        f"Mejor LCOE       : {_fmt(row['best_value'])}",
        f"LCOE (CNY/kWh)   : {_fmt(row['lcoe_cny_per_kwh'])}",
        f"LCOH (CNY/kg)    : {_fmt(row['lcoh_cny_per_kg'])}",
        f"AGSR             : {_fmt(row['agsr'])}",
        f"H2 total (kg)    : {_fmt(row['total_h2_kg'])}",
        f"Factible         : {row['feasible']}",
        f"Evaluaciones     : {row['objective_evaluations']}",
        f"Tiempo (s)       : {_fmt(row['time_seconds'])}",
        f"Alarmas DTW      : {row['dtw_alarm_count']}",
        f"Transiciones     : {row['mode_transition_count']}",
        f"Cambios parámetros: {row['parameter_update_count']}",
        "",
        "Solución decodificada:",
    ]
    lines.extend(f"  {key}: {_fmt(value)}" for key, value in decoded_solution.items())
    lines.extend(["", "Métricas HRES2:"])
    lines.extend(f"  {key}: {_fmt(value)}" for key, value in hres2_metrics.items())
    (run_dir / "resumen_resultado.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    output: list[dict[str, Any]] = []
    for variant, items in grouped.items():
        values = [float(item["best_value"]) for item in items]
        output.append({
            "variant": variant,
            "runs": len(values),
            "best": min(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "worst": max(values),
            "feasible_runs": sum(bool(item["feasible"]) for item in items),
            "feasibility_pct": 100.0 * sum(bool(item["feasible"]) for item in items) / len(items),
            "mean_time_seconds": statistics.mean(float(item["time_seconds"]) for item in items),
        })
    return output


def _write_reports(
    base_dir: Path,
    rows: list[dict[str, Any]],
    title: str,
    traces: dict[str, Any] | None = None,
) -> None:
    summary = _summary(rows)
    write_csv(base_dir / "resultados_variantes.csv", rows)
    write_json(base_dir / "resultados_variantes.json", rows)
    write_csv(base_dir / "resumen_variantes.csv", summary)

    txt = [title, "=" * len(title), ""]
    md = [
        f"# {title}",
        "",
        "| Variante | Runs | Mejor LCOE | Media | Mediana | Std | Factibles | Factibilidad | Tiempo medio (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        txt.append(
            f"{item['variant']}: runs={item['runs']}, mejor_lcoe={_fmt(item['best'])}, "
            f"media={_fmt(item['mean'])}, factibles={item['feasible_runs']}/{item['runs']}, "
            f"tiempo_medio_s={_fmt(item['mean_time_seconds'])}"
        )
        md.append(
            f"| {item['variant']} | {item['runs']} | {_fmt(item['best'])} | "
            f"{_fmt(item['mean'])} | {_fmt(item['median'])} | {_fmt(item['std'])} | "
            f"{item['feasible_runs']} | {_fmt(item['feasibility_pct'])}% | {_fmt(item['mean_time_seconds'])} |"
        )
    (base_dir / "resumen_variantes.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    (base_dir / "resumen_variantes.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    labels = [item["variant"].split("_", 1)[0] for item in summary]

    feasible = [float(item["feasibility_pct"]) for item in summary]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, feasible, color="#2E7D32", alpha=0.82)
    ax.set(title=f"Factibilidad — {title}", xlabel="Variante", ylabel="Corridas factibles (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(base_dir / "factibilidad_variantes.png", dpi=150)
    plt.close(fig)

    if max(item["runs"] for item in summary) > 1:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["variant"]].append(float(row["best_value"]))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot([grouped[item["variant"]] for item in summary], tick_labels=labels, patch_artist=True)
        ax.set(title=f"Distribución de LCOE — {title}", xlabel="Variante", ylabel="LCOE")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(base_dir / "boxplot_variantes.png", dpi=150)
        plt.close(fig)

    if traces:
        save_dtw_comparison_plot(
            base_dir,
            traces,
            rows,
            title="WOA-ABC — HRES2-H2/WPEB",
            optimum=None,
            scale="linear",
            stem="dtw_fire",
        )


def save_problem_report(
    problem_dir: Path,
    rows: list[dict[str, Any]],
    traces: dict[str, Any] | None = None,
) -> None:
    _write_reports(problem_dir, rows, f"HRES2-H2/WPEB — {rows[0]['problem']}", traces)


def save_global_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = _summary(rows)
    write_csv(output_dir / "resumen_global.csv", summary)
    txt = ["RESUMEN GLOBAL WOA--ABC HRES2-H2/WPEB", ""]
    md = [
        "# Resumen global WOA--ABC HRES2-H2/WPEB",
        "",
        "| Variante | Runs | Mejor LCOE | Media | Std | Factibles | Factibilidad |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        txt.append(
            f"{item['variant']} | runs={item['runs']} | mejor={_fmt(item['best'])} | "
            f"media={_fmt(item['mean'])} | factibles={item['feasible_runs']}/{item['runs']}"
        )
        md.append(
            f"| {item['variant']} | {item['runs']} | {_fmt(item['best'])} | {_fmt(item['mean'])} | "
            f"{_fmt(item['std'])} | {item['feasible_runs']} | {_fmt(item['feasibility_pct'])}% |"
        )
    txt.extend(["", "El análisis estadístico inferencial se realizará cuando se habiliten las 31 corridas."])
    md.extend(["", "El análisis estadístico inferencial se realizará cuando se habiliten las 31 corridas."])
    (output_dir / "resumen_global.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    (output_dir / "resumen_global.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    labels = [item["variant"].split("_", 1)[0] for item in summary]
    means = [float(item["mean"]) for item in summary]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, means, color="#1565C0", alpha=0.82)
    ax.set(title="Comparación global de variantes — HRES2-H2/WPEB", xlabel="Variante", ylabel="LCOE medio")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "comparacion_global.png", dpi=150)
    plt.close(fig)


__all__ = ["save_epoch_plots", "save_global_report", "save_problem_report", "save_run_summary"]
