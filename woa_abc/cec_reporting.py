"""Reportes descriptivos y gráficos del experimento WOA--ABC CEC2022."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from woa_abc.plots.dtw import save_dtw_comparison_plot, save_dtw_run_plot
from woa_abc.result_io import write_csv, write_json


_VARIANT_COLORS = {
    "M0": "#1f77b4",
    "M1": "#ff7f0e",
    "M2": "#2ca02c",
    "M4": "#d62728",
    "M5": "#9467bd",
    "M6": "#8c564b",
    "M8": "#e377c2",
}


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.12g}" if math.isfinite(number) else "N/A"


def save_epoch_plots(
    epoch_dir: Path,
    epoch: Any,
    optimum: float | None,
    variant: str,
    *,
    scale: str = "linear",
) -> None:
    """Guarda convergencia, señal DTW y evolución de parámetros."""

    epoch_dir.mkdir(parents=True, exist_ok=True)
    iterations = np.arange(1, len(epoch.historial) + 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(iterations, epoch.historial, label="Mejor global", linewidth=2.4)
    ax.plot(iterations, epoch.historial_woa, label="WOA", linewidth=1.2, alpha=0.8)
    ax.plot(iterations, epoch.historial_abc, label="ABC", linewidth=1.2, alpha=0.8)
    if optimum is not None:
        ax.axhline(optimum, color="black", linestyle="--", linewidth=1.2, label=f"Óptimo: {_fmt(optimum)}")
    if scale == "log" and all(value > 0 for value in epoch.historial):
        ax.set_yscale("log")
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(formatter)
    if epoch.historial:
        ax.annotate(
            f"final: {_fmt(epoch.historial[-1])}",
            xy=(len(epoch.historial), epoch.historial[-1]),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.85},
        )
    ax.set(title=f"Convergencia — {variant}", xlabel="Iteración", ylabel="Valor objetivo")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(epoch_dir / "fitness_convergence.png", dpi=150)
    plt.close(fig)

    parameters = epoch.parametros_historial
    names = (
        ("woa_a", "WOA a"),
        ("abc_phi_scale", "ABC phi scale"),
        ("abc_guide_strength", "ABC guide strength"),
        ("abc_limit", "ABC limit"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (key, label) in zip(axes.flat, names):
        values = [float(item[key]) for item in parameters]
        ax.plot(iterations, values, linewidth=1.8)
        ax.set_title(label)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Iteración")
    axes[1, 1].set_xlabel("Iteración")
    fig.suptitle(f"Parámetros efectivos — {variant}")
    fig.tight_layout()
    fig.savefig(epoch_dir / "parametros_adaptativos.png", dpi=150)
    plt.close(fig)

    # Gráfico DTW de dos paneles con el mismo estilo del ZIP de referencia.
    # El gráfico DTW del ZIP usa escala lineal; la convergencia individual
    # continúa guardándose aparte con la escala solicitada por el dominio.
    save_dtw_run_plot(epoch_dir, epoch, optimum, variant, scale="linear")


def save_run_summary(run_dir: Path, row: dict[str, Any]) -> None:
    lines = [
        "RESULTADO WOA--ABC CEC2022",
        f"Función          : F{row['function_id']:02d} ({row['problem']})",
        f"Dimensión        : {row['dimension']}",
        f"Variante         : {row['variant']}",
        f"Run              : {row['run']}",
        f"Semilla          : {row['seed']}",
        f"Mejor valor      : {_fmt(row['best_value'])}",
        f"Óptimo conocido  : {_fmt(row['optimum_reference'])}",
        f"Error            : {_fmt(row['error'])}",
        f"Gap (%)          : {_fmt(row['gap_pct'])}",
        f"Evaluaciones     : {row['objective_evaluations']}",
        f"Tiempo (s)       : {_fmt(row['time_seconds'])}",
        f"Alarmas DTW      : {row['dtw_alarm_count']}",
        f"Transiciones     : {row['mode_transition_count']}",
        f"Cambios parámetros: {row['parameter_update_count']}",
    ]
    (run_dir / "resumen_resultado.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _explore_mask(epoch: Any) -> list[bool]:
    return ["explore" in str(item.get("mode", "")) for item in epoch.parametros_historial]


def _fire_iterations(epoch: Any) -> set[int]:
    fired: set[int] = set()
    for item in epoch.dtw_info_hist:
        if item.get("fire_aceptado") or item.get("fire"):
            fired.add(int(item.get("iteracion", -1)))
    for item in epoch.eventos_adaptacion:
        if item.get("fire_aceptado"):
            fired.add(int(item.get("iteracion", -1)))
    return {iteration for iteration in fired if iteration >= 0}


def save_style_comparison_plot(
    output_dir: Path,
    traces: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    title: str,
    optimum: float | None,
    scale: str = "linear",
) -> None:
    """Gráfico de comparación con el estilo de DTW_optimization.

    Panel superior: convergencia con segmentos sólidos/punteados para
    explotación/exploración y marcadores cuando DTW acepta una adaptación.
    Panel inferior: delta y umbral theta_delta por variante.
    """

    if not traces:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )
    row_by_variant = {row["variant"]: row for row in rows}

    for variant, epoch in traces.items():
        short = variant.split("_", 1)[0]
        color = _VARIANT_COLORS.get(short, "#666666")
        values = list(epoch.historial)
        if not values:
            continue
        mask = _explore_mask(epoch)
        if len(mask) != len(values):
            mask = [False] * len(values)
        segment_start = 0
        previous = mask[0]
        for index in range(1, len(values)):
            if mask[index] != previous:
                ax1.plot(
                    range(segment_start, index + 1),
                    values[segment_start:index + 1],
                    color=color,
                    linewidth=1.5,
                    linestyle="--" if previous else "-",
                    zorder=3,
                )
                segment_start = index
                previous = mask[index]
        ax1.plot(
            range(segment_start, len(values)),
            values[segment_start:],
            color=color,
            linewidth=1.5,
            linestyle="--" if previous else "-",
            zorder=3,
        )
        row = row_by_variant.get(variant, {})
        ax1.plot(
            [],
            [],
            color=color,
            linewidth=1.5,
            label=f"{short} (mejor={_fmt(row.get('best_value', values[-1]))})",
        )
        fires = sorted(_fire_iterations(epoch))
        if fires:
            valid = [iteration for iteration in fires if iteration < len(values)]
            ax1.scatter(
                valid,
                [values[iteration] for iteration in valid],
                color=color,
                marker="v",
                s=30,
                zorder=5,
                edgecolors="white",
                linewidths=0.3,
            )

    if optimum is not None:
        ax1.axhline(
            y=optimum,
            color="#2E7D32",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Óptimo ({_fmt(optimum)})",
        )
    ax1.plot([], [], color="gray", linewidth=1.5, linestyle="-", label="Explotación")
    ax1.plot([], [], color="gray", linewidth=1.5, linestyle="--", label="Exploración")
    if scale == "log" and all(value > 0 for epoch in traces.values() for value in epoch.historial):
        ax1.set_yscale("log")
    ax1.set_ylabel("Valor objetivo")
    ax1.set_title(title)
    ax1.legend(loc="lower right", framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.2, linestyle=":")

    for variant, epoch in traces.items():
        short = variant.split("_", 1)[0]
        color = _VARIANT_COLORS.get(short, "#666666")
        ready_iterations: list[int] = []
        deltas: list[float] = []
        thresholds: list[float] = []
        for item in epoch.dtw_info_hist:
            if item.get("ready") and item.get("delta") is not None:
                ready_iterations.append(int(item.get("iteracion", len(ready_iterations))))
                deltas.append(float(item["delta"]))
                thresholds.append(float(item.get("theta_delta", 0.0)))
        if ready_iterations:
            ax2.plot(ready_iterations, deltas, color=color, linewidth=1.3, label=rf"$\Delta$ {short}")
            ax2.plot(ready_iterations, thresholds, color=color, linewidth=0.8, linestyle=":", alpha=0.6, label=rf"$\theta_\Delta$ {short}")
            fires = _fire_iterations(epoch)
            valid = [iteration for iteration in fires if iteration in ready_iterations]
            ax2.scatter(
                valid,
                [deltas[ready_iterations.index(iteration)] for iteration in valid],
                color=color,
                marker="v",
                s=35,
                zorder=5,
                edgecolors="white",
                linewidths=0.4,
            )
    ax2.axhline(y=0, color="#95a5a6", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel(r"$\Delta$ (D1 − D2)")
    ax2.set_title("Señal de estancamiento DTW — Δ > 0 indica estancamiento")
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(handles, labels, loc="upper left", framealpha=0.9, fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.2, linestyle=":")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(output_dir / "comparacion_variantes.png")
    fig.savefig(output_dir / "comparacion_variantes.pdf")
    plt.close(fig)


def _descriptive_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    summary: list[dict[str, Any]] = []
    for variant, items in grouped.items():
        values = [float(item["best_value"]) for item in items]
        summary.append({
            "variant": variant,
            "runs": len(values),
            "best": min(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "worst": max(values),
            "optimum_reference": items[0]["optimum_reference"],
            "mean_error": statistics.mean(float(item["error"]) for item in items),
            "mean_time_seconds": statistics.mean(float(item["time_seconds"]) for item in items),
        })
    return summary


def save_function_report(
    function_dir: Path,
    rows: list[dict[str, Any]],
    traces: dict[str, Any] | None = None,
) -> None:
    """Consolida todas las variantes de una función CEC."""

    summary = _descriptive_rows(rows)
    write_csv(function_dir / "resultados_variantes.csv", rows)
    write_json(function_dir / "resultados_variantes.json", rows)
    write_csv(function_dir / "resumen_variantes.csv", summary)

    title = f"CEC F{rows[0]['function_id']:02d} — {rows[0]['problem']} — D={rows[0]['dimension']}"
    text_lines = [title, "=" * len(title), ""]
    md_lines = [f"# {title}", "", "| Variante | Runs | Mejor | Media | Mediana | Std | Error medio | Tiempo medio (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summary:
        text_lines.append(
            f"{item['variant']}: runs={item['runs']}, mejor={_fmt(item['best'])}, "
            f"media={_fmt(item['mean'])}, error_medio={_fmt(item['mean_error'])}, "
            f"tiempo_medio_s={_fmt(item['mean_time_seconds'])}"
        )
        md_lines.append(
            f"| {item['variant']} | {item['runs']} | {_fmt(item['best'])} | "
            f"{_fmt(item['mean'])} | {_fmt(item['median'])} | {_fmt(item['std'])} | "
            f"{_fmt(item['mean_error'])} | {_fmt(item['mean_time_seconds'])} |"
        )
    (function_dir / "resumen_variantes.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (function_dir / "resumen_variantes.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    labels = [item["variant"].split("_", 1)[0] for item in summary]
    optimum = float(summary[0]["optimum_reference"])

    if max(item["runs"] for item in summary) > 1:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["variant"]].append(float(row["best_value"]))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot([grouped[item["variant"]] for item in summary], tick_labels=labels, patch_artist=True)
        ax.axhline(optimum, color="#2E7D32", linestyle="--", linewidth=1.2)
        ax.set(title=f"Distribución por variante — F{rows[0]['function_id']:02d}", xlabel="Variante", ylabel="Valor objetivo")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(function_dir / "boxplot_variantes.png", dpi=150)
        plt.close(fig)

    if traces:
        save_dtw_comparison_plot(
            function_dir,
            traces,
            rows,
            title=f"WOA-ABC — CEC F{rows[0]['function_id']:02d} (D={rows[0]['dimension']})",
            optimum=float(rows[0]["optimum_reference"]),
            scale="linear",
            stem="dtw_fire",
        )


def save_global_report(output_dir: Path, rows: list[dict[str, Any]], dimension: int) -> None:
    """Genera el resumen de las 12 funciones del experimento."""

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["function_id"]), str(row["variant"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (function_id, variant), items in grouped.items():
        values = [float(item["best_value"]) for item in items]
        errors = [float(item["error"]) for item in items]
        summary.append({
            "function_id": function_id,
            "problem": items[0]["problem"],
            "dimension": dimension,
            "variant": variant,
            "runs": len(items),
            "best": min(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "mean_error": statistics.mean(errors),
            "optimum_reference": items[0]["optimum_reference"],
            "mean_time_seconds": statistics.mean(float(item["time_seconds"]) for item in items),
        })
    summary.sort(key=lambda item: (item["function_id"], item["variant"]))
    write_csv(output_dir / "resumen_global.csv", summary)

    txt = [f"RESUMEN GLOBAL WOA--ABC CEC2022 — D={dimension}", ""]
    md = [f"# Resumen global WOA--ABC CEC2022 — D={dimension}", "", "| Función | Variante | Runs | Mejor | Media | Std | Error medio |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in summary:
        txt.append(
            f"F{item['function_id']:02d} | {item['variant']} | runs={item['runs']} | "
            f"mejor={_fmt(item['best'])} | media={_fmt(item['mean'])} | error={_fmt(item['mean_error'])}"
        )
        md.append(
            f"| F{item['function_id']:02d} | {item['variant']} | {item['runs']} | "
            f"{_fmt(item['best'])} | {_fmt(item['mean'])} | {_fmt(item['std'])} | {_fmt(item['mean_error'])} |"
        )
    txt.extend(["", "El análisis estadístico inferencial se realizará cuando se habiliten las 31 corridas."])
    md.extend(["", "El análisis estadístico inferencial se realizará cuando se habiliten las 31 corridas."])
    (output_dir / "resumen_global.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    (output_dir / "resumen_global.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    variants = list(dict.fromkeys(row["variant"] for row in rows))
    fig, ax = plt.subplots(figsize=(14, 7))
    for variant in variants:
        items = [item for item in summary if item["variant"] == variant]
        xs = [item["function_id"] for item in items]
        ys = [item["mean_error"] for item in items]
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=variant.split("_", 1)[0])
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xticks(range(1, 13))
    ax.set(title=f"Error respecto al óptimo por función — CEC2022 D={dimension}", xlabel="Función CEC2022", ylabel="Error medio (escala symlog)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(output_dir / "comparacion_global.png", dpi=150)
    plt.close(fig)


# Compatibilidad para código que importaba el nombre anterior. La implementación
# activa vive en woa_abc/plots/dtw.py y mantiene el formato del ZIP.
save_style_comparison_plot = save_dtw_comparison_plot


__all__ = [
    "save_epoch_plots",
    "save_style_comparison_plot",
    "save_function_report",
    "save_global_report",
    "save_run_summary",
]
