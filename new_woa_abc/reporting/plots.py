"""Gráficos con la misma lectura visual del repositorio DTW de referencia."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from new_woa_abc.core.woa_abc import OptimizationResult


COLORS = {
    "fitness": "#1a5276",
    "woa": "#2471a3",
    "abc": "#17a589",
    "explore": "#e74c3c",
    "delta": "#6c3483",
    "threshold": "#e67e22",
    "intensity": "#c0392b",
    "optimum": "#7f8c8d",
    "explore_bg": "#e74c3c",
    "fire": "#f39c12",
    "d2": "#c0392b",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _is_explore(mode: str) -> bool:
    return str(mode).startswith("explore")


def _explore_ranges(modes: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, mode in enumerate(modes):
        if _is_explore(mode) and start is None:
            start = index
        elif not _is_explore(mode) and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, max(start, len(modes) - 1)))
    return ranges


def _shade_exploration(ax: plt.Axes, modes: list[str]) -> None:
    first = True
    for start, stop in _explore_ranges(modes):
        ax.axvspan(
            start,
            stop,
            color=COLORS["explore_bg"],
            alpha=0.08,
            label="Explore" if first else None,
            zorder=1,
        )
        first = False


def _plot_mode_segments(ax: plt.Axes, values: list[float], modes: list[str]) -> None:
    if not values:
        return
    segment_start = 0
    previous = _is_explore(modes[0])
    for index in range(1, len(values)):
        current = _is_explore(modes[index])
        if current != previous:
            ax.plot(
                range(segment_start, index + 1),
                values[segment_start : index + 1],
                color=COLORS["fitness"],
                linestyle="--" if previous else "-",
                linewidth=1.7,
            )
            segment_start = index
            previous = current
    ax.plot(
        range(segment_start, len(values)),
        values[segment_start:],
        color=COLORS["fitness"],
        linestyle="--" if previous else "-",
        linewidth=1.7,
    )


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}")
    plt.close(fig)


def save_run_plots(
    run_dir: Path,
    result: OptimizationResult,
    optimum: float | None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()
    iterations = np.arange(len(result.convergence_history))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(iterations, result.woa_history, color=COLORS["woa"], alpha=0.55, label="Después de WOA")
    ax.plot(iterations, result.abc_history, color=COLORS["abc"], alpha=0.55, label="Después de ABC")
    _plot_mode_segments(ax, result.convergence_history, result.mode_history)
    ax.plot([], [], color=COLORS["fitness"], linestyle="-", label="Global exploit")
    ax.plot([], [], color=COLORS["fitness"], linestyle="--", label="Global explore")
    if optimum is not None:
        ax.axhline(optimum, color=COLORS["optimum"], linestyle=":", label=f"Óptimo ({optimum:.6g})")
    if result.convergence_history and all(value > 0 for value in result.convergence_history):
        ax.set_yscale("log")
    ax.set(title=f"Convergencia — {result.variant}", xlabel="Iteración", ylabel="Costo")
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, run_dir, "convergence")

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )
    ax_top.plot(
        iterations,
        result.convergence_history,
        color=COLORS["fitness"],
        linewidth=1.6,
        label=f"Mejor costo ({result.best_cost:.6g})",
        zorder=3,
    )
    _shade_exploration(ax_top, result.mode_history)
    fire_iterations = [
        int(status["iteration"])
        for status in result.dtw_history
        if status.get("raw_fire")
    ]
    if fire_iterations:
        fire_values = [result.convergence_history[index] for index in fire_iterations]
        ax_top.scatter(
            fire_iterations,
            fire_values,
            marker="v",
            s=30,
            color=COLORS["fire"],
            label="Fire",
            zorder=4,
        )
    if optimum is not None:
        ax_top.axhline(optimum, color=COLORS["optimum"], linestyle=":", label="Óptimo")
    ax_top.set_ylabel("Costo")
    ax_top.set_title(f"WOA--ABC — {result.variant}")
    ax_top.grid(alpha=0.2, linestyle=":")
    ax_top.legend(loc="upper right")

    ready_iterations: list[int] = []
    signals: list[float] = []
    thresholds: list[float] = []
    intensities: list[float] = []
    for index, status in enumerate(result.dtw_history):
        if status.get("ready"):
            ready_iterations.append(index)
            if result.variant in {"M1_fire_d2", "M4_d2_continuous"}:
                signals.append(float(status["D2_vs_const"]))
                thresholds.append(float(status["theta_c"]))
            else:
                signals.append(float(status["delta"]))
                thresholds.append(float(status["theta_delta"]))
            value = result.intensity_history[index]
            intensities.append(0.0 if value is None else float(value))
    intensity_axis = ax_bottom.twinx()
    d2_view = result.variant in {"M1_fire_d2", "M4_d2_continuous"}
    if ready_iterations:
        signal_label = "D2 (distancia a constante)" if d2_view else r"$\Delta=D1-D2$"
        threshold_label = r"$\theta_c$" if d2_view else r"$\theta_\Delta$"
        signal_color = COLORS["d2"] if d2_view else COLORS["delta"]
        ax_bottom.plot(ready_iterations, signals, color=signal_color, label=signal_label)
        ax_bottom.plot(
            ready_iterations,
            thresholds,
            color=COLORS["threshold"],
            linestyle="--",
            label=threshold_label,
        )
        if d2_view:
            ax_bottom.fill_between(
                ready_iterations,
                signals,
                thresholds,
                where=np.asarray(signals) <= np.asarray(thresholds),
                color=COLORS["d2"],
                alpha=0.06,
                label=r"D2 $\leq \theta_c$",
            )
        else:
            ax_bottom.fill_between(
                ready_iterations,
                signals,
                0.0,
                where=np.asarray(signals) > 0.0,
                color=COLORS["delta"],
                alpha=0.06,
            )
        intensity_axis.plot(
            ready_iterations,
            intensities,
            color=COLORS["intensity"],
            linewidth=1.8,
            alpha=0.8,
            label="Intensidad",
        )
    ax_bottom.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax_bottom.set(
        xlabel="Iteración",
        ylabel="D2" if d2_view else r"$\Delta$",
    )
    ax_bottom.set_title(
        "Señal DTW/DDTW D2 e intensidad aplicada"
        if d2_view
        else "Señal DTW/DDTW delta e intensidad aplicada"
    )
    intensity_axis.set_ylabel("Intensidad", color=COLORS["intensity"])
    intensity_axis.set_ylim(-0.05, 1.05)
    lines_a, labels_a = ax_bottom.get_legend_handles_labels()
    lines_b, labels_b = intensity_axis.get_legend_handles_labels()
    if lines_a or lines_b:
        ax_bottom.legend(lines_a + lines_b, labels_a + labels_b, loc="upper left")
    ax_bottom.grid(alpha=0.2, linestyle=":")
    fig.tight_layout(h_pad=1.5)
    _save_figure(fig, run_dir, "dtw_adaptation")

    rows = result.parameter_history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    if rows:
        base_a = [row["woa_a_base"] for row in rows]
        effective_a = [row["woa_a_effective"] for row in rows]
        axes[0, 0].plot(iterations, base_a, linestyle=":", color="gray", label="base")
        axes[0, 0].plot(iterations, effective_a, color=COLORS["woa"], label="efectivo")
        axes[0, 0].legend()
        axes[0, 1].plot(iterations, [row["woa_a_scale"] for row in rows], color=COLORS["woa"])
        axes[1, 0].plot(iterations, [row["abc_phi_scale"] for row in rows], color=COLORS["abc"])
        axes[1, 1].plot(iterations, [row["abc_limit_factor"] for row in rows], color=COLORS["abc"])
    titles = ("WOA a: base vs. efectivo", "Factor WOA a", "Factor ABC phi", "Factor ABC limit")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.grid(alpha=0.2, linestyle=":")
    axes[1, 0].set_xlabel("Iteración")
    axes[1, 1].set_xlabel("Iteración")
    fig.suptitle(f"Efecto real del controlador — {result.variant}")
    fig.tight_layout()
    _save_figure(fig, run_dir, "parameters")


__all__ = ["save_run_plots"]
