"""Gráficos DTW con el estilo visual de ``DTW_optimization``.

El ZIP original grafica dos paneles: convergencia con segmentos sólido/
punteado según explotación/exploración, y la señal ``Delta`` junto a
``theta_delta``. Aquí se conserva ese formato para las siete variantes
WOA--ABC. La única adaptación semántica es que CEC/HRES2 minimizan, por lo que
las leyendas hablan de objetivo y seleccionamos la mejor corrida por mínimo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANT_COLORS = {
    "M0": "#2196F3",  # ZIP: PSO
    "M1": "#E91E63",  # ZIP: GA
    "M2": "#4CAF50",  # ZIP: GWO
    "M4": "#FF9800",  # ZIP: DE
    "M5": "#9C27B0",
    "M6": "#009688",
    "M8": "#795548",
}

COLORS = {
    "fire_marker": "#c0392b",
    "optimum": "#7f8c8d",
    "theta_delta": "#e67e22",
}


def _short_name(variant: str) -> str:
    return str(variant).split("_", 1)[0]


def _color(variant: str) -> str:
    return VARIANT_COLORS.get(_short_name(variant), "#666666")


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "N/A"


def _mode_history(epoch: Any, length: int) -> list[str]:
    """Obtiene los modos aplicados, incluyendo los estados de cuatro niveles."""

    values = []
    for item in getattr(epoch, "parametros_historial", []):
        values.append(str(item.get("mode", "base")))
    if len(values) != length:
        values = []
        for item in getattr(epoch, "dtw_info_hist", []):
            values.append(str(item.get("mode_aplicado", "base")))
    if len(values) != length:
        values = (values + ["base"] * length)[:length]
    return values


def _is_explore(mode: str) -> bool:
    return str(mode).startswith("explore")


def _get_explore_ranges(modes: list[str]) -> list[tuple[int, int]]:
    """Detecta rangos contiguos de exploración como el gráfico del ZIP."""

    ranges: list[tuple[int, int]] = []
    in_explore = False
    start = 0
    for index, mode in enumerate(modes):
        explore = _is_explore(mode)
        if explore and not in_explore:
            start = index
            in_explore = True
        elif not explore and in_explore:
            ranges.append((start, index - 1))
            in_explore = False
    if in_explore:
        ranges.append((start, len(modes) - 1))
    return ranges


def _get_fire_transitions(modes: list[str]) -> list[int]:
    """Iteraciones en que comienza una exploración nueva."""

    transitions: list[int] = []
    previous = "exploit"
    for index, mode in enumerate(modes):
        if _is_explore(mode) and not _is_explore(previous):
            transitions.append(index)
        previous = mode
    return transitions


def _get_change_events(epoch: Any, modes: list[str]) -> list[tuple[int, str]]:
    """Devuelve cambios que se aplican a la siguiente iteración.

    El ZIP marca con triángulo el comienzo de un ``fire``. En WOA--ABC se
    conserva ese marcador para una transición de modo y se añade un círculo
    pequeño cuando cambia el perfil de parámetros sin cambiar de modo (por
    ejemplo, una transición entre estados de M8 o un ``decay`` de M1).
    """

    events: list[tuple[int, str]] = []
    for item in getattr(epoch, "eventos_adaptacion", []):
        try:
            # El evento se detecta al final de t y el perfil se aplica en t+1.
            iteration = int(item.get("iteracion", -1)) + 1
        except (TypeError, ValueError):
            continue
        if iteration < 0 or iteration >= len(modes):
            continue
        mode_changed = str(item.get("mode_anterior")) != str(item.get("mode_siguiente"))
        profile_changed = item.get("perfil_anterior") != item.get("perfil_siguiente")
        if mode_changed or item.get("fire_aceptado"):
            events.append((iteration, "transition"))
        elif profile_changed:
            events.append((iteration, "parameter"))

    if events:
        return sorted(set(events))
    return [(iteration, "transition") for iteration in _get_fire_transitions(modes)]


def _set_zip_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_dtw_comparison_plot(
    output_dir: Path,
    traces: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    title: str,
    optimum: float | None,
    scale: str = "linear",
    stem: str = "dtw_fire",
) -> None:
    """Guarda el gráfico DTW de dos paneles con el estilo del ZIP."""

    if not traces:
        return
    _set_zip_style()
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )
    row_by_variant = {str(row.get("variant")): row for row in rows}
    has_transition_marker = False
    has_parameter_marker = False

    # Panel 1: convergencia con segmentos exploit/explore.
    for variant, epoch in traces.items():
        values = list(getattr(epoch, "historial", []))
        if not values:
            continue
        modes = _mode_history(epoch, len(values))
        is_explore = [_is_explore(mode) for mode in modes]
        color = _color(variant)
        previous = is_explore[0]
        segment_start = 0
        for index in range(1, len(values)):
            if is_explore[index] != previous:
                ax1.plot(
                    range(segment_start, index + 1),
                    values[segment_start : index + 1],
                    color=color,
                    linewidth=1.5,
                    linestyle="--" if previous else "-",
                    zorder=3,
                )
                segment_start = index
                previous = is_explore[index]
        ax1.plot(
            range(segment_start, len(values)),
            values[segment_start:],
            color=color,
            linewidth=1.5,
            linestyle="--" if previous else "-",
            zorder=3,
        )

        row = row_by_variant.get(str(variant), {})
        ax1.plot(
            [],
            [],
            color=color,
            linewidth=1.5,
            linestyle="-",
            label=f"{_short_name(variant)} (min={_fmt(row.get('best_value', min(values)))})",
        )
        changes = _get_change_events(epoch, modes)
        transitions = [index for index, kind in changes if kind == "transition"]
        parameter_updates = [index for index, kind in changes if kind == "parameter"]
        if transitions:
            has_transition_marker = True
            valid = [index for index in transitions if index < len(values)]
            ax1.scatter(
                valid,
                [values[index] for index in valid],
                color=COLORS["fire_marker"],
                marker="v",
                s=30,
                zorder=5,
                edgecolors="white",
                linewidths=0.3,
            )
        if parameter_updates:
            has_parameter_marker = True
            valid = [index for index in parameter_updates if index < len(values)]
            ax1.scatter(
                valid,
                [values[index] for index in valid],
                color=COLORS["theta_delta"],
                marker="o",
                s=20,
                zorder=5,
                edgecolors="white",
                linewidths=0.3,
            )

    if optimum is not None:
        ax1.axhline(
            y=optimum,
            color=COLORS["optimum"],
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Known optimum ({_fmt(optimum)})",
        )
    ax1.plot([], [], color="gray", linewidth=1.5, linestyle="-", label="Exploit")
    ax1.plot([], [], color="gray", linewidth=1.5, linestyle="--", label="Explore")
    if has_transition_marker:
        ax1.scatter([], [], color=COLORS["fire_marker"], marker="v", s=30, label="DTW fire")
    if has_parameter_marker:
        ax1.scatter([], [], color=COLORS["theta_delta"], marker="o", s=20, label="Parameter update")
    if scale == "log" and all(value > 0 for epoch in traces.values() for value in epoch.historial):
        ax1.set_yscale("log")
    ax1.set_ylabel("Objective")
    ax1.set_title(title)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle=":")

    # Panel 2: Delta y theta_delta.
    variants = list(traces)
    for variant_index, (variant, epoch) in enumerate(traces.items()):
        color = _color(variant)
        ready_iterations: list[int] = []
        deltas: list[float] = []
        thresholds: list[float] = []
        for index, item in enumerate(getattr(epoch, "dtw_info_hist", [])):
            if item.get("ready") and item.get("delta") is not None:
                ready_iterations.append(index)
                deltas.append(float(item["delta"]))
                thresholds.append(float(item.get("theta_delta", 0.0)))
        if not ready_iterations:
            continue
        ax2.plot(
            ready_iterations,
            deltas,
            color=color,
            linewidth=1.3,
            label=rf"$\Delta$ {_short_name(variant)}",
            zorder=3,
        )
        ax2.plot(
            ready_iterations,
            thresholds,
            color=color,
            linewidth=0.8,
            linestyle=":",
            alpha=0.5,
            label=rf"$\theta_\Delta$ {_short_name(variant)}",
            zorder=2,
        )
        modes = _mode_history(epoch, len(getattr(epoch, "historial", [])))
        for change_iteration, change_kind in _get_change_events(epoch, modes):
            if change_iteration in ready_iterations:
                index = ready_iterations.index(change_iteration)
                y_offset = (variant_index - len(variants) / 2) * 0.5
                ax2.scatter(
                    change_iteration,
                    deltas[index] + y_offset,
                    color=COLORS["fire_marker"] if change_kind == "transition" else COLORS["theta_delta"],
                    marker="v" if change_kind == "transition" else "o",
                    s=40,
                    zorder=5,
                    edgecolors="white",
                    linewidths=0.5,
                )

    ax2.axhline(y=0, color="#95a5a6", linestyle="-", linewidth=0.8, alpha=0.4)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\Delta$ (D1 $-$ D2)")
    ax2.set_title(r"DTW Stagnation Signal — $\Delta > 0$ indicates stagnation")
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax2.grid(True, alpha=0.2, linestyle=":")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def save_dtw_run_plot(
    output_dir: Path,
    epoch: Any,
    optimum: float | None,
    variant: str,
    *,
    scale: str = "linear",
) -> None:
    """Guarda el mismo gráfico de dos paneles para una corrida individual."""

    save_dtw_comparison_plot(
        output_dir,
        {variant: epoch},
        [{"variant": variant, "best_value": min(epoch.historial) if epoch.historial else None}],
        title=f"WOA-ABC — {_short_name(variant)}",
        optimum=optimum,
        scale=scale,
        # Conserva el nombre que ya esperaba el pipeline, pero ahora contiene
        # el gráfico DTW de dos paneles del ZIP (PNG y PDF).
        stem="dtw_delta",
    )


__all__ = ["save_dtw_comparison_plot", "save_dtw_run_plot"]
