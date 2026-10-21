"""Gráficos de corrida. El panel DTW copia el cuerpo de HRES2-H2/plots/dtw_delta.py."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from woa_abc_version_vieja.core.woa_abc import OptimizationResult


COLORS = {
    "fitness": "#1a5276",
    "woa": "#2471a3",
    "abc": "#17a589",
    "explore": "#e74c3c",
    "optimum": "#7f8c8d",
    "explore_bg": "#e74c3c",
    "fire": "#f39c12",
    "delta_line": "#1f77b4",
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
    fig.savefig(directory / f"{stem}.pdf")
    plt.close(fig)


_NO_MONITOR = "M0_no_dtw"


class _ControlSegment:
    """Shape adapter for the copied renderer. Not a metaheuristic switch.

    ``mh_nombre`` carries the WOA--ABC variant label. ``n_iters`` is the
    monitor/control segment length between confirmed-fire boundaries.
    """

    def __init__(self, mh_nombre: str, n_iters: int) -> None:
        self.mh_nombre = mh_nombre
        self.n_iters = n_iters


def render_reference_dtw_delta(
    dtw_deltas_global: list,
    log_switches: list,
    colores_mh: dict,
    *,
    title_prefix: str = "WOA-ABC",
) -> plt.Figure | None:
    """Drawing body copied from ``HRES2-H2/plots/dtw_delta.py:grafico_dtw_delta``.

    Provenance: the collection loop, the per-segment plot that prepends the
    previous valid point, the same-color dashed boundary, and the axis
    styling are that function. This copy returns the figure. It does not
    save ``dtw_delta.png``, print the omit line, or close the figure.
    ``title_prefix`` replaces the source text ``HRES2-H2`` and nothing else.
    """

    xs_validos = []
    ys_validos = []
    offset = 0
    for sw in log_switches:
        n_seg = sw.n_iters
        seg_d = dtw_deltas_global[offset: offset + n_seg]
        for idx, val in enumerate(seg_d):
            if val != "" and not (isinstance(val, float) and math.isnan(val)):
                xs_validos.append(offset + idx)
                ys_validos.append(val)
        offset += n_seg

    if not ys_validos:
        return None

    fig, ax = plt.subplots(figsize=(14, 5))

    legend_patches = []
    seen = set()
    dibujar_lineas = len(log_switches) < 50

    offset = 0
    for sw in log_switches:
        n_seg = sw.n_iters
        if n_seg == 0:
            continue
        col = colores_mh.get(sw.mh_nombre, "gray")

        # Puntos del segmento actual
        seg_xs = [x for x in range(offset, offset + n_seg) if x in xs_validos]
        seg_ys = [ys_validos[xs_validos.index(x)] for x in seg_xs]

        if seg_xs:
            # Para conectar sin huecos con el punto válido anterior
            prev_indices = [i for i, x in enumerate(xs_validos) if x < seg_xs[0]]
            if prev_indices:
                last_prev_idx = prev_indices[-1]
                plot_xs = [xs_validos[last_prev_idx]] + seg_xs
                plot_ys = [ys_validos[last_prev_idx]] + seg_ys
            else:
                plot_xs = seg_xs
                plot_ys = seg_ys

            ax.plot(plot_xs, plot_ys, color=col, linewidth=2.5, alpha=0.85)

        if dibujar_lineas:
            ax.axvline(x=offset, color=col, linestyle="--", linewidth=1.5, alpha=0.5)

        offset += n_seg

        if sw.mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=sw.mh_nombre))
            seen.add(sw.mh_nombre)

    ax.set_title(
        f"{title_prefix} DTW Delta per Iteration  [+ = stagnation | - = active improvement]",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_xlabel("Accumulated Iterations", fontsize=18)
    ax.set_ylabel("DTW Delta", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(handles=legend_patches, loc="lower right", fontsize=15)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _aligned_reference_deltas(history: list[dict[str, Any]]) -> list[object]:
    """One slot per iteration. Not-ready and non-finite stay empty, never zero."""

    values: list[object] = []
    for status in history:
        if not status.get("ready"):
            values.append("")
            continue
        raw = status.get("delta")
        if isinstance(raw, float) and math.isnan(raw):
            values.append("")
            continue
        try:
            number = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            values.append("")
            continue
        if not math.isfinite(number):
            values.append("")
            continue
        values.append(number)
    return values


def _control_segments(variant: str, history: list[dict[str, Any]]) -> list[_ControlSegment]:
    """Lengths from sorted unique ``[0] + raw_fire indices + [history length]``.

    Iteration 0 is the source initial boundary, not an extra fire. Later
    boundaries are confirmed detection indices only.
    """

    total = len(history)
    fires = [
        index
        for index in confirmed_fire_iterations(history)
        if 0 <= index < total
    ]
    boundaries = sorted({0, *fires, total})
    segments: list[_ControlSegment] = []
    for start, stop in zip(boundaries, boundaries[1:]):
        length = stop - start
        if length == 0:
            continue
        segments.append(_ControlSegment(variant, length))
    return segments


def _reference_empty_panel(title: str) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.set_title(title, fontsize=18, fontweight="bold")
    axis.set_xlabel("Accumulated Iterations", fontsize=18)
    axis.set_ylabel("DTW Delta", fontsize=18)
    axis.tick_params(axis="both", which="major", labelsize=15)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def confirmed_fire_iterations(history: Iterable[dict]) -> list[int]:
    """Índices de detección de fires confirmados.

    ``raw_fire`` es el alias de esquema del evento (meseta, distancias y
    patience). Las banderas de condición no dibujan marcas.
    """

    indices: list[int] = []
    for status in history:
        if status.get("raw_fire") is not True:
            continue
        index = int(status["iteration"])
        if not indices or indices[-1] != index:
            indices.append(index)
    return indices


def draw_delta_figure(
    variant: str,
    dtw_history: Iterable[dict[str, Any]],
) -> plt.Figure:
    """Panel único de delta. No escribe archivos ni cierra la figura.

    Both engines call this adapter. The curve and boundaries come only from
    ``render_reference_dtw_delta``. M0 is the empty-panel exception: the
    copied omit path would drop the artifact, and there is no monitor series.
    """

    history = list(dtw_history)
    with plt.rc_context({"font.family": "sans-serif"}):
        if variant == _NO_MONITOR:
            return _reference_empty_panel("DTW desactivado")
        figure = render_reference_dtw_delta(
            _aligned_reference_deltas(history),
            _control_segments(variant, history),
            {variant: COLORS["delta_line"]},
        )
        if figure is None:
            return _reference_empty_panel(
                "WOA-ABC DTW Delta per Iteration  [+ = stagnation | - = active improvement]"
            )
        return figure


def draw_dtw_adaptation(
    result: OptimizationResult,
    optimum: float | None = None,
) -> plt.Figure:
    """Figura ``dtw_adaptation`` de un panel, sin escribir PDF.

    Dibuja el delta real de la variante. ``optimum`` no entra en este panel:
    la referencia de costo sigue solo en ``convergence.pdf``. El llamador
    guarda la figura, solo como PDF, en una carpeta nueva.
    """

    del optimum
    return draw_delta_figure(result.variant, result.dtw_history)


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

    _save_figure(draw_dtw_adaptation(result, optimum), run_dir, "dtw_adaptation")

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


__all__ = [
    "confirmed_fire_iterations",
    "draw_dtw_adaptation",
    "render_reference_dtw_delta",
    "save_run_plots",
]
