"""MKP plot renderers copied for the WOA-ABC single-run exporter.

The package ``hybrid_mkp`` is not imported. ``hybrid_mkp/__init__.py`` loads
the orchestrator, and this exporter must stay isolated from that import.

Provenance
----------
- ``hybrid_mkp/plots/convergencia.py:grafico_convergencia``
- ``hybrid_mkp/plots/dtw_delta.py:grafico_dtw_delta``
- ``hybrid_mkp/batch_benchmark.py:grafico_boxplot_runs``

``hybrid_mkp/plots/switches_gantt.py`` is not used. It draws timed algorithm
switches, and this solver does not switch algorithms.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


DPI = 150
CONVERGENCE_SIZE = (14, 6)
DELTA_SIZE = (14, 5)
PARAMETER_SIZE = (14, 6)
BOX_SIZE = (6, 5)
SERIES_COLOR = "#1565C0"
RECOVERY_COLOR = "#FF5722"
FIRE_COLOR = "#FF9800"
SEGMENT_NAME = "WOA\u2013ABC"

EMPTY_DELTA_LABEL = (
    "Empty diagnostic: no finite DDTW deltas.\n"
    "Not evidence of improvement.\n"
    "Diagnostico vacio: no indica mejora ni estancamiento."
)
BOX_TITLE = "1 run (n = 1) — not inferential evidence"
BOX_NOTE = "Un solo run. Sin afirmacion de superioridad."


class _Segment:
    """One algorithm turn, the object shape read by the copied loops."""

    def __init__(self, mh_nombre: str, n_iters: int) -> None:
        self.mh_nombre = mh_nombre
        self.n_iters = n_iters


def _paint_convergencia(ax, historial_global, log_switches, colores_mh, valor_optimo) -> None:
    """Copied from hybrid_mkp/plots/convergencia.py:grafico_convergencia.

    Adaptations: the title names WOA–ABC, and the x-axis says Hybrid cycles.
    The loop, widths, segment join, color mapping, and Patch legend are unchanged.
    """

    offset = 0
    legend_patches = []
    seen = set()

    dibujar_lineas = len(log_switches) < 50

    for sw in log_switches:
        n_seg = sw.n_iters
        if n_seg == 0:
            continue
        seg = historial_global[offset: offset + n_seg]
        col = colores_mh.get(sw.mh_nombre, "gray")

        # Conectar con el punto final de la MH anterior para trazar una línea continua de corrido
        if offset > 0:
            xs_plot = list(range(offset - 1, offset + len(seg)))
            ys_plot = [historial_global[offset - 1]] + seg
        else:
            xs_plot = list(range(offset, offset + len(seg)))
            ys_plot = seg

        ax.plot(xs_plot, ys_plot, color=col, linewidth=3.0, alpha=0.85)
        if dibujar_lineas:
            ax.axvline(x=offset, color=col, linestyle="--", linewidth=1.5, alpha=0.5)
        offset += n_seg

        if sw.mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=sw.mh_nombre))
            seen.add(sw.mh_nombre)

    if valor_optimo > 0:
        ax.axhline(y=valor_optimo, color="red", linestyle="--",
                   linewidth=2.0)
        legend_patches.append(
            mpatches.Patch(color="red", label=f"Optimum ({valor_optimo:.0f})")
        )

    ax.set_title("WOA\u2013ABC - Fitness Convergence",
                 fontsize=20, fontweight="bold")
    ax.set_xlabel("Hybrid cycles", fontsize=18)
    ax.set_ylabel("Best Fitness Value", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    # Keep the legend inside the axes, anchored at the lower-right corner.
    ax.legend(handles=legend_patches, loc="lower right", fontsize=15,
              borderaxespad=0.25, frameon=True, framealpha=0.88)
    ax.grid(True, alpha=0.3)


def _png_name(filename: str, default: str) -> str:
    """Keep a requested file name inside the output directory."""

    name = os.path.basename(filename)
    return name or default


def grafico_convergencia(
    historial_global: list,
    log_switches: list,
    colores_mh: dict,
    valor_optimo: float,
    output_dir: str,
    filename: str = "fitness_convergence.png",
) -> str:
    """Copied wrapper from hybrid_mkp/plots/convergencia.py:grafico_convergencia.

    ``filename`` is the only save adaptation. It defaults to the MKP name and
    cannot leave ``output_dir``.
    """

    if not historial_global:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))
    _paint_convergencia(ax, historial_global, log_switches, colores_mh, valor_optimo)
    fig.tight_layout()

    safe_name = _png_name(filename, "fitness_convergence.png")
    path = os.path.join(output_dir, safe_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if safe_name == "fitness_convergence.png":
        print(f"  [plot] fitness_convergence.png   -> '{path}'")
    else:
        print(f"  [plot] {safe_name} -> '{path}'")
    return path


def _collect_delta_points(dtw_deltas_global, log_switches) -> tuple[list, list]:
    """Copied point filter from hybrid_mkp/plots/dtw_delta.py:grafico_dtw_delta."""

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
    return xs_validos, ys_validos


def _paint_dtw_delta(ax, dtw_deltas_global, log_switches, colores_mh) -> None:
    """Copied from hybrid_mkp/plots/dtw_delta.py:grafico_dtw_delta.

    Adaptations: the title names WOA–ABC, and the x-axis says Hybrid cycles.
    Segment joins, widths, color mapping, and the Patch legend are unchanged.
    """

    xs_validos, ys_validos = _collect_delta_points(dtw_deltas_global, log_switches)
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

    # Two lines so the adapted WOA–ABC title stays inside the 14x5 figure.
    # Source fonts are unchanged: title 20, axis labels 18, ticks and legend 15.
    ax.set_title("WOA\u2013ABC DTW/DDTW Delta\n[+ = stagnation | - = active improvement]",
                 fontsize=20, fontweight="bold")
    ax.set_xlabel("Hybrid cycles", fontsize=18)
    ax.set_ylabel("DTW Delta", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    # Keep the legend inside the axes, anchored at the lower-right corner.
    ax.legend(handles=legend_patches, loc="lower right", fontsize=15,
              borderaxespad=0.25, frameon=True, framealpha=0.88)

    ax.grid(True, alpha=0.3)


def grafico_dtw_delta(
    dtw_deltas_global: list,
    log_switches: list,
    colores_mh: dict,
    output_dir: str,
    filename: str = "dtw_delta.png",
) -> str:
    """Copied wrapper from hybrid_mkp/plots/dtw_delta.py:grafico_dtw_delta.

    The original returns without a file when no finite delta exists. The
    exporter does not use that skip; ``plot_delta`` still writes the diagnostic.
    ``filename`` defaults to the MKP name and cannot leave ``output_dir``.
    """

    _xs_validos, ys_validos = _collect_delta_points(dtw_deltas_global, log_switches)
    if not ys_validos:
        print("  [plot] dtw_delta.png  -> sin datos suficientes, omitido.")
        return ""

    fig, ax = plt.subplots(figsize=(14, 5))
    _paint_dtw_delta(ax, dtw_deltas_global, log_switches, colores_mh)
    fig.tight_layout()

    safe_name = _png_name(filename, "dtw_delta.png")
    path = os.path.join(output_dir, safe_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if safe_name == "dtw_delta.png":
        print(f"  [plot] dtw_delta.png             -> '{path}'")
    else:
        print(f"  [plot] {safe_name} -> '{path}'")
    return path


def _delta_series(rows: list) -> list:
    """Cycle-ordered delta list. Missing and non-finite values stay holes."""

    series = []
    for row in rows:
        diagnostic = row.diagnostic
        if not isinstance(diagnostic, dict) or diagnostic.get("delta") is None:
            series.append("")
            continue
        value = diagnostic.get("delta")
        try:
            number = float(value)
        except (TypeError, ValueError):
            series.append("")
            continue
        if not math.isfinite(number):
            series.append("")
            continue
        series.append(number)
    return series


def _one_segment(length: int) -> list[_Segment]:
    if length <= 0:
        return []
    return [_Segment(SEGMENT_NAME, length)]


def _colors() -> dict[str, str]:
    return {SEGMENT_NAME: SERIES_COLOR}


def draw_convergence(ax, rows: list, reference: float) -> None:
    """Draw one full-run segment with the copied MKP convergence renderer."""

    historial = [float(row.best) for row in rows]
    if not historial:
        return
    _paint_convergencia(ax, historial, _one_segment(len(historial)), _colors(), float(reference))


def _draw_empty_delta(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.text(
        0.5,
        0.5,
        EMPTY_DELTA_LABEL,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=14,
    )
    ax.set_title("DDTW Delta per Hybrid Cycle — empty diagnostic", fontsize=20, fontweight="bold")
    ax.set_xlabel("Hybrid cycles", fontsize=18)
    ax.set_ylabel("DTW Delta", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid(True, alpha=0.3)


def draw_delta(ax, rows: list) -> None:
    """Draw one full-run segment with the copied MKP delta renderer.

    An empty series still draws the diagnostic. The original skip remains only
    in ``grafico_dtw_delta``.
    """

    series = _delta_series(rows)
    _xs, ys = _collect_delta_points(series, _one_segment(len(series)))
    if not ys:
        _draw_empty_delta(ax)
        return
    _paint_dtw_delta(ax, series, _one_segment(len(series)), _colors())


def _style_copied_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Typography copied from hybrid_mkp/plots/convergencia.py:grafico_convergencia."""

    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.legend(
        loc="lower right",
        fontsize=15,
        borderaxespad=0.25,
        frameon=True,
        framealpha=0.88,
    )
    ax.grid(True, alpha=0.3)


def draw_parameters(ax, rows: list) -> None:
    """a/rho clock. No MKP source plot exists; the Gantt chart is not reused.

    Recovery markers stay on this axes only. Convergence and delta no longer
    draw them, because those copied renderers mark algorithm segments.
    """

    cycles = [row.cycle for row in rows]
    ax.plot(cycles, [row.a for row in rows], color=SERIES_COLOR, linewidth=2.5, label="a")
    ax.plot(cycles, [row.rho for row in rows], color="#2E7D32", linewidth=2.5, label="rho")
    recovery_x = []
    recovery_y = []
    for row in rows:
        if not row.recovery_applied:
            continue
        recovery_x.append(row.cycle)
        recovery_y.append(row.a)
    if recovery_x:
        if len(recovery_x) < 50:
            for cycle in recovery_x:
                ax.axvline(
                    cycle,
                    color=RECOVERY_COLOR,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.5,
                )
        ax.scatter(
            recovery_x,
            recovery_y,
            marker="^",
            s=90,
            color=RECOVERY_COLOR,
            zorder=5,
            label="parameter recovery",
        )
    fire_x = []
    fire_y = []
    for row in rows:
        fired = bool(row.diagnostic and row.diagnostic.get("fire"))
        if fired and not row.recovery_applied:
            fire_x.append(row.cycle)
            fire_y.append(row.a)
    if fire_x:
        ax.scatter(
            fire_x,
            fire_y,
            marker="x",
            s=80,
            color=FIRE_COLOR,
            zorder=5,
            label="fire without recovery",
        )
    _style_copied_axes(
        ax,
        "WOA\u2013ABC a/rho clock and recovery markers",
        "Hybrid cycles",
        "Parameter value",
    )


def _draw_boxplot_artists(ax, valores: list[float], valor_opt: float, nombre_inst: str) -> None:
    """Copied from hybrid_mkp/batch_benchmark.py:grafico_boxplot_runs.

    Adaptations: the xlabel is WOA–ABC, and the title states the real sample
    count n = 1. The fixed run-count title is not copied. Reference 0 is not
    plotted. No best-known solution is injected.
    """

    bp = ax.boxplot(
        valores,
        patch_artist=True,
        medianprops=dict(color="#FF5722", linewidth=2.5),
        boxprops=dict(facecolor="#1565C0", alpha=0.7),
        flierprops=dict(marker="o", color="#FF9800", markersize=5),
        whiskerprops=dict(color="#90CAF9", linewidth=1.5),
        capprops=dict(color="#90CAF9", linewidth=2),
    )
    del bp

    if valor_opt > 0:
        ax.axhline(valor_opt, color="#4CAF50", linestyle="--", linewidth=1.5,
                   label=f"Known Optimum = {valor_opt:.1f}")

    mu = np.mean(valores)
    ax.scatter([1], [mu], color="#FFEB3B", zorder=5, s=60, label=f"Mean = {mu:.1f}")

    ax.set_title(f"{BOX_TITLE}\n{nombre_inst}", fontsize=11, fontweight="bold")
    ax.set_xlabel("WOA\u2013ABC", fontsize=10)
    ax.set_ylabel("Final Best Value (Maximization)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)


def draw_boxplot(fig, ax, value: float, name: str, reference: float) -> None:
    _draw_boxplot_artists(ax, [float(value)], float(reference), name)
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02, BOX_NOTE, ha="center", fontsize=9)


def plot_convergence(rows: list, reference: float, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    historial = [float(row.best) for row in rows]
    if not historial:
        fig, ax = plt.subplots(figsize=CONVERGENCE_SIZE)
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        return
    grafico_convergencia(
        historial,
        _one_segment(len(historial)),
        _colors(),
        float(reference),
        str(path.parent),
        filename=path.name,
    )


def plot_delta(rows: list, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series = _delta_series(rows)
    switches = _one_segment(len(series))
    _xs, ys = _collect_delta_points(series, switches)
    if not ys:
        fig, ax = plt.subplots(figsize=DELTA_SIZE)
        draw_delta(ax, rows)
        fig.tight_layout()
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        return
    grafico_dtw_delta(series, switches, _colors(), str(path.parent), filename=path.name)


def plot_parameters(rows: list, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=PARAMETER_SIZE)
    draw_parameters(ax, rows)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_boxplot(value: float, name: str, reference: float, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=BOX_SIZE)
    draw_boxplot(fig, ax, value, name, reference)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
