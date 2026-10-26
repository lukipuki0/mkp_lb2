"""Native hybrid figures, recorded temporal diagnostics, and pool probabilities."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

from hybrid_mkp import plots as native_plots
from hybrid_mkp.orchestrator import COLORES_MH, POOL_POBLACIONAL, POOL_TRAYECTORIA, PipelineResult

SELECTION_PROBABILITY_FIGURE = "selection_probabilities.png"
SELECTION_PROBABILITY_PDF = "selection_probabilities.pdf"
_POOL_PANELS = (
    ("poblacional", tuple(POOL_POBLACIONAL), "Poblacional"),
    ("trayectoria", tuple(POOL_TRAYECTORIA), "Trayectoria"),
)


def plot_run(result: PipelineResult, output: Path, metadata: dict | None = None, *,
             selection_records=None) -> list[str]:
    """Keep three native renderers and add recorded diagnostics when data exists.

    grafico_solo_instantaneo stays available for other callers but is not used
    here, and a preexisting instantaneous_fitness_only.png is left untouched.
    The rejected instantaneous_convergence overlay is removed on redraw. Missing
    native delta samples and missing selection records are not fabricated.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    obsolete = output / "instantaneous_convergence.png"
    if obsolete.is_file() or obsolete.is_symlink():
        obsolete.unlink()
    common = dict(log_switches=result.log_switches, colores_mh=COLORES_MH,
                  output_dir=str(output))
    fitness = dict(historial_global=result.historial_global,
                   valor_optimo=result.valor_optimo)
    paths = [
        native_plots.grafico_convergencia(**fitness, **common),
        native_plots.grafico_dtw_delta(dtw_deltas_global=result.dtw_deltas_global, **common),
        native_plots.grafico_switches(**common),
    ]
    from .temporal_diagnostics import plot_temporal_diagnostics
    paths.extend(plot_temporal_diagnostics(result, output, metadata))
    probability = plot_selection_probabilities(selection_records, output, metadata=metadata)
    if probability:
        paths.extend((probability, str(output / SELECTION_PROBABILITY_PDF)))
    return [path for path in paths if path]


def plot_distribution(name: str, values: list[float], reference: float, output: Path) -> str:
    """Reuse the first-instance hybrid runner's run-distribution renderer."""
    from hybrid_mkp.parallel_mkp_first_inst import grafico_boxplot_runs
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # No reference => the native legend has no handles. This is not missing data.
        warnings.filterwarnings("ignore", message="No artists with labels found.*", category=UserWarning)
        grafico_boxplot_runs(nombre_inst=name, valores=values, valor_opt=reference,
                            output_dir=str(output))
    return str(output / "boxplot_runs.png")


def _restore_cell(value):
    if not isinstance(value, str) or not value or value[0] not in "{[":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_selection_trace(path: Path) -> list[dict]:
    """Restore a selection_trace.csv written by exports.write_csv. No optimization."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [{key: _restore_cell(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]


def _finite_probabilities(value):
    if isinstance(value, str):
        value = _restore_cell(value)
    if not isinstance(value, dict) or not value:
        return None
    clean = {}
    for name, raw in value.items():
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean[str(name)] = number
    return clean or None


def _decisions_by_pool(records):
    if not isinstance(records, (list, tuple)) or not records:
        return None
    grouped = {pool: [] for pool, _, _ in _POOL_PANELS}
    for record in records:
        if not isinstance(record, dict) or record.get("pool") not in grouped:
            continue
        probabilities = _finite_probabilities(record.get("probabilities_before"))
        if probabilities is None:
            continue
        grouped[record["pool"]].append(probabilities)
    if not any(grouped.values()):
        return None
    return grouped


def plot_selection_probabilities(records, output: Path, *, metadata: dict | None = None) -> str | None:
    """Plot recorded within-pool probabilities_before over selection decisions.

    Each panel is conditional on that pool and uses COLORES_MH. Values are not
    renormalized, mixed into a global probability, or replaced by choice frequency.
    Panels use configured/recorded MHs; a wholly missing pool falls back to its
    native names only when configuration is unavailable, with no invented curve.
    Returns the PNG path (and writes a matching PDF), or None when no recorded
    probabilities are available.
    Preview from a saved trace, without running solvers::

        plot_selection_probabilities(load_selection_trace(csv_path), output_dir)
    """
    grouped = _decisions_by_pool(records)
    if grouped is None:
        return None
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = (metadata or {}).get("settings", {})
    pools = []
    for (pool, native_order, title), field in zip(
            _POOL_PANELS, ("population_pool", "trajectory_pool")):
        recorded = {name for item in grouped[pool] for name in item}
        configured = settings.get(field)
        names = recorded | set(configured or ())
        # Legacy records with an absent pool keep explicit no-decision panels.
        order = tuple(name for name in native_order if name in names) if names else native_order
        pools.append((pool, order, title))
    panel_specs = []
    for index in range(max(len(order) for _, order, _ in pools)):
        panel_specs.extend((pool, order[index], title)
                           for pool, order, title in pools if index < len(order))
    rows = (len(panel_specs) + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(8.4, 2.04 * rows),
                                sharey=True, sharex=False, squeeze=False)
    try:
        for axis in list(axes.flat)[len(panel_specs):]:
            figure.delaxes(axis)
        for axis, (pool, name, pool_title) in zip(axes.flat, panel_specs):
            axis.set_title(f"{pool_title} — {name}", loc="left", fontsize=9)
            axis.set_xlabel("Decision within pool", fontsize=8)
            axis.set_ylabel("Conditional probability (%)", fontsize=8)
            axis.tick_params(labelsize=7)
            decisions = grouped[pool]
            values = [(index, item[name] * 100.0)
                      for index, item in enumerate(decisions, 1) if name in item]
            if values:
                xs, ys = zip(*values)
                axis.plot(xs, ys, color=COLORES_MH[name], label=name,
                          lw=1.35, marker="o", ms=3)
                axis.set_xlim(.5, max(1, len(decisions)) + .5)
            else:
                message = "No recorded decisions" if not decisions else "No recorded probability"
                axis.text(.5, .5, message, transform=axis.transAxes,
                          ha="center", va="center", fontsize=8)
                axis.set_xlim(0, 1)
            axis.set_ylim(0, 100)
            axis.set_autoscaley_on(False)
        figure.suptitle(
            "Recorded probabilities_before — conditional within each pool; not a global probability; not a selection frequency",
            fontsize=9)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        path = output / SELECTION_PROBABILITY_FIGURE
        figure.savefig(path, dpi=160)
        figure.savefig(output / SELECTION_PROBABILITY_PDF)
    finally:
        plt.close(figure)
    return str(path)
