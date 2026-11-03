"""Native hybrid figures and a recorded-data temporal diagnostic panel."""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

from hybrid_mkp import plots as native_plots
from hybrid_mkp.orchestrator import COLORES_MH, PipelineResult


def plot_run(result: PipelineResult, output: Path, metadata: dict | None = None) -> list[str]:
    """Keep four native renderers and add a four-panel recorded-data diagnostic.

    The rejected instantaneous_convergence overlay is removed on redraw. The
    instantaneous_fitness_only figure remains. Missing native delta samples are
    not fabricated; original delta plotting may skip its standalone figure.
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
        native_plots.grafico_solo_instantaneo(
            historial_inst_global=result.historial_inst_global,
            valor_optimo=result.valor_optimo, **common),
    ]
    from .temporal_diagnostics import plot_temporal_diagnostics
    paths.extend(plot_temporal_diagnostics(result, output, metadata))
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
