"""
Plots of one WOA-ABC hybrid run, copied from gwo_de_dtw/plots/ on
feat/gwo-de-dtw-hybrid (cross-branch import is impossible).

Every function takes the GWO-DE run record; record.plot_record builds it
from a WOA-ABC result.

One PNG per plot, each function `(res, output_dir) -> path`:
  - plot_convergence : fitness_convergence.png, as hybrid_mkp (segments = profile turns)
  - plot_dtw_delta   : dtw_delta.png, as hybrid_mkp (segments = profile turns)
  - plot_parameters  : parameters.png, a and rho used per cycle
  - plot_temporal    : four-panel PNG/PDF and verified temporal series (NPZ/CSV)

One group of runs (the four policies):
  - plot_convergence_comparison : convergence_comparison.png/.pdf, best-so-far of
    every record on one axes (convergence_comparison_figure builds the figure)

Spec: odd/tasks/woa-abc-gwo-de-reaction.md (S8).
"""

import os
from typing import List

from .convergence import plot_convergence
from .dtw_delta import plot_dtw_delta
from .parameters import plot_parameters

from .temporal import plot_temporal
from .comparison import convergence_comparison_figure, plot_convergence_comparison
from .record import plot_record

PLOTS = (plot_convergence, plot_dtw_delta, plot_parameters, plot_temporal)


def save_all_plots(res: dict, output_dir) -> List[str]:
    """Creates `output_dir` and saves every plot; returns the paths written."""
    os.makedirs(output_dir, exist_ok=True)
    paths = [plot(res, output_dir) for plot in PLOTS]
    return [path for path in paths if path]


__all__ = [
    "convergence_comparison_figure",
    "plot_convergence",
    "plot_convergence_comparison",
    "plot_dtw_delta",
    "plot_parameters",
    "plot_record",
    "plot_temporal",
    "save_all_plots",
]
