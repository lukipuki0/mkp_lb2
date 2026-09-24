"""Gráficos de los experimentos WOA--ABC.

Los gráficos que dependen de la señal DTW viven en este paquete para que CEC
y HRES2 utilicen exactamente el mismo estilo visual.
"""

from .dtw import save_dtw_comparison_plot, save_dtw_run_plot

__all__ = ["save_dtw_comparison_plot", "save_dtw_run_plot"]
