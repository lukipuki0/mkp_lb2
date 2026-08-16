"""
continuous_benchmark/plots/
----------------------------
Módulo de visualización del Pipeline Híbrido DTW — exclusivo para CEC2022.

Exporta:
  - grafico_convergencia : Curva de fitness coloreada por MH (leyenda upper right)
  - grafico_dtw_delta    : Curva del Delta DTW a lo largo del pipeline
  - grafico_switches     : Diagrama de Gantt con los turnos de cada MH
"""

from .convergencia    import grafico_convergencia
from .dtw_delta       import grafico_dtw_delta
from .switches_gantt  import grafico_switches

__all__ = [
    "grafico_convergencia",
    "grafico_dtw_delta",
    "grafico_switches",
]
