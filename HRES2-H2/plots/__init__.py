"""
HRES2-H2/plots/
---------------
Módulo de visualización del Pipeline Híbrido DTW — exclusivo para HRES2-H2.

Exporta:
  - grafico_convergencia_hres2 : Curva LCOE coloreada por MH (sin línea de óptimo externo)
  - grafico_dtw_delta          : Curva del Delta DTW a lo largo del pipeline
  - grafico_switches           : Diagrama de Gantt con los turnos de cada MH
"""

from .convergencia    import grafico_convergencia_hres2
from .dtw_delta       import grafico_dtw_delta
from .switches_gantt  import grafico_switches

__all__ = [
    "grafico_convergencia_hres2",
    "grafico_dtw_delta",
    "grafico_switches",
]

