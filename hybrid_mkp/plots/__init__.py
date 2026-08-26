"""
plots/
------
Módulo de visualización del Pipeline Híbrido DTW para MKP.

Exporta funciones para generar cada gráfico de forma independiente:
  - grafico_convergencia   : Curva de fitness coloreada por MH (leyenda upper right)
  - grafico_dtw_delta      : Curva del Delta DTW a lo largo del pipeline
  - grafico_switches       : Diagrama de Gantt con los turnos de cada MH
  - grafico_instantaneo    : Curva combinada mejor global + fitness instantáneo
  - grafico_solo_instantaneo : Curva individual de fitness instantáneo
"""

from hybrid_mkp.plots.convergencia  import grafico_convergencia
from hybrid_mkp.plots.dtw_delta     import grafico_dtw_delta
from hybrid_mkp.plots.switches_gantt import grafico_switches
from hybrid_mkp.plots.instantaneo   import grafico_instantaneo
from hybrid_mkp.plots.solo_instantaneo import grafico_solo_instantaneo

__all__ = [
    "grafico_convergencia",
    "grafico_dtw_delta",
    "grafico_switches",
    "grafico_instantaneo",
    "grafico_solo_instantaneo",
]
