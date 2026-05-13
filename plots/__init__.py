"""
plots/
------
Módulo de visualización del Pipeline Híbrido DTW.

Exporta funciones para generar cada gráfico de forma independiente:
  - grafico_convergencia   : Curva de fitness coloreada por MH
  - grafico_instantaneo    : Fitness instantáneo y mejor histórico superpuestos
  - grafico_solo_instantaneo: Únicamente el fitness instantáneo
  - grafico_dtw_delta      : Curva del Delta DTW a lo largo del pipeline
  - grafico_switches       : Diagrama de Gantt con los turnos de cada MH
"""

from plots.convergencia  import grafico_convergencia
from plots.instantaneo   import grafico_instantaneo
from plots.solo_instantaneo import grafico_solo_instantaneo
from plots.dtw_delta     import grafico_dtw_delta
from plots.switches_gantt import grafico_switches

__all__ = [
    "grafico_convergencia",
    "grafico_instantaneo",
    "grafico_solo_instantaneo",
    "grafico_dtw_delta",
    "grafico_switches",
]
