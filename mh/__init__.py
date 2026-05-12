"""
mh – Paquete unificado de metaheurísticas para el MKP.

Contiene 5 algoritmos listos para el pipeline híbrido:
  - SA  (Simulated Annealing)
  - TS  (Tabu Search)
  - GA  (Genetic Algorithm)
  - PSO (Particle Swarm Optimization + LB2)
  - GWO (Grey Wolf Optimizer + LB2)

Cada algoritmo expone: Params, EpochResult, Result, ejecutar_epoch().
"""

from mh.sa  import SAParams,  SAEpochResult,  SAResult,  ejecutar_epoch as sa_epoch   # noqa: F401
from mh.ts  import TSParams,  TSEpochResult,  TSResult,  ejecutar_epoch as ts_epoch   # noqa: F401
from mh.ga  import GAParams,  GAEpochResult,  GAResult,  ejecutar_epoch as ga_epoch   # noqa: F401
from mh.pso import PSOParams, PSOEpochResult, PSOResult, ejecutar_epoch as pso_epoch  # noqa: F401
from mh.gwo import GWOParams, GWOEpochResult, GWOResult, ejecutar_epoch as gwo_epoch  # noqa: F401

