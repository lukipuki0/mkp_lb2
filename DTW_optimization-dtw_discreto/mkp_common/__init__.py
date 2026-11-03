"""
mkp_common — Core compartido para MKP Solver
=============================================
Problema, metaheurísticas, monitor DTW y utilidades.
Compartido entre todas las estrategias de adaptación.
"""

from .problem import cargar_instancia, reparar
from .monitor import StagnationConfig, StagnationMonitor
from .base import BaseMH
from .mh import BinaryPSO, GeneticAlgorithm, BinaryGWO, BinaryDE

__all__ = [
    "cargar_instancia",
    "reparar",
    "StagnationConfig",
    "StagnationMonitor",
    "BaseMH",
    "BinaryPSO",
    "GeneticAlgorithm",
    "BinaryGWO",
    "BinaryDE",
]
