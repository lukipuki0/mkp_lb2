"""
__init__.py
───────────
Paquete sa_mkp – Simulated Annealing para el MKP.

API pública del paquete:
  from sa_mkp import MKPInstance, SAParams, SAResult
  from sa_mkp import cargar_instancias, seleccionar_instancia
  from sa_mkp import ejecutar_sa
  from sa_mkp import imprimir_resumen, graficar_convergencia
"""

from mkp_core.problem import MKPInstance
from sa_mkp.algorithm import SAParams, SAResult, ejecutar_sa, ejecutar_epoch
from mkp_core.data_loader import cargar_instancias, seleccionar_instancia
from mkp_core.repair import reparar_solucion
from sa_mkp.neighborhood import flip_bits
from dtw_stagnation import StagnationConfig, StagnationMonitor
from sa_mkp.results import imprimir_resumen, graficar_convergencia

__all__ = [
    "MKPInstance",
    "SAParams",
    "SAResult",
    "ejecutar_sa",
    "ejecutar_epoch",
    "cargar_instancias",
    "seleccionar_instancia",
    "reparar_solucion",
    "flip_bits",
    "StagnationConfig",
    "StagnationMonitor",
    "imprimir_resumen",
    "graficar_convergencia",
]
