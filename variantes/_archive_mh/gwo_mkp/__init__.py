"""
gwo_mkp/__init__.py
-------------------
Paquete gwo_mkp - Grey Wolf Optimizer para el MKP.

API publica:
  from gwo_mkp import GWOParams, GWOResult, ejecutar_gwo
"""

from gwo_mkp.algorithm import GWOParams, GWOResult, GWOEpochResult, ejecutar_gwo
from mkp_core.data_loader import cargar_instancias, seleccionar_instancia
from mkp_core.problem    import MKPInstance

__all__ = [
    "GWOParams",
    "GWOResult",
    "GWOEpochResult",
    "ejecutar_gwo",
    "cargar_instancias",
    "seleccionar_instancia",
    "MKPInstance",
]
