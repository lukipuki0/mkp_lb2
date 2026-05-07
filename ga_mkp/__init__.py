"""
ga_mkp/__init__.py
------------------
Paquete ga_mkp - Genetic Algorithm para el MKP.

API publica:
  from ga_mkp import GAParams, GAResult, ejecutar_ga
  from ga_mkp import imprimir_resumen_ga, graficar_convergencia_ga
"""

from ga_mkp.algorithm import GAParams, GAResult, GAEpochResult, ejecutar_ga
from mkp_core.data_loader import cargar_instancias, seleccionar_instancia
from mkp_core.problem    import MKPInstance

__all__ = [
    "GAParams",
    "GAResult",
    "GAEpochResult",
    "ejecutar_ga",
    "cargar_instancias",
    "seleccionar_instancia",
    "MKPInstance",
]
