"""
ts_mkp/__init__.py
------------------
Paquete ts_mkp - Tabu Search para el MKP.

API publica:
  from ts_mkp import TSParams, TSResult, ejecutar_ts
"""

from ts_mkp.algorithm import TSParams, TSResult, TSEpochResult, ejecutar_ts

__all__ = [
    "TSParams",
    "TSResult",
    "TSEpochResult",
    "ejecutar_ts",
]
