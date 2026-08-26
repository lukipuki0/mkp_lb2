"""
hybrid_mkp
----------
Framework modular de rotación híbrida de metaheurísticas con detección DTW
para el problema de la mochila multidimensional (MKP).
"""

from hybrid_mkp.orchestrator import (
    ejecutar_pipeline,
    ejecutar_mh_standalone,
    PipelineResult,
    SwitchLog,
    COLORES_MH,
    POOL_POBLACIONAL,
    POOL_TRAYECTORIA,
)

__all__ = [
    "ejecutar_pipeline",
    "ejecutar_mh_standalone",
    "PipelineResult",
    "SwitchLog",
    "COLORES_MH",
    "POOL_POBLACIONAL",
    "POOL_TRAYECTORIA",
]
