"""Configuración manual exclusiva de los experimentos WOA--ABC.

Editar este archivo permite fijar la ventana DTW sin modificar el benchmark
CEC anterior, HRES2 anterior ni ningún experimento de MCDP. Los argumentos de
línea de comandos siguen pudiendo sobreescribir estos valores puntualmente.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DTWManualConfig:
    window: int = 20
    band: int = 2
    min_slope: float = 2.0
    plateau_max: int = 4
    patience: int = 2
    use_ddtw: bool = True
    adapt_thresholds: bool = True
    p_low: float = 30.0
    p_high: float = 70.0
    improvement_tol: float = 1e-6


@dataclass(frozen=True)
class WOAABCManualConfig:
    """Parámetros base ajustados para la búsqueda continua CEC/HRES2."""

    woa_exploration: float = 1.20
    exploration_probability: float = 0.65
    step_initial: float = 2.0
    step_final: float = 0.10
    momentum_factor: float = 0.20
    abc_limit_divisor: int = 4


# Configuraciones independientes: pueden hacerse diferentes para CEC y HRES2.
CEC_DTW = DTWManualConfig()
HRES2_DTW = DTWManualConfig()
CEC_MH = WOAABCManualConfig()
HRES2_MH = WOAABCManualConfig()


__all__ = [
    "CEC_DTW",
    "HRES2_DTW",
    "CEC_MH",
    "HRES2_MH",
    "DTWManualConfig",
    "WOAABCManualConfig",
]
