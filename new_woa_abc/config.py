"""Configuración central y explícita de los experimentos nuevos."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DTWConfig:
    """Configuración basada en ``DTW_optimization-main/mkp_common/config.py``."""

    window: int = 20
    band: int = 2
    min_slope: float = 2.0
    plateau_max: int = 4
    patience: int = 2
    use_ddtw: bool = True
    adapt_thresholds: bool = True
    p_low: float = 30.0
    p_high: float = 70.0

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window debe ser al menos 2")
        if self.band < 0:
            raise ValueError("band no puede ser negativo")
        if self.min_slope < 0:
            raise ValueError("min_slope no puede ser negativo")
        if self.plateau_max < 1 or self.patience < 1:
            raise ValueError("plateau_max y patience deben ser positivos")
        if not 0 <= self.p_low <= 100 or not 0 <= self.p_high <= 100:
            raise ValueError("los percentiles deben estar entre 0 y 100")

    @property
    def effective_band(self) -> int:
        return self.band if self.band > 0 else max(1, int(0.1 * self.window))


@dataclass(frozen=True)
class ParameterProfile:
    """Factores de los parámetros nativos de WOA y ABC."""

    woa_a_scale: float
    woa_a_floor: float
    woa_a_cap: float
    abc_phi_scale: float
    abc_limit_factor: float
    abc_guide_scale: float = 1.0
    step_scale: float = 1.0
    momentum_scale: float = 1.0


BASE_PROFILE = ParameterProfile(
    woa_a_scale=1.0,
    woa_a_floor=0.0,
    woa_a_cap=2.0,
    abc_phi_scale=1.0,
    abc_limit_factor=1.0,
    abc_guide_scale=1.0,
    step_scale=1.0,
    momentum_scale=1.0,
)

EXPLOIT_PROFILE = ParameterProfile(
    woa_a_scale=0.75,
    woa_a_floor=0.0,
    woa_a_cap=0.99,
    abc_phi_scale=0.50,
    abc_limit_factor=1.50,
    abc_guide_scale=1.0,
    step_scale=0.75,
    momentum_scale=0.75,
)

EXPLORE_PROFILE = ParameterProfile(
    woa_a_scale=1.50,
    woa_a_floor=1.25,
    woa_a_cap=2.50,
    abc_phi_scale=1.50,
    abc_limit_factor=0.50,
    abc_guide_scale=0.45,
    step_scale=1.50,
    momentum_scale=1.0,
)


@dataclass(frozen=True)
class StrategyConfig:
    """Hiperparámetros de las políticas continuas y extensiones."""

    d2_scale: float = 2.0
    sigmoid_k: float = 5.0
    sigmoid_center: float = 0.5
    hysteresis_enter: float = 0.65
    hysteresis_exit: float = 0.35
    four_state_low: float = 0.25
    four_state_mid: float = 0.50
    four_state_high: float = 0.75

    def __post_init__(self) -> None:
        if self.d2_scale <= 0:
            raise ValueError("d2_scale debe ser positivo")
        if self.sigmoid_k <= 0:
            raise ValueError("sigmoid_k debe ser positivo")
        if not 0 <= self.sigmoid_center <= 1:
            raise ValueError("sigmoid_center debe estar entre 0 y 1")
        if not 0 <= self.hysteresis_exit < self.hysteresis_enter <= 1:
            raise ValueError("se requiere 0 <= hysteresis_exit < hysteresis_enter <= 1")
        thresholds = (self.four_state_low, self.four_state_mid, self.four_state_high)
        if not 0 <= thresholds[0] < thresholds[1] < thresholds[2] <= 1:
            raise ValueError("los umbrales de cuatro estados deben ser crecientes")


@dataclass(frozen=True)
class WOAABCConfig:
    """Configuración de una corrida del motor WOA--ABC."""

    pop_size: int = 30
    iterations: int = 1000
    seed: int = 42
    variant: str = "M0_no_dtw"
    b_spiral: float = 1.0
    abc_limit: int | None = None
    abc_guide_strength: float = 0.20
    abc_vector_probability: float = 0.35
    abc_vector_scale: float = 0.10
    step_initial: float = 1.0
    step_final: float = 0.05
    momentum_factor: float = 0.20
    abc_limit_divisor: int = 4
    min_abc_limit: int = 2
    dtw: DTWConfig = field(default_factory=DTWConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    base_profile: ParameterProfile = BASE_PROFILE
    exploit_profile: ParameterProfile = EXPLOIT_PROFILE
    explore_profile: ParameterProfile = EXPLORE_PROFILE

    def __post_init__(self) -> None:
        if self.pop_size < 4:
            raise ValueError("pop_size debe ser al menos 4")
        if self.iterations < 1:
            raise ValueError("iterations debe ser positivo")
        if self.b_spiral <= 0:
            raise ValueError("b_spiral debe ser positivo")
        if self.abc_limit is not None and self.abc_limit < 1:
            raise ValueError("abc_limit debe ser positivo")
        if self.abc_guide_strength < 0:
            raise ValueError("abc_guide_strength no puede ser negativo")
        if not 0.0 <= self.abc_vector_probability <= 1.0:
            raise ValueError("abc_vector_probability debe estar entre 0 y 1")
        if self.abc_vector_scale < 0:
            raise ValueError("abc_vector_scale no puede ser negativo")
        if self.step_initial <= 0 or self.step_final <= 0:
            raise ValueError("step_initial y step_final deben ser positivos")
        if self.momentum_factor < 0:
            raise ValueError("momentum_factor no puede ser negativo")
        if self.abc_limit_divisor < 1 or self.min_abc_limit < 1:
            raise ValueError("abc_limit_divisor y min_abc_limit deben ser positivos")


# La escala de LCOE de HRES2 es muy distinta al fitness entero del MKP.
# Se conserva la arquitectura de referencia, pero la rampa queda automática.
CEC_DTW = DTWConfig()
HRES2_DTW = DTWConfig(min_slope=0.0)
MKP_DTW = DTWConfig()


__all__ = [
    "BASE_PROFILE",
    "CEC_DTW",
    "DTWConfig",
    "EXPLOIT_PROFILE",
    "EXPLORE_PROFILE",
    "HRES2_DTW",
    "MKP_DTW",
    "ParameterProfile",
    "StrategyConfig",
    "WOAABCConfig",
]
