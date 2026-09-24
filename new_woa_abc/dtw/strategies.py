"""Políticas DTW inspiradas en A3, A4, B3 y B1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from new_woa_abc.config import StrategyConfig


M0_NO_DTW = "M0_no_dtw"
M1_FIRE_D2 = "M1_fire_d2"
M2_FIRE_3COND = "M2_fire_3cond"
M4_D2_CONTINUOUS = "M4_d2_continuous"
M5_SIGMOID_DELTA = "M5_sigmoid_delta"
M6_HYSTERESIS_WOA_ABC = "M6_hysteresis_woa_abc"
M8_FOUR_STATE_WOA_ABC = "M8_four_state_woa_abc"

VARIANT_NAMES: tuple[str, ...] = (
    M0_NO_DTW,
    M1_FIRE_D2,
    M2_FIRE_3COND,
    M4_D2_CONTINUOUS,
    M5_SIGMOID_DELTA,
    M6_HYSTERESIS_WOA_ABC,
    M8_FOUR_STATE_WOA_ABC,
)


def resolve_variants(values: Sequence[str] | None) -> list[str]:
    if not values or any(value.lower() == "all" for value in values):
        return list(VARIANT_NAMES)
    aliases = {variant.split("_", 1)[0].upper(): variant for variant in VARIANT_NAMES}
    aliases.update({variant.lower(): variant for variant in VARIANT_NAMES})
    resolved: list[str] = []
    for value in values:
        variant = aliases.get(value.upper(), aliases.get(value.lower()))
        if variant is None:
            raise ValueError(
                f"variante desconocida {value!r}; opciones: {', '.join(VARIANT_NAMES)}"
            )
        if variant not in resolved:
            resolved.append(variant)
    return [variant for variant in VARIANT_NAMES if variant in resolved]


@dataclass(frozen=True)
class AdaptationDecision:
    mode: str
    intensity: float | None
    reason: str
    raw_fire: bool = False
    transition: bool = False
    parameter_changed: bool = False


def d2_intensity(status: dict[str, Any], config: StrategyConfig) -> float:
    epsilon = 1e-12
    ratio = float(status["D2_vs_const"]) / (
        abs(float(status["theta_c"])) * config.d2_scale + epsilon
    )
    return 1.0 - max(0.0, min(1.0, ratio))


def sigmoid_delta_intensity(status: dict[str, Any], config: StrategyConfig) -> float:
    epsilon = 1e-12
    # Se conserva la fórmula B1 del repositorio de referencia.
    balance = float(status["delta"]) / (float(status["theta_delta"]) + epsilon)
    argument = config.sigmoid_k * (balance - config.sigmoid_center)
    if argument >= 100:
        return 1.0
    if argument <= -100:
        return 0.0
    return 1.0 / (1.0 + math.exp(-argument))


class AdaptationController:
    """Convierte el diagnóstico DTW en una intensidad reproducible."""

    def __init__(self, variant: str, config: StrategyConfig) -> None:
        if variant not in VARIANT_NAMES:
            raise ValueError(f"variante desconocida: {variant}")
        self.variant = variant
        self.config = config
        self.previous = AdaptationDecision("base", None, "initial")

    def reset(self) -> None:
        self.previous = AdaptationDecision("base", None, "initial")

    def decide(self, status: dict[str, Any]) -> AdaptationDecision:
        if self.variant == M0_NO_DTW or not status.get("ready", False):
            current = AdaptationDecision("base", None, "dtw_disabled" if self.variant == M0_NO_DTW else "warmup")
        elif self.variant == M1_FIRE_D2:
            fire = float(status["D2_vs_const"]) <= float(status["theta_c"])
            current = AdaptationDecision(
                "explore" if fire else "exploit",
                1.0 if fire else 0.0,
                "d2_below_theta_c" if fire else "d2_above_theta_c",
                raw_fire=fire,
            )
        elif self.variant == M2_FIRE_3COND:
            fire = bool(status.get("fire", False))
            current = AdaptationDecision(
                "explore" if fire else "exploit",
                1.0 if fire else 0.0,
                "three_conditions_fire" if fire else "three_conditions_clear",
                raw_fire=fire,
            )
        elif self.variant == M4_D2_CONTINUOUS:
            intensity = d2_intensity(status, self.config)
            current = AdaptationDecision(
                "explore" if intensity > 0.5 else "exploit",
                intensity,
                "continuous_d2",
            )
        elif self.variant == M5_SIGMOID_DELTA:
            intensity = sigmoid_delta_intensity(status, self.config)
            current = AdaptationDecision(
                "explore" if intensity > 0.5 else "exploit",
                intensity,
                "sigmoid_delta",
            )
        elif self.variant == M6_HYSTERESIS_WOA_ABC:
            raw = sigmoid_delta_intensity(status, self.config)
            mode = self.previous.mode
            if mode not in {"explore", "exploit"}:
                mode = "explore" if raw >= self.config.hysteresis_enter else "exploit"
            elif mode == "exploit" and raw >= self.config.hysteresis_enter:
                mode = "explore"
            elif mode == "explore" and raw <= self.config.hysteresis_exit:
                mode = "exploit"
            current = AdaptationDecision(
                mode,
                1.0 if mode == "explore" else 0.0,
                "sigmoid_hysteresis",
            )
        else:
            raw = sigmoid_delta_intensity(status, self.config)
            if raw < self.config.four_state_low:
                mode, intensity = "exploit_high", 0.0
            elif raw < self.config.four_state_mid:
                mode, intensity = "exploit_low", 1.0 / 3.0
            elif raw < self.config.four_state_high:
                mode, intensity = "explore_low", 2.0 / 3.0
            else:
                mode, intensity = "explore_high", 1.0
            current = AdaptationDecision(mode, intensity, "four_state_sigmoid")

        transition = current.mode != self.previous.mode
        old_intensity = self.previous.intensity
        new_intensity = current.intensity
        changed = transition or (
            old_intensity is None and new_intensity is not None
        ) or (
            old_intensity is not None
            and new_intensity is not None
            and abs(old_intensity - new_intensity) > 1e-12
        )
        current = AdaptationDecision(
            current.mode,
            current.intensity,
            current.reason,
            current.raw_fire,
            transition,
            changed,
        )
        self.previous = current
        return current


__all__ = [
    "AdaptationController",
    "AdaptationDecision",
    "M0_NO_DTW",
    "M1_FIRE_D2",
    "M2_FIRE_3COND",
    "M4_D2_CONTINUOUS",
    "M5_SIGMOID_DELTA",
    "M6_HYSTERESIS_WOA_ABC",
    "M8_FOUR_STATE_WOA_ABC",
    "VARIANT_NAMES",
    "d2_intensity",
    "resolve_variants",
    "sigmoid_delta_intensity",
]
