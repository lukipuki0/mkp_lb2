"""Interpolación explícita entre perfiles de explotación y exploración."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from new_woa_abc.config import ParameterProfile, WOAABCConfig
from new_woa_abc.dtw.strategies import AdaptationDecision


def _mix(left: float, right: float, intensity: float) -> float:
    return left + intensity * (right - left)


def interpolate_profile(
    exploit: ParameterProfile,
    explore: ParameterProfile,
    intensity: float,
) -> ParameterProfile:
    intensity = max(0.0, min(1.0, float(intensity)))
    return ParameterProfile(
        woa_a_scale=_mix(exploit.woa_a_scale, explore.woa_a_scale, intensity),
        woa_a_floor=_mix(exploit.woa_a_floor, explore.woa_a_floor, intensity),
        woa_a_cap=_mix(exploit.woa_a_cap, explore.woa_a_cap, intensity),
        abc_phi_scale=_mix(exploit.abc_phi_scale, explore.abc_phi_scale, intensity),
        abc_limit_factor=_mix(exploit.abc_limit_factor, explore.abc_limit_factor, intensity),
        abc_guide_scale=_mix(exploit.abc_guide_scale, explore.abc_guide_scale, intensity),
        step_scale=_mix(exploit.step_scale, explore.step_scale, intensity),
        momentum_scale=_mix(exploit.momentum_scale, explore.momentum_scale, intensity),
    )


@dataclass(frozen=True)
class EffectiveParameters:
    mode: str
    intensity: float | None
    woa_a_base: float
    woa_a_scale: float
    woa_a_floor: float
    woa_a_cap: float
    woa_a_effective: float
    abc_phi_base: float
    abc_phi_scale: float
    abc_phi_effective: float
    abc_limit_base: int
    abc_limit_factor: float
    abc_limit_effective: int
    abc_guide_strength: float
    abc_vector_probability: float
    abc_vector_scale: float
    step_base: float
    step_scale: float
    step_effective: float
    momentum_factor: float
    momentum_scale: float

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_parameters(
    config: WOAABCConfig,
    decision: AdaptationDecision,
    iteration: int,
    dimension: int,
) -> EffectiveParameters:
    if decision.intensity is None:
        profile = config.base_profile
    else:
        profile = interpolate_profile(
            config.exploit_profile,
            config.explore_profile,
            decision.intensity,
        )

    if config.iterations == 1:
        woa_base = 2.0
    else:
        woa_base = 2.0 - 2.0 * iteration / (config.iterations - 1)
    woa_effective = min(
        profile.woa_a_cap,
        max(profile.woa_a_floor, woa_base * profile.woa_a_scale),
    )
    abc_limit_base = config.abc_limit
    if abc_limit_base is None:
        abc_limit_base = max(
            config.min_abc_limit,
            config.pop_size * dimension // config.abc_limit_divisor,
        )
    abc_limit_effective = max(
        config.min_abc_limit,
        int(round(abc_limit_base * profile.abc_limit_factor)),
    )
    if config.iterations == 1:
        step_base = config.step_final
    else:
        progress = iteration / (config.iterations - 1)
        step_base = config.step_initial * (config.step_final / config.step_initial) ** progress

    return EffectiveParameters(
        mode=decision.mode,
        intensity=decision.intensity,
        woa_a_base=float(woa_base),
        woa_a_scale=float(profile.woa_a_scale),
        woa_a_floor=float(profile.woa_a_floor),
        woa_a_cap=float(profile.woa_a_cap),
        woa_a_effective=float(woa_effective),
        abc_phi_base=1.0,
        abc_phi_scale=float(profile.abc_phi_scale),
        abc_phi_effective=float(profile.abc_phi_scale),
        abc_limit_base=int(abc_limit_base),
        abc_limit_factor=float(profile.abc_limit_factor),
        abc_limit_effective=int(abc_limit_effective),
        abc_guide_strength=float(config.abc_guide_strength * profile.abc_guide_scale),
        abc_vector_probability=float(config.abc_vector_probability),
        abc_vector_scale=float(config.abc_vector_scale * profile.step_scale),
        step_base=float(step_base),
        step_scale=float(profile.step_scale),
        step_effective=float(step_base * profile.step_scale),
        momentum_factor=float(config.momentum_factor),
        momentum_scale=float(profile.momentum_scale),
    )


__all__ = ["EffectiveParameters", "interpolate_profile", "resolve_parameters"]
