"""Control DTW de parámetros para las variantes WOA--ABC.

Este módulo no conoce CEC ni HRES2 y nunca modifica una población. Su única
responsabilidad es observar la curva del mejor costo y devolver el perfil de
parámetros que cada motor aplicará en la iteración siguiente.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isclose
from typing import Any

from dtw_stagnation import StagnationConfig, StagnationMonitor


M0_NO_DTW = "M0_no_dtw"
M1_LEGACY_FIRE_FULL = "M1_legacy_fire_full"
M2_FIRE_PARAMS_ONLY = "M2_fire_params_only"
M4_HYSTERESIS_WOA = "M4_hysteresis_woa"
M5_HYSTERESIS_ABC = "M5_hysteresis_abc"
M6_HYSTERESIS_WOA_ABC = "M6_hysteresis_woa_abc"
M8_FOUR_STATE_WOA_ABC = "M8_four_state_woa_abc"

VARIANT_NAMES: tuple[str, ...] = (
    M0_NO_DTW,
    M1_LEGACY_FIRE_FULL,
    M2_FIRE_PARAMS_ONLY,
    M4_HYSTERESIS_WOA,
    M5_HYSTERESIS_ABC,
    M6_HYSTERESIS_WOA_ABC,
    M8_FOUR_STATE_WOA_ABC,
)


class ControlMode(str, Enum):
    BASE = "base"
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    EXPLORE_LOW = "explore_low"
    EXPLORE_HIGH = "explore_high"
    EXPLOIT_LOW = "exploit_low"
    EXPLOIT_HIGH = "exploit_high"


@dataclass(frozen=True)
class ParameterProfile:
    """Valores efectivos que consumen los motores WOA--ABC."""

    name: str
    woa_a_scale: float = 1.0
    woa_a_floor: float = 0.0
    woa_a_cap: float = 2.0
    abc_phi_scale: float = 1.0
    abc_guide_strength: float = 0.20
    abc_limit_factor: float = 1.0
    step_scale: float = 1.0
    momentum_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlDecision:
    """Decisión para la próxima iteración; nunca contiene acciones poblacionales."""

    mode: ControlMode
    profile: ParameterProfile
    reason: str
    changed: bool = False
    transition: bool = False
    fire_accepted: bool = False


def validate_variant(variant: str) -> str:
    if variant not in VARIANT_NAMES:
        valid = ", ".join(VARIANT_NAMES)
        raise ValueError(f"variante desconocida: {variant!r}. Opciones: {valid}")
    return variant


def resolve_variants(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normaliza ``--variants`` y conserva el orden oficial."""

    if not values or any(value.lower() == "all" for value in values):
        return list(VARIANT_NAMES)
    aliases = {variant.split("_", 1)[0]: variant for variant in VARIANT_NAMES}
    requested = {
        aliases.get(value.upper(), value)
        for value in values
    }
    unknown = requested.difference(VARIANT_NAMES)
    if unknown:
        raise ValueError(f"variantes desconocidas: {', '.join(sorted(unknown))}")
    return [variant for variant in VARIANT_NAMES if variant in requested]


def _same_profile(left: ParameterProfile, right: ParameterProfile) -> bool:
    fields = (
        "woa_a_scale",
        "woa_a_floor",
        "woa_a_cap",
        "abc_phi_scale",
        "abc_guide_strength",
        "abc_limit_factor",
        "step_scale",
        "momentum_scale",
    )
    return all(isclose(getattr(left, field), getattr(right, field), rel_tol=1e-12, abs_tol=1e-12) for field in fields)


class AdaptiveParameterController:
    """Convierte el estado de :class:`StagnationMonitor` en parámetros.

    La decisión producida después de observar la iteración ``t`` se usa en
    ``t+1``. El controlador no recibe ni devuelve individuos.
    """

    def __init__(
        self,
        variant: str,
        stag_cfg: StagnationConfig,
        *,
        base_phi_scale: float = 1.0,
        base_guide_strength: float = 0.20,
        cooldown: int = 3,
        decay: float = 0.90,
        max_woa_scale: float = 2.50,
        max_phi_scale: float = 2.50,
    ) -> None:
        self.variant = validate_variant(variant)
        self.stag_cfg = stag_cfg
        self.base_phi_scale = float(base_phi_scale)
        self.base_guide_strength = float(base_guide_strength)
        self.cooldown = max(0, int(cooldown))
        self.decay = float(decay)
        self.max_woa_scale = float(max_woa_scale)
        self.max_phi_scale = float(max_phi_scale)
        self.monitor = None if self.variant == M0_NO_DTW else StagnationMonitor(stag_cfg)
        self.reset()

    def _base(self, name: str = "base") -> ParameterProfile:
        return ParameterProfile(
            name=name,
            abc_phi_scale=self.base_phi_scale,
            abc_guide_strength=self.base_guide_strength,
        )

    def _explore(self, name: str = "explore") -> ParameterProfile:
        return ParameterProfile(
            name=name,
            woa_a_scale=1.50,
            woa_a_floor=1.25,
            woa_a_cap=2.50,
            abc_phi_scale=self.base_phi_scale * 1.50,
            abc_guide_strength=max(0.0, self.base_guide_strength * 0.25),
            abc_limit_factor=0.50,
        )

    def _exploit(self, name: str = "exploit") -> ParameterProfile:
        return ParameterProfile(
            name=name,
            woa_a_scale=0.75,
            woa_a_floor=0.0,
            woa_a_cap=0.99,
            abc_phi_scale=self.base_phi_scale * 0.60,
            abc_guide_strength=self.base_guide_strength * 1.75,
            abc_limit_factor=1.25,
            step_scale=0.85,
        )

    def _explore_low(self) -> ParameterProfile:
        return ParameterProfile(
            name="explore_low",
            woa_a_scale=1.20,
            woa_a_floor=0.60,
            woa_a_cap=2.20,
            abc_phi_scale=self.base_phi_scale * 1.20,
            abc_guide_strength=self.base_guide_strength * 0.60,
            abc_limit_factor=0.75,
        )

    def _explore_high(self) -> ParameterProfile:
        return ParameterProfile(
            name="explore_high",
            woa_a_scale=1.75,
            woa_a_floor=1.50,
            woa_a_cap=2.75,
            abc_phi_scale=self.base_phi_scale * 1.75,
            abc_guide_strength=max(0.0, self.base_guide_strength * 0.10),
            abc_limit_factor=0.40,
        )

    def _exploit_low(self) -> ParameterProfile:
        return ParameterProfile(
            name="exploit_low",
            woa_a_scale=0.90,
            woa_a_cap=1.40,
            abc_phi_scale=self.base_phi_scale * 0.80,
            abc_guide_strength=self.base_guide_strength * 1.30,
            abc_limit_factor=1.10,
            step_scale=0.92,
        )

    def _exploit_high(self) -> ParameterProfile:
        return ParameterProfile(
            name="exploit_high",
            woa_a_scale=0.60,
            woa_a_cap=0.80,
            abc_phi_scale=self.base_phi_scale * 0.50,
            abc_guide_strength=self.base_guide_strength * 2.00,
            abc_limit_factor=1.50,
            step_scale=0.75,
        )

    def reset(self) -> None:
        if self.monitor is not None:
            self.monitor.reset()
        self.iteration = -1
        self.last_accepted_fire = -10**9
        self.fire_latched = False
        self.pulse_remaining = 0
        self.hysteresis_mode = ControlMode.BASE
        self.legacy_woa_scale = 1.0
        self.legacy_phi_scale = self.base_phi_scale
        self.legacy_guide = self.base_guide_strength
        self.legacy_limit_factor = 1.0
        self.current = ControlDecision(ControlMode.BASE, self._base(), "initial")
        self.fire_count = 0
        self.transition_count = 0
        self.parameter_update_count = 0

    def initial_decision(self) -> ControlDecision:
        return self.current

    def observe(self, best_cost: float) -> tuple[ControlDecision, dict[str, Any]]:
        """Observa un costo de minimización y decide parámetros para ``t+1``."""

        self.iteration += 1
        if self.monitor is None:
            status: dict[str, Any] = {
                "ready": False,
                "fire": False,
                "n": self.iteration + 1,
                "variant": self.variant,
            }
            decision = ControlDecision(ControlMode.BASE, self._base(), "dtw_disabled")
            self.current = decision
            return decision, status

        status = dict(self.monitor.update(-float(best_cost)))
        status["variant"] = self.variant
        previous = self.current
        decision = self._decide(status)
        changed = not _same_profile(previous.profile, decision.profile)
        transition = previous.mode != decision.mode
        decision = ControlDecision(
            mode=decision.mode,
            profile=decision.profile,
            reason=decision.reason,
            changed=changed,
            transition=transition,
            fire_accepted=decision.fire_accepted,
        )
        if decision.fire_accepted:
            self.fire_count += 1
        if transition:
            self.transition_count += 1
        if changed:
            self.parameter_update_count += 1
        self.current = decision
        return decision, status

    def _accept_fire(self, status: dict[str, Any]) -> bool:
        # Un `fire` es un evento de entrada al episodio de estancamiento, no
        # una acción que se repite en cada iteración mientras la alarma siga
        # alta. El latch se libera cuando el monitor devuelve `fire=False`.
        if not status.get("fire", False):
            self.fire_latched = False
            return False
        if self.fire_latched:
            return False
        if self.iteration - self.last_accepted_fire < self.cooldown:
            return False
        self.last_accepted_fire = self.iteration
        self.fire_latched = True
        return True

    def _decide(self, status: dict[str, Any]) -> ControlDecision:
        if self.variant == M1_LEGACY_FIRE_FULL:
            return self._legacy_fire(status)
        if self.variant == M2_FIRE_PARAMS_ONLY:
            return self._fire_pulse(status)
        if self.variant in (M4_HYSTERESIS_WOA, M5_HYSTERESIS_ABC, M6_HYSTERESIS_WOA_ABC):
            return self._hysteresis(status)
        if self.variant == M8_FOUR_STATE_WOA_ABC:
            return self._four_state(status)
        return ControlDecision(ControlMode.BASE, self._base(), "dtw_disabled")

    def _legacy_fire(self, status: dict[str, Any]) -> ControlDecision:
        accepted = self._accept_fire(status)
        if accepted:
            self.legacy_woa_scale = min(self.max_woa_scale, self.legacy_woa_scale * 1.35)
            self.legacy_phi_scale = min(self.max_phi_scale, self.legacy_phi_scale * 1.30)
            self.legacy_guide = max(0.02, self.legacy_guide * 0.60)
            self.legacy_limit_factor = max(0.20, self.legacy_limit_factor * 0.70)
        else:
            self.legacy_woa_scale = 1.0 + (self.legacy_woa_scale - 1.0) * self.decay
            self.legacy_phi_scale = self.base_phi_scale + (
                self.legacy_phi_scale - self.base_phi_scale
            ) * self.decay
            self.legacy_guide = self.base_guide_strength + (
                self.legacy_guide - self.base_guide_strength
            ) * self.decay
            self.legacy_limit_factor = 1.0 + (self.legacy_limit_factor - 1.0) * self.decay
        legacy_active = not (
            isclose(self.legacy_woa_scale, 1.0, abs_tol=1e-12)
            and isclose(self.legacy_phi_scale, self.base_phi_scale, abs_tol=1e-12)
            and isclose(self.legacy_guide, self.base_guide_strength, abs_tol=1e-12)
            and isclose(self.legacy_limit_factor, 1.0, abs_tol=1e-12)
        )
        profile = ParameterProfile(
            name="legacy_fire" if accepted else ("legacy_decay" if legacy_active else "base"),
            woa_a_scale=self.legacy_woa_scale,
            woa_a_floor=0.0,
            woa_a_cap=self.max_woa_scale if legacy_active else 2.0,
            abc_phi_scale=self.legacy_phi_scale,
            abc_guide_strength=self.legacy_guide,
            abc_limit_factor=self.legacy_limit_factor,
        )
        mode = ControlMode.EXPLORE if legacy_active else ControlMode.BASE
        return ControlDecision(mode, profile, "fire" if accepted else "decay", fire_accepted=accepted)

    def _fire_pulse(self, status: dict[str, Any]) -> ControlDecision:
        accepted = self._accept_fire(status)
        if accepted:
            self.pulse_remaining = max(1, self.cooldown)
        if self.pulse_remaining > 0:
            self.pulse_remaining -= 1
            explore = self._explore("fire_params_only")
            profile = ParameterProfile(
                name=explore.name,
                woa_a_scale=explore.woa_a_scale,
                woa_a_floor=explore.woa_a_floor,
                woa_a_cap=explore.woa_a_cap,
                abc_phi_scale=explore.abc_phi_scale,
                abc_guide_strength=self.base_guide_strength,
                abc_limit_factor=1.0,
            )
            return ControlDecision(
                ControlMode.EXPLORE,
                profile,
                "fire_pulse",
                fire_accepted=accepted,
            )
        return ControlDecision(ControlMode.BASE, self._base(), "pulse_inactive")

    def _masked_profile(self, target: ParameterProfile) -> ParameterProfile:
        base = self._base()
        if self.variant == M4_HYSTERESIS_WOA:
            return ParameterProfile(
                name=f"{target.name}_woa",
                woa_a_scale=target.woa_a_scale,
                woa_a_floor=target.woa_a_floor,
                woa_a_cap=target.woa_a_cap,
                abc_phi_scale=base.abc_phi_scale,
                abc_guide_strength=base.abc_guide_strength,
                abc_limit_factor=base.abc_limit_factor,
            )
        if self.variant == M5_HYSTERESIS_ABC:
            return ParameterProfile(
                name=f"{target.name}_abc",
                woa_a_scale=base.woa_a_scale,
                woa_a_floor=base.woa_a_floor,
                woa_a_cap=base.woa_a_cap,
                abc_phi_scale=target.abc_phi_scale,
                abc_guide_strength=target.abc_guide_strength,
                abc_limit_factor=target.abc_limit_factor,
            )
        return target

    def _hysteresis(self, status: dict[str, Any]) -> ControlDecision:
        if not status.get("ready", False):
            self.hysteresis_mode = ControlMode.BASE
            return ControlDecision(ControlMode.BASE, self._base(), "monitor_not_ready")
        delta = float(status.get("delta", 0.0))
        threshold = max(abs(float(status.get("theta_delta", 0.0))), 1e-12)
        if self.hysteresis_mode != ControlMode.EXPLORE and delta >= threshold:
            self.hysteresis_mode = ControlMode.EXPLORE
            reason = "enter_explore"
        elif self.hysteresis_mode == ControlMode.EXPLORE and delta <= 0.0:
            self.hysteresis_mode = ControlMode.EXPLOIT
            reason = "exit_explore"
        elif self.hysteresis_mode == ControlMode.BASE:
            self.hysteresis_mode = ControlMode.EXPLOIT
            reason = "ready_exploit"
        else:
            reason = "hold_mode"
        target = self._explore() if self.hysteresis_mode == ControlMode.EXPLORE else self._exploit()
        return ControlDecision(self.hysteresis_mode, self._masked_profile(target), reason)

    def _four_state(self, status: dict[str, Any]) -> ControlDecision:
        if not status.get("ready", False):
            return ControlDecision(ControlMode.BASE, self._base(), "monitor_not_ready")
        delta = float(status.get("delta", 0.0))
        threshold = max(abs(float(status.get("theta_delta", 0.0))), 1e-12)
        if delta > threshold:
            return ControlDecision(ControlMode.EXPLORE_HIGH, self._explore_high(), "delta_high_positive")
        if delta >= 0.0:
            return ControlDecision(ControlMode.EXPLORE_LOW, self._explore_low(), "delta_low_positive")
        if delta >= -threshold:
            return ControlDecision(ControlMode.EXPLOIT_LOW, self._exploit_low(), "delta_low_negative")
        return ControlDecision(ControlMode.EXPLOIT_HIGH, self._exploit_high(), "delta_high_negative")


__all__ = [
    "AdaptiveParameterController",
    "ControlDecision",
    "ControlMode",
    "ParameterProfile",
    "VARIANT_NAMES",
    "M0_NO_DTW",
    "M1_LEGACY_FIRE_FULL",
    "M2_FIRE_PARAMS_ONLY",
    "M4_HYSTERESIS_WOA",
    "M5_HYSTERESIS_ABC",
    "M6_HYSTERESIS_WOA_ABC",
    "M8_FOUR_STATE_WOA_ABC",
    "resolve_variants",
    "validate_variant",
]
