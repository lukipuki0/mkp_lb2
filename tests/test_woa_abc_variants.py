from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuous_benchmark.funciones_cec2022 import ContinuousFunction
from dtw_stagnation import StagnationConfig
from woa_abc.adaptive import (
    AdaptiveParameterController,
    M1_LEGACY_FIRE_FULL,
    M2_FIRE_PARAMS_ONLY,
    M4_HYSTERESIS_WOA,
    M5_HYSTERESIS_ABC,
    M6_HYSTERESIS_WOA_ABC,
    M8_FOUR_STATE_WOA_ABC,
    VARIANT_NAMES,
    resolve_variants,
)
from woa_abc.cooperativo_cec_dtw import CooperativeCECParams, ejecutar_cec_cooperativo
from woa_abc.cooperativo_hres2_dtw import CooperativeHRES2Params, ejecutar_hres2_cooperativo


EXPECTED_VARIANTS = (
    "M0_no_dtw",
    "M1_legacy_fire_full",
    "M2_fire_params_only",
    "M4_hysteresis_woa",
    "M5_hysteresis_abc",
    "M6_hysteresis_woa_abc",
    "M8_four_state_woa_abc",
)


def _controller(variant: str) -> AdaptiveParameterController:
    return AdaptiveParameterController(
        variant,
        StagnationConfig(window=2, plateau_max=1, patience=1, adapt_thresholds=False),
    )


def test_exact_variant_catalog() -> None:
    assert VARIANT_NAMES == EXPECTED_VARIANTS
    assert resolve_variants(["M0", "M6"]) == [
        "M0_no_dtw",
        "M6_hysteresis_woa_abc",
    ]


def test_fire_variants_only_return_parameter_profiles() -> None:
    status = {"ready": True, "fire": True, "delta": 2.0, "theta_delta": 1.0}
    legacy = _controller(M1_LEGACY_FIRE_FULL)._legacy_fire(status)
    pulse = _controller(M2_FIRE_PARAMS_ONLY)._fire_pulse(status)

    assert legacy.profile.woa_a_scale > 1.0
    assert legacy.profile.abc_phi_scale > 1.0
    assert legacy.profile.abc_guide_strength < 0.20
    assert legacy.profile.abc_limit_factor < 1.0
    assert pulse.profile.woa_a_scale > 1.0
    assert pulse.profile.abc_phi_scale > 1.0
    assert pulse.profile.abc_guide_strength == 0.20
    assert pulse.profile.abc_limit_factor == 1.0
    assert not hasattr(legacy, "rescue")
    assert not hasattr(pulse, "rescue")


def test_hysteresis_actuators_are_isolated() -> None:
    status = {"ready": True, "fire": False, "delta": 2.0, "theta_delta": 1.0}
    woa = _controller(M4_HYSTERESIS_WOA)._hysteresis(status).profile
    abc = _controller(M5_HYSTERESIS_ABC)._hysteresis(status).profile
    both = _controller(M6_HYSTERESIS_WOA_ABC)._hysteresis(status).profile

    assert woa.woa_a_scale != 1.0
    assert woa.abc_phi_scale == 1.0
    assert woa.abc_guide_strength == 0.20
    assert abc.woa_a_scale == 1.0
    assert abc.abc_phi_scale != 1.0
    assert both.woa_a_scale != 1.0
    assert both.abc_phi_scale != 1.0


def test_four_state_policy_covers_all_delta_regions() -> None:
    controller = _controller(M8_FOUR_STATE_WOA_ABC)
    modes = []
    for delta in (2.0, 0.5, -0.5, -2.0):
        decision = controller._four_state(
            {"ready": True, "fire": False, "delta": delta, "theta_delta": 1.0}
        )
        modes.append(decision.mode.value)
    assert modes == ["explore_high", "explore_low", "exploit_low", "exploit_high"]


def test_cec_engine_is_reproducible_and_monotonic() -> None:
    function = ContinuousFunction(
        name="sphere",
        func=lambda x: float(np.sum(x * x)),
        lb=-5.0,
        ub=5.0,
        optimum=0.0,
        n_dim=4,
    )
    params = CooperativeCECParams(
        pop_size=6,
        iterations=8,
        variant="M6_hysteresis_woa_abc",
        seed=17,
        stag_cfg=StagnationConfig(window=3, plateau_max=1, patience=1),
    )
    first = ejecutar_cec_cooperativo(function, params, verbose=False)
    second = ejecutar_cec_cooperativo(function, params, verbose=False)
    assert first.mejor_valor_global == second.mejor_valor_global
    assert first.epochs[0].historial == second.epochs[0].historial
    assert all(
        current <= previous
        for previous, current in zip(first.epochs[0].historial, first.epochs[0].historial[1:])
    )


@dataclass
class _FakeHRES2:
    n_dim: int = 4

    def __post_init__(self) -> None:
        self.name = "fake_hres2"
        self.lb_vector = np.array([0.0, 10.0, 0.0, 0.0])
        self.ub_vector = np.array([200.0, 20.0, 50.0, 2.0])
        self.func = lambda x: float(np.sum((np.asarray(x) - np.array([50.0, 15.0, 20.0, 1.0])) ** 2))

    def get_info(self, x: np.ndarray) -> dict:
        inside = np.all(x >= self.lb_vector) and np.all(x <= self.ub_vector)
        return {"feasible": bool(inside), "lcoe_cny_per_kwh": self.func(x)}


def test_hres2_engine_uses_vector_bounds() -> None:
    function = _FakeHRES2()
    result = ejecutar_hres2_cooperativo(
        function,
        CooperativeHRES2Params(
            pop_size=6,
            iterations=8,
            variant="M8_four_state_woa_abc",
            seed=23,
            stag_cfg=StagnationConfig(window=3, plateau_max=1, patience=1),
        ),
        verbose=False,
    )
    solution = np.asarray(result.mejor_sol_global)
    assert np.all(solution >= function.lb_vector)
    assert np.all(solution <= function.ub_vector)
    assert result.mejor_info_global["feasible"] is True
