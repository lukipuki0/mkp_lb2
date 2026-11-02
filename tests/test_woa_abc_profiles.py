"""GWO-DE stagnation reaction in WOA-ABC: profiles, policies and timing.

Monitor outputs are scripted so each policy rule is checked on exact values.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from dtw_stagnation import StagnationConfig
from mkp_core.problem import MKPInstance
from woa_abc import HybridConfig, solve
from woa_abc.profiles import (
    EXPLOIT,
    EXPLORE,
    POLICIES,
    ProfileController,
    blend,
)
from woa_abc.solver import (
    PHASE_EMPLOYED,
    PHASE_INITIALIZATION,
    PHASE_ONLOOKERS,
    PHASE_SCOUTS,
    PHASE_WOA,
)


def _out(ready: bool = True, **values: float) -> dict:
    if not ready:
        return {"ready": False, "fire": False, "no_improve_len": values.get("no_improve_len", 0), "n": 1}
    out = {
        "ready": True,
        "fire": False,
        "D1_vs_ramp": 0.0,
        "D2_vs_const": 10.0,
        "delta": 0.0,
        "theta_c": 1.0,
        "theta_r": 1.0,
        "theta_delta": 1.0,
        "no_improve_len": 0,
        "trigger_streak": 0,
        "n": 4,
    }
    out.update(values)
    return out


class _Script:
    """Monitor factory whose monitors return the next scripted output."""

    def __init__(self, outputs: list[dict]) -> None:
        self.outputs = list(outputs)
        self.monitors: list[_ScriptedMonitor] = []

    def __call__(self, *args, **kwargs) -> "_ScriptedMonitor":
        monitor = _ScriptedMonitor(self)
        self.monitors.append(monitor)
        return monitor


class _ScriptedMonitor:
    def __init__(self, script: _Script) -> None:
        self.script = script
        self.seen: list[float] = []

    def update(self, best: float) -> dict:
        self.seen.append(best)
        return dict(self.script.outputs.pop(0))


def _cfg(plateau_max: int = 3) -> StagnationConfig:
    return StagnationConfig(window=4, band=1, plateau_max=plateau_max, patience=2)


def _controller(policy: str, outputs: list[dict], **kwargs) -> tuple[ProfileController, _Script]:
    script = _Script(outputs)
    controller = ProfileController(_cfg(), policy=policy, monitor_factory=script, **kwargs)
    return controller, script


def _observe_all(controller: ProfileController, count: int) -> list[dict]:
    return [controller.observe(float(index)) for index in range(count)]


class BinaryPolicyTests(unittest.TestCase):
    def test_binary_simple_enters_explore_on_d2_at_or_below_theta_c(self) -> None:
        controller, _ = _controller(
            "binary_simple",
            [
                _out(ready=False),
                _out(D2_vs_const=1.5, theta_c=1.0, fire=True),
                _out(D2_vs_const=1.0, theta_c=1.0),
            ],
        )
        outs = _observe_all(controller, 3)
        self.assertEqual([out["profile"] for out in outs], ["exploit", "exploit", "explore"])
        self.assertEqual([out["changed"] for out in outs], [False, False, True])
        self.assertEqual([out["intensity"] for out in outs], [None, None, None])
        self.assertIs(controller.profile, EXPLORE)

    def test_binary_complex_enters_explore_only_when_the_monitor_fires(self) -> None:
        controller, _ = _controller(
            "binary_complex",
            [
                _out(D2_vs_const=0.0, theta_c=1.0),
                _out(ready=False),
                _out(fire=True),
            ],
        )
        outs = _observe_all(controller, 3)
        self.assertEqual([out["profile"] for out in outs], ["exploit", "exploit", "explore"])
        self.assertEqual([out["changed"] for out in outs], [False, False, True])

    def test_explore_returns_to_exploit_when_no_improve_len_is_below_plateau_max(self) -> None:
        for policy in ("binary_simple", "binary_complex"):
            with self.subTest(policy=policy):
                controller, _ = _controller(
                    policy,
                    [
                        _out(fire=True, D2_vs_const=0.0),
                        _out(ready=False, no_improve_len=0),
                        _out(no_improve_len=2, fire=True, D2_vs_const=0.0),
                    ],
                )
                outs = _observe_all(controller, 3)
                self.assertEqual([out["profile"] for out in outs], ["explore", "explore", "exploit"])
                self.assertEqual([out["changed"] for out in outs], [True, False, True])
                self.assertIs(controller.profile, EXPLOIT)

    def test_explore_stays_while_no_improve_len_reaches_plateau_max(self) -> None:
        controller, _ = _controller(
            "binary_complex",
            [
                _out(fire=True),
                _out(no_improve_len=3),
                _out(no_improve_len=7, fire=False),
            ],
        )
        outs = _observe_all(controller, 3)
        self.assertEqual([out["profile"] for out in outs], ["explore", "explore", "explore"])
        self.assertEqual([out["changed"] for out in outs], [True, False, False])

    def test_every_binary_change_replaces_the_monitor_with_a_fresh_one(self) -> None:
        controller, script = _controller(
            "binary_complex",
            [
                _out(ready=False),
                _out(fire=True),
                _out(no_improve_len=5),
                _out(no_improve_len=0),
                _out(fire=False),
            ],
        )
        _observe_all(controller, 5)
        # Changes after observations 1 (exploit->explore) and 3 (explore->exploit).
        self.assertEqual(len(script.monitors), 3)
        self.assertEqual([monitor.seen for monitor in script.monitors], [[0.0, 1.0], [2.0, 3.0], [4.0]])

    def test_dtw_kirito_keeps_one_monitor_and_follows_the_trigger(self) -> None:
        for policy, trigger, quiet in (
            ("binary_complex", {"fire": True}, {"fire": False}),
            ("binary_simple", {"D2_vs_const": 0.5}, {"D2_vs_const": 2.0}),
        ):
            with self.subTest(policy=policy):
                controller, script = _controller(
                    policy,
                    [
                        _out(**trigger),
                        _out(**trigger, no_improve_len=0),
                        _out(**quiet, no_improve_len=9),
                        _out(ready=False),
                        _out(**trigger),
                    ],
                    dtw_kirito=True,
                )
                outs = _observe_all(controller, 5)
                self.assertEqual(
                    [out["profile"] for out in outs],
                    ["explore", "explore", "exploit", "exploit", "explore"],
                )
                self.assertEqual([out["changed"] for out in outs], [True, False, True, False, True])
                self.assertEqual(len(script.monitors), 1)
                self.assertEqual(script.monitors[0].seen, [0.0, 1.0, 2.0, 3.0, 4.0])


class ContinuousPolicyTests(unittest.TestCase):
    def test_continuous_simple_intensity_and_blend(self) -> None:
        controller, script = _controller(
            "continuous_simple",
            [
                _out(ready=False),
                _out(D2_vs_const=1.0, theta_c=2.0),
                _out(D2_vs_const=0.0, theta_c=2.0),
                _out(D2_vs_const=10.0, theta_c=2.0),
                _out(D2_vs_const=-3.0, theta_c=2.0),
            ],
        )
        outs = _observe_all(controller, 5)
        intensities = [out["intensity"] for out in outs]
        self.assertEqual(intensities[0], 0.0)
        self.assertAlmostEqual(intensities[1], 1.0 - 1.0 / 4.0)
        self.assertAlmostEqual(intensities[2], 1.0)
        self.assertAlmostEqual(intensities[3], 0.0)
        self.assertAlmostEqual(intensities[4], 1.0)
        self.assertEqual([out["profile"] for out in outs], ["exploit", "explore", "explore", "exploit", "explore"])
        self.assertEqual([out["changed"] for out in outs], [False, True, False, True, True])
        self.assertEqual(len(script.monitors), 1)

    def test_continuous_complex_intensity_is_the_centered_sigmoid(self) -> None:
        controller, script = _controller(
            "continuous_complex",
            [
                _out(delta=0.5, theta_delta=1.0),
                _out(delta=1.0, theta_delta=1.0),
                _out(delta=-4.0, theta_delta=2.0),
            ],
        )
        outs = _observe_all(controller, 3)
        self.assertAlmostEqual(outs[0]["intensity"], 0.5)
        self.assertEqual(outs[0]["profile"], "exploit")
        self.assertAlmostEqual(outs[1]["intensity"], 1.0 / (1.0 + math.exp(-2.5)))
        self.assertEqual(outs[1]["profile"], "explore")
        self.assertAlmostEqual(outs[2]["intensity"], 1.0 / (1.0 + math.exp(12.5)))
        self.assertEqual(outs[2]["profile"], "exploit")
        self.assertEqual(len(script.monitors), 1)

    def test_blend_is_linear_in_a_rho_and_g3_and_named_by_half(self) -> None:
        low = blend(EXPLOIT, EXPLORE, 0.25)
        self.assertEqual(low.name, "exploit")
        self.assertAlmostEqual(low.a, 0.75 * 0.5 + 0.25 * 2.0)
        self.assertAlmostEqual(low.rho, 0.75 * 1.0 + 0.25 * 2.0)
        self.assertAlmostEqual(low.G3, 0.5)
        self.assertEqual(blend(EXPLOIT, EXPLORE, 0.5).name, "exploit")
        self.assertEqual(blend(EXPLOIT, EXPLORE, 0.51).name, "explore")
        self.assertEqual(
            (blend(EXPLOIT, EXPLORE, 0.0).a, blend(EXPLOIT, EXPLORE, 0.0).rho),
            (EXPLOIT.a, EXPLOIT.rho),
        )
        self.assertEqual(
            (blend(EXPLOIT, EXPLORE, 1.0).a, blend(EXPLOIT, EXPLORE, 1.0).rho),
            (EXPLORE.a, EXPLORE.rho),
        )

    def test_intensity_is_zero_before_the_first_ready_observation(self) -> None:
        for policy in ("continuous_simple", "continuous_complex"):
            with self.subTest(policy=policy):
                controller, _ = _controller(policy, [_out(ready=False), _out(ready=False)])
                self.assertEqual(controller.intensity, 0.0)
                outs = _observe_all(controller, 2)
                self.assertEqual([out["intensity"] for out in outs], [0.0, 0.0])
                self.assertEqual([out["profile"] for out in outs], ["exploit", "exploit"])
                self.assertEqual((controller.profile.a, controller.profile.rho), (EXPLOIT.a, EXPLOIT.rho))


class DefaultsTests(unittest.TestCase):
    def test_provisional_profiles_and_policy_names(self) -> None:
        self.assertEqual((EXPLOIT.a, EXPLOIT.rho, EXPLOIT.G3), (0.5, 1.0, 0.5))
        self.assertEqual((EXPLORE.a, EXPLORE.rho, EXPLORE.G3), (2.0, 2.0, 0.5))
        self.assertEqual(
            POLICIES,
            ("binary_simple", "binary_complex", "continuous_simple", "continuous_complex"),
        )
        with self.assertRaises(ValueError):
            ProfileController(_cfg(), policy="nope")


def _varied_instance() -> MKPInstance:
    return MKPInstance(
        n=3,
        m=1,
        valor_optimo=9.0,
        p=np.array([5.0, 4.0, 3.0]),
        r=np.array([[2.0, 2.0, 2.0]]),
        b=np.array([4.0]),
    )


def _solve_scripted(outputs: list[dict], **overrides) -> tuple:
    script = _Script(outputs)
    values = dict(
        population_size=2,
        iterations=len(outputs),
        max_evaluations=10_000,
        scout_limit=1,
        stag_cfg=_cfg(),
    )
    values.update(overrides)
    with patch("woa_abc.profiles.StagnationMonitor", script):
        result = solve(_varied_instance(), HybridConfig(**values), seed=4)
    return result, script


class SolverTimingTests(unittest.TestCase):
    def test_change_decided_after_a_cycle_applies_from_the_next_cycle(self) -> None:
        result, script = _solve_scripted(
            [_out(ready=False), _out(fire=True), _out(no_improve_len=9), _out(no_improve_len=0), _out()]
        )
        self.assertEqual(result.profile_changes, [1, 3])
        self.assertEqual(
            [record.profile for record in result.history],
            ["exploit", "exploit", "explore", "explore", "exploit"],
        )
        self.assertEqual(
            [(record.a, record.rho) for record in result.history],
            [(0.5, 1.0), (0.5, 1.0), (2.0, 2.0), (2.0, 2.0), (0.5, 1.0)],
        )
        self.assertEqual([item["changed"] for item in result.diagnostics], [False, True, False, True, False])
        self.assertEqual([item["profile"] for item in result.diagnostics], [record.profile for record in result.history])
        self.assertEqual(result.diagnostics[1]["next_profile"], "explore")
        self.assertEqual(len(script.monitors), 3)
        # The monitor sees the archived best of each complete cycle, once.
        seen = [best for monitor in script.monitors for best in monitor.seen]
        self.assertEqual(seen, [record.best for record in result.history])

    def test_sequential_phase_order_and_archive_survive_profile_changes(self) -> None:
        result, _ = _solve_scripted([_out(fire=True), _out(no_improve_len=0)] * 3)
        self.assertEqual(len(result.profile_changes), 6)
        phases = result.phases_executed
        self.assertEqual(phases[0], PHASE_INITIALIZATION)
        self.assertEqual(phases[1:], [PHASE_WOA, PHASE_EMPLOYED, PHASE_ONLOOKERS, PHASE_SCOUTS] * 6)
        bests = [record.best for record in result.history]
        self.assertEqual(bests, sorted(bests))
        self.assertEqual(result.best_value, bests[-1])
        self.assertEqual(sum(result.evaluations_by_phase.values()), result.evaluations)

    def test_active_profile_sets_a_rho_and_g3_for_the_whole_cycle(self) -> None:
        import woa_abc.solver as hybrid

        seen: list[tuple[str, float]] = []
        real_l1 = hybrid._calcular_L1
        real_pair = hybrid._lb2_pair
        phase_now = {"phase": ""}

        def spy_pair(state, position, bits, phase):
            phase_now["phase"] = phase
            return real_pair(state, position, bits, phase)

        def spy_l1(position, v_max, G1, G2, G3):
            seen.append((phase_now["phase"], G3))
            return real_l1(position, v_max, G1, G2, G3)

        # Distinct G3 values make the profile visible inside LB2.
        explore = EXPLORE.__class__("explore", a=2.0, rho=2.0, G3=0.9)
        script = _Script([_out(fire=True), _out(no_improve_len=9)])

        def controller(cfg, **kwargs):
            return ProfileController(cfg, explore=explore, monitor_factory=script, **kwargs)

        config = HybridConfig(population_size=2, iterations=2, max_evaluations=10_000, scout_limit=100, stag_cfg=_cfg())
        with patch.object(hybrid, "ProfileController", controller), patch.object(
            hybrid, "_lb2_pair", spy_pair
        ), patch.object(hybrid, "_calcular_L1", spy_l1):
            result = solve(_varied_instance(), config, seed=4)

        self.assertEqual([record.G3 for record in result.history], [0.5, 0.9])
        per_cycle = 3 * 2  # WOA, employed and onlookers, one LB2 pair per source
        self.assertEqual(len(seen), 2 * per_cycle)
        self.assertEqual({g3 for _phase, g3 in seen[:per_cycle]}, {0.5})
        self.assertEqual({g3 for _phase, g3 in seen[per_cycle:]}, {0.9})
        self.assertEqual(
            [phase for phase, _g3 in seen[per_cycle:]],
            [PHASE_WOA] * 2 + [PHASE_EMPLOYED] * 2 + [PHASE_ONLOOKERS] * 2,
        )

    def test_partial_cycle_is_not_observed(self) -> None:
        script = _Script([_out(ready=False)] * 5)
        config = HybridConfig(
            population_size=2,
            iterations=5,
            max_evaluations=2 + 2 * 7 + 4,
            scout_limit=100,
            stag_cfg=_cfg(),
        )
        with patch("woa_abc.profiles.StagnationMonitor", script):
            result = solve(_varied_instance(), config, seed=4)
        self.assertEqual(result.cycles_completed, 1)
        self.assertFalse(result.history[-1].completeness)
        self.assertEqual([item["cycle"] for item in result.diagnostics], [0])
        self.assertEqual(sum(len(monitor.seen) for monitor in script.monitors), 1)

    def test_continuous_intensity_is_recorded_with_a_one_cycle_lag(self) -> None:
        result, script = _solve_scripted(
            [_out(ready=False), _out(D2_vs_const=0.0), _out(D2_vs_const=1.0, theta_c=2.0), _out()],
            policy="continuous_simple",
        )
        intensities = [record.intensity for record in result.history]
        self.assertEqual(intensities[:2], [0.0, 0.0])
        self.assertAlmostEqual(intensities[2], 1.0)
        self.assertAlmostEqual(intensities[3], 0.75)
        self.assertEqual([record.profile for record in result.history], ["exploit", "exploit", "explore", "explore"])
        self.assertAlmostEqual(result.history[3].a, 0.25 * 0.5 + 0.75 * 2.0)
        self.assertAlmostEqual(result.history[3].rho, 0.25 * 1.0 + 0.75 * 2.0)
        self.assertEqual(result.profile_changes, [1, 3])
        self.assertEqual(len(script.monitors), 1)

    def test_no_stagnation_keeps_exploit_and_builds_no_monitor(self) -> None:
        result, script = _solve_scripted([_out(fire=True)] * 4, use_stagnation=False)
        self.assertEqual(script.monitors, [])
        self.assertEqual([record.profile for record in result.history], ["exploit"] * 4)
        self.assertEqual(result.profile_changes, [])
        self.assertEqual(result.diagnostics, [])


if __name__ == "__main__":
    unittest.main()
