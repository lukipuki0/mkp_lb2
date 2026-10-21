from __future__ import annotations

import unittest

import numpy as np

from woa_abc_version_vieja.config import DTWConfig, StrategyConfig, WOAABCConfig
from woa_abc_version_vieja.core.woa_abc import WOAABCOptimizer
from woa_abc_version_vieja.domains.base import Problem
from woa_abc_version_vieja.dtw.monitor import StagnationMonitor, ddtw_distance, dtw_distance
from woa_abc_version_vieja.dtw.strategies import (
    AdaptationController,
    M1_FIRE_D2,
    M2_FIRE_3COND,
    M4_D2_CONTINUOUS,
    M5_SIGMOID_DELTA,
    VARIANT_NAMES,
)


class TestDTW(unittest.TestCase):
    def test_distances_for_identical_and_shifted_curves(self) -> None:
        source = np.asarray([1.0, 2.0, 3.0, 4.0])
        shifted = source + 10.0
        self.assertEqual(dtw_distance(source, source, window=1), 0.0)
        self.assertGreater(dtw_distance(source, shifted, window=1), 0.0)
        self.assertEqual(ddtw_distance(source, shifted, window=1), 0.0)

    def test_flat_sequence_activates_three_condition_fire(self) -> None:
        monitor = StagnationMonitor(
            DTWConfig(
                window=5,
                band=1,
                min_slope=2.0,
                plateau_max=2,
                patience=2,
                use_ddtw=True,
                adapt_thresholds=False,
            )
        )
        states = [monitor.update(10.0) for _ in range(13)]
        self.assertFalse(states[3]["ready"])
        self.assertFalse(states[4]["fire"])
        self.assertEqual(states[4]["trigger_streak"], 1)
        self.assertTrue(states[5]["cond_plateau"])
        self.assertTrue(states[5]["cond_constant"])
        self.assertTrue(states[5]["cond_ramp"])
        self.assertTrue(states[5]["fire"])
        self.assertEqual(states[5]["n"], 6)
        self.assertEqual(states[6]["n"], 1)
        self.assertFalse(states[6]["ready"])
        self.assertFalse(states[6]["fire"])
        self.assertTrue(states[11]["fire"])
        self.assertFalse(states[12]["fire"])
        self.assertEqual(
            [index for index, state in enumerate(states) if state["fire"]],
            [5, 11],
        )


class TestStagnationFlow(unittest.TestCase):
    def _trace(self, values: list[float]) -> list[dict[str, object]]:
        monitor = StagnationMonitor(DTWConfig())
        controllers = {
            variant: AdaptationController(variant, StrategyConfig())
            for variant in VARIANT_NAMES
        }
        rows: list[dict[str, object]] = []
        for value in values:
            status = monitor.update(value)
            rows.append({
                "status": status,
                "decisions": {
                    variant: controller.decide(status)
                    for variant, controller in controllers.items()
                },
            })
        return rows

    def test_perfect_improvement_does_not_trigger_exploration(self) -> None:
        rows = self._trace([100.0 + 2.0 * index for index in range(60)])
        self.assertTrue(all(not row["status"]["fire"] for row in rows))
        result = rows[-1]
        decisions = result["decisions"]
        self.assertFalse(result["status"]["fire"])
        self.assertEqual(decisions[M1_FIRE_D2].mode, "exploit")
        self.assertEqual(decisions[M2_FIRE_3COND].mode, "exploit")
        self.assertLessEqual(float(decisions[M4_D2_CONTINUOUS].intensity), 0.5)
        self.assertEqual(decisions[M4_D2_CONTINUOUS].mode, "exploit")
        self.assertLess(float(decisions[M5_SIGMOID_DELTA].intensity), 0.5)
        self.assertEqual(decisions["M6_hysteresis_woa_abc"].mode, "exploit")
        self.assertEqual(decisions["M8_four_state_woa_abc"].mode, "exploit_high")

    def test_sustained_plateau_still_triggers_exploration(self) -> None:
        rows = self._trace([100.0] * 60)
        fires = [index for index, row in enumerate(rows) if row["status"]["fire"]]
        self.assertEqual(fires, [20, 41])
        for index in fires:
            decisions = rows[index]["decisions"]
            self.assertEqual(decisions[M1_FIRE_D2].mode, "explore")
            self.assertTrue(decisions[M1_FIRE_D2].raw_fire)
            self.assertEqual(decisions[M2_FIRE_3COND].mode, "explore")
            self.assertTrue(decisions[M2_FIRE_3COND].raw_fire)
            self.assertGreater(float(decisions[M4_D2_CONTINUOUS].intensity), 0.5)
            self.assertGreater(float(decisions[M5_SIGMOID_DELTA].intensity), 0.5)
            self.assertEqual(decisions["M6_hysteresis_woa_abc"].mode, "explore")
            self.assertEqual(decisions["M8_four_state_woa_abc"].mode, "explore_high")
        clear = [
            row for row in rows
            if row["status"].get("ready") and not row["status"]["fire"]
        ]
        self.assertTrue(clear)
        for row in clear:
            self.assertEqual(row["decisions"][M1_FIRE_D2].mode, "exploit")
            self.assertFalse(row["decisions"][M1_FIRE_D2].raw_fire)
            self.assertFalse(row["decisions"][M4_D2_CONTINUOUS].raw_fire)
        self.assertFalse(rows[-1]["status"]["fire"])
        self.assertEqual(rows[-1]["status"]["n"], 18)


class TestStrategies(unittest.TestCase):
    def setUp(self) -> None:
        self.status = {
            "ready": True,
            "fire": True,
            "D2_vs_const": 1.0,
            "theta_c": 2.0,
            "delta": 4.0,
            "theta_delta": 2.0,
        }
        self.config = StrategyConfig()

    def test_all_seven_variants_are_present(self) -> None:
        self.assertEqual(len(VARIANT_NAMES), 7)

    def test_discrete_reference_policies(self) -> None:
        self.assertEqual(
            AdaptationController(M1_FIRE_D2, self.config).decide(self.status).mode,
            "explore",
        )
        self.assertEqual(
            AdaptationController(M2_FIRE_3COND, self.config).decide(self.status).mode,
            "explore",
        )

    def test_continuous_reference_policies_return_valid_intensity(self) -> None:
        for variant in (M4_D2_CONTINUOUS, M5_SIGMOID_DELTA):
            decision = AdaptationController(variant, self.config).decide(self.status)
            self.assertIsNotNone(decision.intensity)
            self.assertGreaterEqual(float(decision.intensity), 0.0)
            self.assertLessEqual(float(decision.intensity), 1.0)


class TestOptimizer(unittest.TestCase):
    @staticmethod
    def problem() -> Problem:
        return Problem(
            name="sphere",
            dimension=3,
            lower=np.full(3, -5.0),
            upper=np.full(3, 5.0),
            objective=lambda x: float(np.sum(np.square(x))),
            optimum=0.0,
        )

    def test_reproducible_bounded_and_monotone(self) -> None:
        config = WOAABCConfig(
            pop_size=6,
            iterations=25,
            seed=17,
            variant=M4_D2_CONTINUOUS,
            dtw=DTWConfig(window=5, band=1, min_slope=0.0),
        )
        first = WOAABCOptimizer(self.problem(), config).run()
        second = WOAABCOptimizer(self.problem(), config).run()
        self.assertEqual(first.best_cost, second.best_cost)
        self.assertEqual(first.convergence_history, second.convergence_history)
        self.assertTrue(np.all(np.diff(first.convergence_history) <= 1e-12))
        self.assertTrue(np.all(np.asarray(first.best_position) >= -5.0))
        self.assertTrue(np.all(np.asarray(first.best_position) <= 5.0))
        self.assertEqual(len(first.parameter_history), config.iterations)
        self.assertEqual(len(first.dtw_history), config.iterations)


if __name__ == "__main__":
    unittest.main()
