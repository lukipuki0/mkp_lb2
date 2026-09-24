from __future__ import annotations

import unittest

import numpy as np

from new_woa_abc.config import DTWConfig, StrategyConfig, WOAABCConfig
from new_woa_abc.core.woa_abc import WOAABCOptimizer
from new_woa_abc.domains.base import Problem
from new_woa_abc.dtw.monitor import StagnationMonitor, ddtw_distance, dtw_distance
from new_woa_abc.dtw.strategies import (
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
        states = [monitor.update(10.0) for _ in range(7)]
        self.assertFalse(states[3]["ready"])
        self.assertTrue(states[-1]["cond_plateau"])
        self.assertTrue(states[-1]["cond_constant"])
        self.assertTrue(states[-1]["cond_ramp"])
        self.assertTrue(states[-1]["fire"])


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
