from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from new_woa_abc.config import DTWConfig, WOAABCConfig
from new_woa_abc.core.binary_woa_abc import (
    BinaryWOAABCConfig,
    BinaryWOAABCOptimizer,
    lb2_probabilities,
)
from new_woa_abc.domains.mkp import MKPInstance, parse_mkp_file
from new_woa_abc.dtw.strategies import M4_D2_CONTINUOUS, VARIANT_NAMES
from new_woa_abc.runners.run_mkp import resolve_instance_indices


class TestMKPDomain(unittest.TestCase):
    @staticmethod
    def instance() -> MKPInstance:
        return MKPInstance(
            family="toy",
            index=0,
            profits=np.asarray([10, 8, 7, 6, 4], dtype=float),
            weights=np.asarray([
                [6, 5, 4, 3, 2],
                [3, 5, 4, 4, 2],
            ], dtype=float),
            capacities=np.asarray([10, 9], dtype=float),
            best_known=18.0,
        )

    def test_repair_is_binary_feasible_and_maximal(self) -> None:
        instance = self.instance()
        solution, value = instance.repair(np.ones(instance.dimension, dtype=np.int8))
        self.assertTrue(instance.is_feasible(solution))
        self.assertEqual(value, instance.evaluate(solution))
        for index in np.flatnonzero(1 - solution):
            candidate = solution.copy()
            candidate[index] = 1
            self.assertFalse(instance.is_feasible(candidate))

    def test_real_chu_beasley_file_and_known_optimum(self) -> None:
        root = Path(__file__).resolve().parents[2]
        instances = parse_mkp_file(root / "instancias" / "mknapcb1.txt")
        self.assertEqual(len(instances), 30)
        self.assertEqual(instances[0].dimension, 100)
        self.assertEqual(instances[0].constraints, 5)
        self.assertEqual(instances[0].best_known, 24381.0)

    def test_three_exchange_closes_known_instance_zero_gap(self) -> None:
        root = Path(__file__).resolve().parents[2]
        instance = parse_mkp_file(root / "instancias" / "mknapcb1.txt")[0]
        incumbent = np.zeros(instance.dimension, dtype=np.int8)
        incumbent[[
            1, 3, 6, 7, 8, 17, 18, 23, 26, 28, 29, 31, 34, 43, 49,
            56, 61, 62, 65, 68, 70, 73, 76, 78, 84, 85, 91, 92, 98,
        ]] = 1
        improved, value, evaluations = instance.improve(
            incumbent,
            np.random.default_rng(7),
            passes=1,
            ejection_candidates=15,
            exchange_depth=3,
        )
        self.assertTrue(instance.is_feasible(improved))
        self.assertEqual(value, instance.best_known)
        self.assertGreater(evaluations, 1)

    def test_instance_range_parser(self) -> None:
        self.assertEqual(resolve_instance_indices(["0", "2-4"], 6), [0, 2, 3, 4])
        self.assertEqual(resolve_instance_indices(["all"], 3), [0, 1, 2])


class TestBinaryWOAABC(unittest.TestCase):
    def test_lb2_probabilities_are_bounded(self) -> None:
        signal = np.linspace(-6.0, 6.0, 25)
        l1, l2 = lb2_probabilities(signal, 6.0, 0.8, 2.0, 0.2)
        self.assertTrue(np.all((0.0 <= l1) & (l1 <= 1.0)))
        self.assertTrue(np.all((0.0 <= l2) & (l2 <= 1.0)))

    def test_reproducible_feasible_and_monotone_maximization(self) -> None:
        instance = TestMKPDomain.instance()
        algorithm = WOAABCConfig(
            pop_size=6,
            iterations=20,
            seed=19,
            variant=M4_D2_CONTINUOUS,
            dtw=DTWConfig(window=5, band=1, min_slope=0.0),
        )
        config = BinaryWOAABCConfig(
            algorithm=algorithm,
            local_search_interval=4,
            local_search_passes=1,
            local_search_ejections=3,
            early_stop_at_best_known=False,
        )
        first = BinaryWOAABCOptimizer(instance, config).run()
        second = BinaryWOAABCOptimizer(instance, config).run()
        self.assertEqual(first.best_profit, second.best_profit)
        self.assertEqual(first.best_solution, second.best_solution)
        self.assertEqual(first.convergence_history, second.convergence_history)
        self.assertTrue(instance.is_feasible(first.best_solution))
        self.assertTrue(np.all(np.diff(first.convergence_history) >= -1e-12))
        self.assertEqual(len(first.parameter_history), algorithm.iterations)
        self.assertEqual(len(first.dtw_history), algorithm.iterations)
        self.assertEqual(first.parameter_history[0]["abc_limit_base"], algorithm.pop_size)

    def test_all_seven_variants_run_on_binary_engine(self) -> None:
        instance = TestMKPDomain.instance()
        for variant in VARIANT_NAMES:
            with self.subTest(variant=variant):
                algorithm = WOAABCConfig(
                    pop_size=4,
                    iterations=7,
                    seed=5,
                    variant=variant,
                    dtw=DTWConfig(window=3, band=1, min_slope=0.0),
                )
                result = BinaryWOAABCOptimizer(
                    instance,
                    BinaryWOAABCConfig(
                        algorithm=algorithm,
                        local_search_interval=0,
                        early_stop_at_best_known=False,
                    ),
                ).run()
                self.assertTrue(instance.is_feasible(result.best_solution))
                self.assertEqual(len(result.convergence_history), 7)


if __name__ == "__main__":
    unittest.main()
