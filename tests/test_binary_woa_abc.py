"""Focused regression tests for the shared-population WOA/ABC baseline."""

from __future__ import annotations

import inspect
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

import binary_woa_abc as hybrid
from binary_woa_abc import (
    PHASE_EMPLOYED,
    PHASE_INITIALIZATION,
    PHASE_ONLOOKERS,
    PHASE_SCOUTS,
    PHASE_WOA,
    PHASES,
    TERMINATION_EVALUATION_BUDGET,
    TERMINATION_ITERATIONS,
    HybridConfig,
    HybridResult,
    _RunState,
    _a2,
    _clip_coordinates,
    _commit_abc,
    _commit_woa,
    _control_parameters,
    _freeze_leader,
    _prefer_l1,
    _replace_scout,
    _roulette_probabilities,
    _woa_destination,
    main,
    solve,
)
from dtw_stagnation import StagnationConfig
from mkp_core.problem import MKPInstance
from mkp_core.repair import reparar_solucion


def _instance(
    profits: list[float],
    weights: list[list[float]],
    capacities: list[float],
    optimum: float = 0.0,
) -> MKPInstance:
    profit_row = np.array(profits, dtype=float)
    weight_matrix = np.array(weights, dtype=float)
    capacity_row = np.array(capacities, dtype=float)
    return MKPInstance(
        n=int(profit_row.size),
        m=int(capacity_row.size),
        valor_optimo=optimum,
        p=profit_row,
        r=weight_matrix,
        b=capacity_row,
    )


def _constant_instance(optimum: float = 6.0) -> MKPInstance:
    """Every repair fills the knapsack, so fitness never changes."""

    return _instance([1.0, 2.0, 3.0], [[1.0, 1.0, 1.0]], [10.0], optimum)


def _varied_instance(optimum: float = 9.0) -> MKPInstance:
    """Capacity admits two items, so repaired subsets have different values."""

    return _instance([5.0, 4.0, 3.0], [[2.0, 2.0, 2.0]], [4.0], optimum)


def _config(**overrides: object) -> HybridConfig:
    values: dict[str, object] = {
        "population_size": 2,
        "iterations": 2,
        "max_evaluations": 10_000,
        "scout_limit": 100,
        "v_max": 6.0,
        "b": 1.0,
        "G1": 0.5,
        "G2": 0.5,
        "G3": 0.5,
    }
    values.update(overrides)
    return HybridConfig(**values)  # type: ignore[arg-type]


def _assert_accounting(test: unittest.TestCase, result: HybridResult) -> None:
    test.assertEqual(set(result.evaluations_by_phase), set(PHASES))
    test.assertEqual(sum(result.evaluations_by_phase.values()), result.evaluations)
    test.assertEqual(result.archive_size, result.evaluations)
    test.assertEqual(
        result.cycles_completed,
        sum(1 for record in result.history if record.completeness),
    )
    test.assertLessEqual(result.cycles_completed, len(result.history))
    if result.history:
        incomplete = [record.completeness for record in result.history]
        test.assertEqual(incomplete, [True] * result.cycles_completed + [False] * (
            len(result.history) - result.cycles_completed
        ))
        test.assertEqual(result.history[-1].best, result.best_value)
        test.assertEqual(result.history[-1].evaluations, result.evaluations)


class TestValidation(unittest.TestCase):
    def test_defaults_match_the_legacy_baseline(self) -> None:
        config = HybridConfig()

        self.assertEqual(config.population_size, 30)
        self.assertEqual(config.iterations, 300)
        self.assertIsNone(config.max_evaluations)
        self.assertEqual(config.v_max, 6.0)
        self.assertEqual(config.b, 1.0)
        self.assertEqual((config.G1, config.G2, config.G3), (0.5, 0.5, 0.5))
        self.assertIsNone(config.scout_limit)
        self.assertTrue(config.use_stagnation)
        self.assertIsNone(config.stag_cfg)

    def test_config_rejects_malformed_numbers(self) -> None:
        rejected = (
            {"population_size": 1},
            {"population_size": True},
            {"population_size": 2.5},
            {"iterations": 1},
            {"population_size": 2, "iterations": 2, "max_evaluations": 1},
            {"max_evaluations": True},
            {"v_max": 0.0},
            {"v_max": -1.0},
            {"v_max": float("nan")},
            {"v_max": float("inf")},
            {"b": float("nan")},
            {"G1": float("inf")},
            {"G2": float("-inf")},
            {"scout_limit": 0},
            {"scout_limit": -3},
            {"scout_limit": True},
        )
        for overrides in rejected:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    _config(**overrides)

    def test_seed_must_be_a_nonnegative_integer(self) -> None:
        inst = _constant_instance()
        config = _config()
        for seed in (-1, 1.5, True, None):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    solve(inst, config, seed=seed)  # type: ignore[arg-type]

    def test_malformed_instance_is_rejected_before_repair(self) -> None:
        config = _config(max_evaluations=2)
        valid = _constant_instance()

        zero_weight = _constant_instance()
        zero_weight.r = np.zeros_like(zero_weight.r)
        negative_profit = _constant_instance()
        negative_profit.p = np.array([1.0, -1.0, 3.0])
        negative_capacity = _constant_instance()
        negative_capacity.b = np.array([-1.0])
        nan_weight = _constant_instance()
        nan_weight.r = np.array([[1.0, float("nan"), 1.0]])
        bad_shape = _constant_instance()
        bad_shape.p = np.array([1.0, 2.0])
        negative_reference = _constant_instance()
        negative_reference.valor_optimo = -5.0

        cases = (
            zero_weight,
            negative_profit,
            negative_capacity,
            nan_weight,
            bad_shape,
            negative_reference,
            object(),
        )
        for inst in cases:
            with self.subTest(inst=getattr(inst, "p", type(inst))):
                with self.assertRaises(ValueError):
                    solve(inst, config, seed=0)  # type: ignore[arg-type]

        self.assertEqual(valid.n, 3)

    def test_solve_does_not_fill_omitted_config_defaults(self) -> None:
        config = HybridConfig(population_size=2, iterations=2, scout_limit=1)
        before = (config.max_evaluations, config.scout_limit)

        result = solve(_constant_instance(), config, seed=0)

        self.assertEqual((config.max_evaluations, config.scout_limit), before)
        self.assertEqual(result.scout_limit, 1)
        self.assertEqual(result.cycles_completed, 2)
        self.assertEqual(result.termination_reason, TERMINATION_ITERATIONS)
        self.assertEqual(result.evaluations, 2 * (1 + 7 * 2))

    def test_omitted_scout_limit_uses_population_times_items(self) -> None:
        inst = _constant_instance()
        result = solve(inst, _config(scout_limit=None, max_evaluations=4), seed=1)

        self.assertEqual(result.scout_limit, 2 * inst.n)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_repeats_and_global_rng_streams_stay_put(self) -> None:
        inst = _varied_instance(optimum=100.0)
        config = _config(iterations=3, scout_limit=1, max_evaluations=200)
        random.seed(99)
        np.random.seed(99)
        expected_python = random.random()
        expected_numpy = float(np.random.random())
        random.seed(99)
        np.random.seed(99)

        first = solve(inst, config, seed=7)
        second = solve(inst, config, seed=7)

        self.assertEqual(asdict(first), asdict(second))
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(float(np.random.random()), expected_numpy)
        self.assertNotIn("import mh", inspect.getsource(hybrid))
        self.assertNotIn("binarizar_posicion", inspect.getsource(hybrid))

    def test_rejected_input_does_not_consume_global_rng(self) -> None:
        random.seed(3)
        np.random.seed(3)
        expected_python = random.random()
        expected_numpy = float(np.random.random())
        random.seed(3)
        np.random.seed(3)
        broken = _constant_instance()
        broken.p = np.array([-1.0, 1.0, 1.0])

        with self.assertRaises(ValueError):
            solve(broken, _config(), seed=1)

        self.assertEqual(random.random(), expected_python)
        self.assertEqual(float(np.random.random()), expected_numpy)


class TestSearchContract(unittest.TestCase):
    def test_output_is_binary_feasible_and_archive_survives_scouts(self) -> None:
        inst = _varied_instance(optimum=9.0)
        seen: list[tuple[list[int], float]] = []

        def record(solution: list[int], problem: MKPInstance) -> tuple[list[int], float]:
            repaired, value = reparar_solucion(solution, problem)
            seen.append((list(repaired), float(value)))
            return repaired, value

        with patch.object(hybrid, "reparar_solucion", record):
            result = solve(inst, _config(iterations=2, scout_limit=1), seed=4)

        self.assertEqual(len(result.best_solution), inst.n)
        self.assertTrue(all(type(bit) is int and bit in (0, 1) for bit in result.best_solution))
        self.assertTrue(inst.es_factible(result.best_solution))
        self.assertEqual(inst.evaluar(result.best_solution), result.best_value)
        self.assertEqual(result.best_value, max(value for _, value in seen))
        self.assertIn(
            result.best_solution,
            [solution for solution, value in seen if value == result.best_value],
        )
        self.assertGreater(result.evaluations_by_phase[PHASE_SCOUTS], 0)
        self.assertGreater(result.best_value, min(value for _, value in seen))
        self.assertEqual(len(seen), result.evaluations)

    def test_solver_does_not_revisit_gap_or_feasibility_helpers(self) -> None:
        inst = _varied_instance(optimum=20.0)

        def forbidden(*_args: object, **_kwargs: object) -> float:
            raise AssertionError("solver recomputed the objective or feasibility")

        inst.evaluar = forbidden  # type: ignore[method-assign]
        inst.es_factible = forbidden  # type: ignore[method-assign]
        inst.gap = forbidden  # type: ignore[method-assign]

        result = solve(inst, _config(max_evaluations=2), seed=2)

        self.assertIsNotNone(result.best_solution)
        self.assertGreater(result.evaluations, 0)

    def test_positive_gap_and_unknown_reference(self) -> None:
        known = solve(_varied_instance(optimum=100.0), _config(max_evaluations=2), seed=5)
        unknown = solve(_constant_instance(optimum=0.0), _config(max_evaluations=2), seed=5)
        empty = solve(
            _instance([1.0, 2.0], [[1.0, 1.0]], [0.0], 0.0),
            _config(max_evaluations=2),
            seed=5,
        )

        self.assertIsNotNone(known.gap)
        assert known.gap is not None
        self.assertGreater(known.gap, 0.0)
        self.assertAlmostEqual(known.gap, 100.0 * (100.0 - known.best_value) / 100.0)
        self.assertIsNone(unknown.gap)
        self.assertEqual(empty.best_solution, [0, 0])
        self.assertEqual(empty.best_value, 0.0)
        self.assertIsNone(empty.gap)
        self.assertIsNone(json.loads(json.dumps(asdict(unknown)))["gap"])

    def test_phase_counts_scouts_and_full_order(self) -> None:
        result = solve(
            _constant_instance(),
            _config(iterations=2, scout_limit=1, max_evaluations=10_000),
            seed=8,
        )

        self.assertEqual(result.termination_reason, TERMINATION_ITERATIONS)
        self.assertEqual(result.cycles_completed, 2)
        self.assertEqual(result.evaluations_by_phase[PHASE_INITIALIZATION], 2)
        self.assertEqual(result.evaluations_by_phase[PHASE_WOA], 8)
        self.assertEqual(result.evaluations_by_phase[PHASE_EMPLOYED], 8)
        self.assertEqual(result.evaluations_by_phase[PHASE_ONLOOKERS], 8)
        self.assertEqual(result.evaluations_by_phase[PHASE_SCOUTS], 4)
        self.assertEqual(result.evaluations, 2 + 2 * (4 + 4 + 4 + 2))
        self.assertEqual(
            result.phases_executed,
            [
                PHASE_INITIALIZATION,
                PHASE_WOA,
                PHASE_EMPLOYED,
                PHASE_ONLOOKERS,
                PHASE_SCOUTS,
                PHASE_WOA,
                PHASE_EMPLOYED,
                PHASE_ONLOOKERS,
                PHASE_SCOUTS,
            ],
        )
        self.assertEqual(result.source_trials, [0, 0])
        self.assertTrue(all(record.completeness and record.rho == 1.0 for record in result.history))
        _assert_accounting(self, result)

    def test_failure_counters_persist_across_cycles_and_woa_does_not_increment_them(self) -> None:
        result = solve(
            _constant_instance(),
            _config(iterations=2, scout_limit=100, max_evaluations=10_000),
            seed=9,
        )

        self.assertEqual(result.evaluations_by_phase[PHASE_SCOUTS], 0)
        self.assertEqual(sum(result.source_trials), 2 * 2 * 2)
        self.assertTrue(all(trial >= 2 for trial in result.source_trials))

    def test_short_budgets_stop_at_phase_boundaries_without_later_phases(self) -> None:
        inst = _constant_instance()
        cases = (
            ("initialization-only", 2, [PHASE_INITIALIZATION], {"initialization": 2, "woa": 0, "employed": 0, "onlookers": 0, "scouts": 0}, 2, [0, 0]),
            ("unused-before-woa", 3, [PHASE_INITIALIZATION], {"initialization": 2, "woa": 0, "employed": 0, "onlookers": 0, "scouts": 0}, 2, [0, 0]),
            ("after-woa", 6, [PHASE_INITIALIZATION, PHASE_WOA], {"initialization": 2, "woa": 4, "employed": 0, "onlookers": 0, "scouts": 0}, 6, [0, 0]),
            ("unused-before-employed", 7, [PHASE_INITIALIZATION, PHASE_WOA], {"initialization": 2, "woa": 4, "employed": 0, "onlookers": 0, "scouts": 0}, 6, [0, 0]),
            ("mid-woa-unused", 5, [PHASE_INITIALIZATION, PHASE_WOA], {"initialization": 2, "woa": 2, "employed": 0, "onlookers": 0, "scouts": 0}, 4, [0, 0]),
            ("after-employed", 10, [PHASE_INITIALIZATION, PHASE_WOA, PHASE_EMPLOYED], {"initialization": 2, "woa": 4, "employed": 4, "onlookers": 0, "scouts": 0}, 10, [1, 1]),
            ("one-employed-source", 8, [PHASE_INITIALIZATION, PHASE_WOA, PHASE_EMPLOYED], {"initialization": 2, "woa": 4, "employed": 2, "onlookers": 0, "scouts": 0}, 8, [1, 0]),
            ("after-onlookers-before-scouts", 14, [PHASE_INITIALIZATION, PHASE_WOA, PHASE_EMPLOYED, PHASE_ONLOOKERS], {"initialization": 2, "woa": 4, "employed": 4, "onlookers": 4, "scouts": 0}, 14, None),
            ("mid-scouts", 15, [PHASE_INITIALIZATION, PHASE_WOA, PHASE_EMPLOYED, PHASE_ONLOOKERS, PHASE_SCOUTS], {"initialization": 2, "woa": 4, "employed": 4, "onlookers": 4, "scouts": 1}, 15, None),
        )
        for name, budget, phases, counts, evaluations, trials in cases:
            with self.subTest(name=name):
                result = solve(
                    inst,
                    _config(iterations=2, scout_limit=1, max_evaluations=budget),
                    seed=1,
                )
                self.assertEqual(result.termination_reason, TERMINATION_EVALUATION_BUDGET)
                self.assertEqual(result.cycles_completed, 0)
                self.assertEqual(result.phases_executed, phases)
                self.assertEqual(result.evaluations_by_phase, counts)
                self.assertEqual(result.evaluations, evaluations)
                self.assertLess(result.evaluations, budget if budget != evaluations else budget + 1)
                self.assertFalse(result.history[-1].completeness)
                self.assertEqual(result.history[0].a, 2.0)
                self.assertEqual(result.history[0].rho, 1.0)
                self.assertEqual(len(result.history), 1)
                if trials is not None:
                    self.assertEqual(result.source_trials, trials)
                _assert_accounting(self, result)

        between_cycles = solve(
            inst,
            _config(iterations=2, scout_limit=1, max_evaluations=16),
            seed=1,
        )
        self.assertEqual(between_cycles.cycles_completed, 1)
        self.assertEqual(len(between_cycles.history), 2)
        self.assertTrue(between_cycles.history[0].completeness)
        self.assertFalse(between_cycles.history[1].completeness)
        self.assertEqual(between_cycles.evaluations_by_phase[PHASE_SCOUTS], 2)
        self.assertEqual(
            between_cycles.phases_executed,
            [PHASE_INITIALIZATION, PHASE_WOA, PHASE_EMPLOYED, PHASE_ONLOOKERS, PHASE_SCOUTS],
        )
        self.assertEqual(between_cycles.termination_reason, TERMINATION_EVALUATION_BUDGET)

    def test_archive_updates_before_partial_termination(self) -> None:
        inst = _varied_instance()
        seen: list[float] = []

        def record(solution: list[int], problem: MKPInstance) -> tuple[list[int], float]:
            repaired, value = reparar_solucion(solution, problem)
            seen.append(float(value))
            return repaired, value

        with patch.object(hybrid, "reparar_solucion", record):
            result = solve(inst, _config(max_evaluations=4, scout_limit=100), seed=36)

        self.assertEqual(result.evaluations, 4)
        self.assertEqual(result.evaluations_by_phase[PHASE_WOA], 2)
        self.assertEqual(result.evaluations_by_phase[PHASE_EMPLOYED], 0)
        self.assertGreater(max(seen[2:]), max(seen[:2]))
        self.assertEqual(result.best_value, max(seen))
        self.assertGreater(result.best_value, max(seen[:2]))
        self.assertFalse(result.history[-1].completeness)
        self.assertEqual(result.history[-1].best, result.best_value)
        _assert_accounting(self, result)

    def test_baseline_a_endpoints_and_json_history(self) -> None:
        result = solve(
            _constant_instance(),
            _config(iterations=5, scout_limit=100, max_evaluations=10_000, use_stagnation=False),
            seed=3,
        )

        self.assertEqual(result.cycles_completed, 5)
        self.assertEqual(result.history[0].a, 2.0)
        self.assertEqual(result.history[-1].a, 0.0)
        self.assertEqual([record.schedule_index for record in result.history], [0, 1, 2, 3, 4])
        self.assertTrue(all(record.rho == 1.0 for record in result.history))
        self.assertEqual(result.diagnostics, [])
        payload = json.loads(json.dumps(asdict(result)))
        self.assertEqual(payload["history"][0]["a"], 2.0)
        self.assertEqual(payload["history"][-1]["completeness"], True)
        self.assertEqual(payload["termination_reason"], TERMINATION_ITERATIONS)

    def test_clipped_lb2_position_stays_inside_v_max(self) -> None:
        captured: list[np.ndarray] = []

        def capture_l1(position: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
            captured.append(np.array(position, copy=True))
            return np.zeros(len(position), dtype=float)

        def capture_l2(position: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
            return np.zeros(len(position), dtype=float)

        with patch.object(hybrid, "_calcular_L1", capture_l1), patch.object(
            hybrid, "_calcular_L2", capture_l2
        ):
            solve(
                _constant_instance(),
                _config(v_max=0.25, max_evaluations=4, scout_limit=100),
                seed=0,
            )

        self.assertGreaterEqual(len(captured), 1)
        self.assertTrue(all(np.max(np.abs(position)) <= 0.25 + 1e-12 for position in captured))


class TestScheduleAndAcceptance(unittest.TestCase):
    def test_a_schedule_endpoints_and_unchanged_a2(self) -> None:
        self.assertEqual(_control_parameters(0, 300, False), (2.0, 1.0))
        self.assertEqual(_control_parameters(299, 300, False), (0.0, 1.0))
        self.assertEqual(_control_parameters(0, 300, True), (2.0, 2.0))
        self.assertEqual(_a2(0, 300), -1.0)
        self.assertEqual(_a2(299, 300), -2.0)

    def test_woa_move_covers_leader_peer_and_spiral(self) -> None:
        position = np.array([0.0, 0.0])
        leader = np.array([1.0, -1.0])
        peer = np.array([2.0, 2.0])

        encircling = _woa_destination(position, leader, peer, A=0.5, C=0.25, p=0.1, l=0.0, b=1.0)
        peer_move = _woa_destination(position, leader, peer, A=1.0, C=0.25, p=0.1, l=0.0, b=1.0)
        spiral = _woa_destination(position, leader, peer, A=0.0, C=0.0, p=0.9, l=0.0, b=1.0)

        np.testing.assert_allclose(encircling, [0.875, -1.125])
        np.testing.assert_allclose(peer_move, [1.5, 1.5])
        np.testing.assert_allclose(spiral, [2.0, 0.0])
        np.testing.assert_allclose(_clip_coordinates(np.array([2.0, -3.0]), 1.0), [1.0, -1.0])

    def test_frozen_leader_does_not_share_memory(self) -> None:
        positions = np.array([[1.0, 2.0], [3.0, 4.0]])
        leader = _freeze_leader(positions, 0)

        positions[0, 0] = 99.0

        self.assertEqual(leader[0], 1.0)
        self.assertFalse(np.shares_memory(leader, positions))

    def test_lb2_ties_prefer_l2(self) -> None:
        self.assertFalse(_prefer_l1(5.0, 5.0))
        self.assertTrue(_prefer_l1(6.0, 5.0))
        self.assertFalse(_prefer_l1(4.0, 5.0))

    def test_woa_and_abc_acceptance_rules(self) -> None:
        state = _state_with_source([1, 0], fitness=5.0, trials=4)

        _commit_woa(state, 0, np.array([9.0, -9.0]), [0, 1], 4.0)
        self.assertEqual(state.bits[0], [1, 0])
        self.assertEqual(state.fitness[0], 5.0)
        self.assertEqual(int(state.trials[0]), 4)
        np.testing.assert_allclose(state.positions[0], [6.0, -6.0])

        _commit_woa(state, 0, np.array([1.0, 1.0]), [0, 0], 5.0)
        self.assertEqual(state.bits[0], [0, 0])
        self.assertEqual(int(state.trials[0]), 4)

        _commit_woa(state, 0, np.array([0.2, 0.2]), [1, 1], 8.0)
        self.assertEqual(state.bits[0], [1, 1])
        self.assertEqual(state.fitness[0], 8.0)
        self.assertEqual(int(state.trials[0]), 0)

        _commit_abc(state, 0, np.array([0.3, 0.3]), [0, 1], 8.0)
        self.assertEqual(state.bits[0], [1, 1])
        self.assertEqual(int(state.trials[0]), 1)

        _commit_abc(state, 0, np.array([0.4, 0.4]), [1, 0], 9.0)
        self.assertEqual(state.bits[0], [1, 0])
        self.assertEqual(state.fitness[0], 9.0)
        self.assertEqual(int(state.trials[0]), 0)
        np.testing.assert_allclose(state.positions[0], [0.4, 0.4])

    def test_roulette_uses_fitness_and_handles_an_all_zero_population(self) -> None:
        np.testing.assert_allclose(_roulette_probabilities([1.0, 3.0]), [0.25, 0.75])
        np.testing.assert_allclose(_roulette_probabilities([0.0, 0.0]), [0.5, 0.5])

    def test_scout_resets_its_source_without_clearing_the_archive(self) -> None:
        state = _state_with_source([1, 1, 1], fitness=6.0, trials=3)
        state.best_value = 6.0
        state.best_solution = [1, 1, 1]
        state.trials[0] = 3

        _replace_scout(state, 0)

        self.assertEqual(state.best_value, 6.0)
        self.assertEqual(state.best_solution, [1, 1, 1])
        self.assertEqual(int(state.trials[0]), 0)
        self.assertEqual(state.by_phase[PHASE_SCOUTS], 1)
        self.assertEqual(state.fitness[0], 6.0)

    def test_complete_cycle_hook_is_skipped_when_the_cycle_is_partial(self) -> None:
        calls: list[int] = []

        def hook(state: _RunState, record: hybrid.CycleRecord) -> bool:
            del state
            calls.append(record.cycle)
            return False

        with patch.object(hybrid, "_after_complete_cycle", hook):
            solve(_constant_instance(), _config(max_evaluations=2, iterations=4), seed=0)

        self.assertEqual(calls, [])

    def test_hook_can_restart_the_next_cycle_without_resetting_trials(self) -> None:
        def hook(state: _RunState, record: hybrid.CycleRecord) -> bool:
            state.diagnostics.append({"cycle": record.cycle, "best": record.best})
            return record.cycle == 0

        with patch.object(hybrid, "_after_complete_cycle", hook):
            result = solve(
                _constant_instance(),
                _config(iterations=3, scout_limit=100, max_evaluations=10_000),
                seed=0,
            )

        self.assertEqual([record.schedule_index for record in result.history], [0, 0, 1])
        self.assertEqual(result.history[1].a, 2.0)
        self.assertEqual(result.history[1].rho, 2.0)
        self.assertEqual(result.history[2].a, 1.0)
        self.assertEqual(result.history[2].rho, 1.5)
        self.assertEqual(
            result.diagnostics,
            [
                {"cycle": record.cycle, "best": record.best}
                for record in result.history
            ],
        )
        self.assertGreater(sum(result.source_trials), 0)


def _state_with_source(bits: list[int], fitness: float, trials: int) -> _RunState:
    inst = _constant_instance()
    config = _config(v_max=6.0)
    state = _RunState(
        inst=inst,
        config=config,
        rng=np.random.default_rng(0),
        positions=np.zeros((1, len(bits)), dtype=float),
        bits=[list(bits)],
        fitness=[fitness],
        trials=np.array([trials], dtype=int),
        max_evaluations=10,
        scout_limit=1,
    )
    state.by_phase = {phase: 0 for phase in PHASES}
    return state


def _flat_monitor(**overrides: object) -> StagnationConfig:
    values: dict[str, object] = {
        "window": 4,
        "band": 1,
        "min_slope": 2.0,
        "plateau_max": 2,
        "patience": 2,
        "adapt_thresholds": False,
        "use_ddtw": False,
    }
    values.update(overrides)
    return StagnationConfig(**values)  # type: ignore[arg-type]


def _recovery_config(**overrides: object) -> HybridConfig:
    values: dict[str, object] = {
        "population_size": 2,
        "iterations": 12,
        "max_evaluations": 10_000,
        "scout_limit": 100,
        "use_stagnation": True,
        "stag_cfg": _flat_monitor(),
    }
    values.update(overrides)
    return HybridConfig(**values)  # type: ignore[arg-type]


class TestRecovery(unittest.TestCase):
    def test_flat_detector_restarts_next_cycle_and_keeps_the_same_slope(self) -> None:
        snapshots: list[int] = []
        real = hybrid._after_complete_cycle

        def spy(state: _RunState, record: hybrid.CycleRecord) -> bool:
            positions = state.positions.copy()
            trials = state.trials.copy()
            bits = [list(row) for row in state.bits]
            best = state.best_value
            fired = real(state, record)
            if fired:
                self.assertTrue(np.array_equal(state.positions, positions))
                self.assertTrue(np.array_equal(state.trials, trials))
                self.assertEqual(state.bits, bits)
                self.assertEqual(state.best_value, best)
                snapshots.append(record.cycle)
            return fired

        with patch.object(hybrid, "_after_complete_cycle", spy):
            result = solve(_constant_instance(), _recovery_config(), seed=1)

        self.assertEqual(snapshots, [4, 9])
        self.assertEqual(result.cycles_completed, 12)
        self.assertEqual(result.termination_reason, TERMINATION_ITERATIONS)
        self.assertEqual([item["cycle"] for item in result.diagnostics if item["fire"]], [4, 9])
        self.assertTrue(result.diagnostics[4]["recovery_applied"])
        self.assertTrue(result.diagnostics[9]["recovery_applied"])
        self.assertEqual(result.history[0].rho, 1.0)
        self.assertEqual(result.history[4].rho, 1.0)
        self.assertEqual(result.history[5].a, 2.0)
        self.assertEqual(result.history[5].rho, 2.0)
        self.assertEqual(result.history[5].schedule_index, 0)
        self.assertEqual(result.history[10].a, 2.0)
        self.assertEqual(result.history[10].rho, 2.0)
        slope = -2.0 / 11.0
        self.assertAlmostEqual(result.history[6].a - result.history[5].a, slope)
        self.assertAlmostEqual(result.history[7].a - result.history[6].a, slope)
        self.assertAlmostEqual(result.history[11].a - result.history[10].a, slope)
        self.assertAlmostEqual(result.history[6].rho, 2.0 - 1.0 / 11.0)
        for offset in (5, 6, 7):
            self.assertFalse(result.diagnostics[offset]["ready"])
            self.assertEqual(result.diagnostics[offset]["n"], offset - 4)
        self.assertTrue(result.diagnostics[8]["ready"])
        self.assertFalse(result.diagnostics[8]["fire"])
        self.assertEqual(result.diagnostics[4]["D1_vs_ramp"], 12.0)
        self.assertEqual(result.diagnostics[4]["D2_vs_const"], 0.0)
        self.assertEqual(result.diagnostics[4]["trigger_streak"], 2)
        self.assertEqual(result.evaluations_by_phase[PHASE_SCOUTS], 0)
        self.assertEqual(sum(result.source_trials), 2 * 2 * 12)
        self.assertTrue(all(trial >= 12 for trial in result.source_trials))
        _assert_accounting(self, result)

    def test_ddtw_uses_the_same_rearm_cadence_with_derivative_distance(self) -> None:
        result = solve(
            _constant_instance(),
            _recovery_config(stag_cfg=_flat_monitor(use_ddtw=True)),
            seed=1,
        )

        self.assertTrue(result.metadata["config"]["stagnation"]["use_ddtw"])
        self.assertEqual([item["cycle"] for item in result.diagnostics if item["fire"]], [4, 9])
        self.assertEqual(result.diagnostics[4]["D1_vs_ramp"], 6.0)
        self.assertEqual(result.diagnostics[5]["n"], 1)
        self.assertFalse(result.diagnostics[5]["ready"])

    def test_last_cycle_fire_is_not_marked_applied(self) -> None:
        result = solve(
            _constant_instance(),
            _recovery_config(iterations=5),
            seed=1,
        )

        self.assertEqual(result.cycles_completed, 5)
        self.assertTrue(result.diagnostics[4]["fire"])
        self.assertFalse(result.diagnostics[4]["recovery_applied"])
        self.assertEqual([record.schedule_index for record in result.history], [0, 1, 2, 3, 4])
        self.assertTrue(all(record.rho == 1.0 for record in result.history))
        self.assertEqual(result.history[-1].a, 0.0)

    def test_terminal_remainder_does_not_claim_an_unexecuted_recovery(self) -> None:
        for budget in (62, 63):
            with self.subTest(budget=budget):
                result = solve(
                    _constant_instance(),
                    _recovery_config(iterations=6, max_evaluations=budget),
                    seed=1,
                )
                self.assertEqual(result.evaluations, 62)
                self.assertEqual(result.cycles_completed, 5)
                self.assertEqual(result.phases_executed.count(PHASE_WOA), 5)
                self.assertFalse(result.history[-1].completeness)
                self.assertEqual(result.history[-1].a, 0.0)
                self.assertEqual(result.history[-1].rho, 1.0)
                self.assertEqual(result.diagnostics[-1]["cycle"], 4)
                self.assertTrue(result.diagnostics[-1]["fire"])
                self.assertFalse(result.diagnostics[-1]["recovery_applied"])
                _assert_accounting(self, result)

        executed = solve(
            _constant_instance(),
            _recovery_config(iterations=6, max_evaluations=64),
            seed=1,
        )
        self.assertEqual(executed.evaluations, 64)
        self.assertEqual(executed.cycles_completed, 5)
        self.assertEqual(executed.phases_executed.count(PHASE_WOA), 6)
        self.assertEqual(executed.evaluations_by_phase[PHASE_WOA], 22)
        self.assertFalse(executed.history[-1].completeness)
        self.assertEqual(executed.history[-1].schedule_index, 0)
        self.assertEqual(executed.history[-1].a, 2.0)
        self.assertEqual(executed.history[-1].rho, 2.0)
        self.assertTrue(executed.diagnostics[4]["fire"])
        self.assertTrue(executed.diagnostics[4]["recovery_applied"])
        _assert_accounting(self, executed)

    def test_partial_cycle_does_not_update_the_detector(self) -> None:
        result = solve(
            _constant_instance(),
            _recovery_config(iterations=3, max_evaluations=14),
            seed=1,
        )

        self.assertEqual(result.cycles_completed, 1)
        self.assertEqual(len(result.history), 2)
        self.assertFalse(result.history[1].completeness)
        self.assertEqual([item["cycle"] for item in result.diagnostics], [0])
        self.assertFalse(result.diagnostics[0]["ready"])
        self.assertEqual(result.evaluations, 14)
        _assert_accounting(self, result)

    def test_none_config_observes_and_disabled_config_does_not(self) -> None:
        observed = solve(
            _constant_instance(),
            _config(iterations=2, use_stagnation=True, stag_cfg=None),
            seed=2,
        )
        baseline = solve(
            _constant_instance(),
            _config(iterations=2, use_stagnation=False, stag_cfg=_flat_monitor()),
            seed=2,
        )

        self.assertEqual(len(observed.diagnostics), 2)
        self.assertIsNotNone(observed.metadata["config"]["stagnation"])
        self.assertEqual(baseline.diagnostics, [])
        self.assertIsNone(baseline.metadata["config"]["stagnation"])
        self.assertEqual([record.rho for record in baseline.history], [1.0, 1.0])
        self.assertEqual([record.schedule_index for record in baseline.history], [0, 1])

    def test_two_solves_do_not_share_or_mutate_the_monitor_config(self) -> None:
        stag_cfg = _flat_monitor()
        stag_cfg.band = 0
        config = _recovery_config(stag_cfg=stag_cfg)
        first = solve(_constant_instance(), config, seed=3)
        second = solve(_constant_instance(), config, seed=3)

        self.assertEqual(stag_cfg.band, 0)
        self.assertEqual(asdict(first), asdict(second))
        self.assertEqual(first.metadata["config"]["stagnation"]["band"], 1)
        self.assertEqual([item["cycle"] for item in first.diagnostics if item["fire"]], [4, 9])

    def test_invalid_monitor_options_are_rejected_without_mutating_the_object(self) -> None:
        stag_cfg = StagnationConfig(window=4, band=1, min_slope=2.0, plateau_max=2, patience=2)
        stag_cfg.window = 1
        with self.assertRaises(ValueError):
            HybridConfig(population_size=2, iterations=2, stag_cfg=stag_cfg)
        self.assertEqual(stag_cfg.window, 1)
        with self.assertRaises(ValueError):
            HybridConfig(population_size=2, iterations=2, use_stagnation=False, stag_cfg=stag_cfg)


class TestCommandLine(unittest.TestCase):
    def test_help_lists_local_controls(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            code = main(["--help"])

        self.assertEqual(code, 0)
        text = stdout.getvalue()
        for flag in (
            "--instance",
            "--instance-index",
            "--seed",
            "--population-size",
            "--iterations",
            "--max-evaluations",
            "--scout-limit",
            "--no-stagnation",
            "--ddtw",
            "--window",
            "--band",
            "--min-slope",
            "--plateau-max",
            "--patience",
        ):
            self.assertIn(flag, text)

    def test_cli_json_uses_only_a_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.dat"
            path.write_text("1\n3 1 9\n5 4 3\n2 2 2\n4\n", encoding="utf-8")
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main([
                    "--instance",
                    str(path),
                    "--instance-index",
                    "0",
                    "--seed",
                    "1",
                    "--population-size",
                    "2",
                    "--iterations",
                    "2",
                    "--max-evaluations",
                    "40",
                    "--no-stagnation",
                ])

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["metadata"]["seed"], 1)
        self.assertEqual(payload["metadata"]["config"]["population_size"], 2)
        self.assertEqual(payload["metadata"]["config"]["iterations"], 2)
        self.assertEqual(payload["metadata"]["config"]["max_evaluations"], 40)
        self.assertFalse(payload["metadata"]["config"]["use_stagnation"])
        self.assertEqual(payload["metadata"]["instance"]["n"], 3)
        self.assertEqual(payload["metadata"]["instance"]["index"], 0)
        self.assertEqual(payload["metadata"]["instance"]["reference"], 9.0)
        self.assertEqual(sum(payload["evaluations_by_phase"].values()), payload["evaluations"])
        self.assertEqual(payload["cycles_completed"], 2)

    def test_cli_rejects_urls_missing_files_and_invalid_monitor_options(self) -> None:
        def fail_if_called(_path: Path) -> list[dict]:
            raise AssertionError("loader was called")

        with patch.object(hybrid, "_read_orlibrary", fail_if_called):
            for argv in (
                ["--instance", "https://example.com/mknap.txt", "--seed", "1"],
                ["--instance", "http://example.com/mknap.txt", "--seed", "1"],
                ["--instance", str(Path(tempfile.gettempdir()) / "missing-mkp-file.txt"), "--seed", "1"],
                ["--instance", str(Path(tempfile.gettempdir())), "--seed", "1", "--window", "1"],
            ):
                with self.subTest(argv=argv), redirect_stderr(StringIO()):
                    self.assertEqual(main(argv), 1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.dat"
            path.write_text("1\n2 1 3\n1 2\n1 1\n2\n", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr):
                code = main([
                    "--instance",
                    str(path),
                    "--seed",
                    "1",
                    "--window",
                    "1",
                    "--population-size",
                    "2",
                    "--iterations",
                    "2",
                    "--max-evaluations",
                    "8",
                ])
            self.assertEqual(code, 1)
            self.assertIn("window", stderr.getvalue())
            with redirect_stderr(StringIO()):
                self.assertEqual(
                    main([
                        "--instance",
                        str(path),
                        "--seed",
                        "1",
                        "--instance-index",
                        "3",
                        "--population-size",
                        "2",
                        "--iterations",
                        "2",
                        "--max-evaluations",
                        "8",
                    ]),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
