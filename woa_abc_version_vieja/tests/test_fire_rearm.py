"""Fire confirmado: tres condiciones, patience y rearme de ventana."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np

from woa_abc_version_vieja.config import DTWConfig, StrategyConfig, WOAABCConfig
from woa_abc_version_vieja.core.binary_woa_abc import (
    BinaryWOAABCConfig,
    BinaryWOAABCOptimizer,
    MKPOptimizationResult,
)
from woa_abc_version_vieja.core.woa_abc import OptimizationResult, WOAABCOptimizer
from woa_abc_version_vieja.domains.base import Problem
from woa_abc_version_vieja.domains.mkp import MKPInstance
from woa_abc_version_vieja.dtw.monitor import StagnationMonitor
from woa_abc_version_vieja.dtw.strategies import (
    AdaptationController,
    M0_NO_DTW,
    M1_FIRE_D2,
    M2_FIRE_3COND,
    M4_D2_CONTINUOUS,
    M5_SIGMOID_DELTA,
    M6_HYSTERESIS_WOA_ABC,
    M8_FOUR_STATE_WOA_ABC,
    VARIANT_NAMES,
)
from woa_abc_version_vieja.reporting.mkp import save_mkp_run_plots
from woa_abc_version_vieja.reporting.plots import save_run_plots


PULSE_WINDOW = 5
PULSE_PATIENCE = 2
PULSE_FIRES = (5, 11, 17)


def pulse_config(**overrides: object) -> DTWConfig:
    values: dict[str, object] = {
        "window": PULSE_WINDOW,
        "band": 1,
        "min_slope": 2.0,
        "plateau_max": 2,
        "patience": PULSE_PATIENCE,
        "use_ddtw": True,
        "adapt_thresholds": False,
    }
    values.update(overrides)
    return DTWConfig(**values)


def feed(monitor: StagnationMonitor, values: list[float]) -> list[dict[str, object]]:
    return [monitor.update(value) for value in values]


def fire_indices(states: list[dict[str, object]]) -> list[int]:
    return [index for index, state in enumerate(states) if state.get("fire")]


class TestConfirmedFireGate(unittest.TestCase):
    def test_distance_to_constant_can_block_fire(self) -> None:
        sequences = {
            "bootstrap": [0.0, 100.0, 100.0, 100.0, 100.0],
            "learned_threshold": [2.0 * index for index in range(14)] + [100.0, 100.0],
        }
        for use_ddtw in (False, True):
            for adaptive in (False, True):
                for phase, values in sequences.items():
                    with self.subTest(ddtw=use_ddtw, adaptive=adaptive, phase=phase):
                        monitor = StagnationMonitor(pulse_config(
                            plateau_max=1, patience=1,
                            use_ddtw=use_ddtw, adapt_thresholds=adaptive,
                        ))
                        state = feed(monitor, values)[-1]
                        self.assertTrue(state["ready"])
                        self.assertTrue(state["cond_plateau"])
                        self.assertTrue(state["cond_ramp"])
                        self.assertGreater(state["D2_vs_const"], state["theta_c"])
                        self.assertFalse(state["cond_constant"])
                        self.assertFalse(state["fire"])

    def test_flat_window_fires_on_patience_then_rearms(self) -> None:
        monitor = StagnationMonitor(pulse_config())
        states = feed(monitor, [10.0] * 19)

        self.assertFalse(states[3]["ready"])
        self.assertNotIn("D1_vs_ramp", states[3])
        self.assertEqual(states[4]["trigger_streak"], 1)
        self.assertTrue(states[4]["cond_plateau"])
        self.assertTrue(states[4]["cond_constant"])
        self.assertTrue(states[4]["cond_ramp"])
        self.assertFalse(states[4]["fire"])
        self.assertEqual(states[5]["n"], 6)
        self.assertEqual(fire_indices(states), list(PULSE_FIRES))
        for index in PULSE_FIRES:
            self.assertTrue(states[index]["ready"])
            self.assertGreaterEqual(int(states[index]["trigger_streak"]), PULSE_PATIENCE)
            self.assertEqual(states[index]["D1_vs_ramp"], 8.0)
            self.assertEqual(states[index]["D2_vs_const"], 0.0)
            self.assertEqual(states[index + 1]["n"], 1)
            self.assertFalse(states[index + 1]["ready"])
            self.assertFalse(states[index + 1]["fire"])
            self.assertEqual(states[index + 1]["trigger_streak"], 0)
            self.assertNotIn("D2_vs_const", states[index + 1])

    def test_smooth_improvement_never_confirms_fire(self) -> None:
        states = feed(
            StagnationMonitor(pulse_config()),
            [10.0 + 2.0 * index for index in range(16)],
        )
        self.assertEqual(fire_indices(states), [])
        ready = [state for state in states if state["ready"]]
        self.assertTrue(ready)
        self.assertTrue(all(not state["cond_plateau"] for state in ready))
        self.assertTrue(all(not state["cond_ramp"] for state in ready))

    def test_failed_ramp_blocks_fire_and_keeps_nonnegative_thresholds(self) -> None:
        states = feed(
            StagnationMonitor(pulse_config(min_slope=0.0)),
            [10.0] * 16,
        )
        self.assertEqual(fire_indices(states), [])
        ready = [state for state in states if state["ready"]]
        self.assertTrue(ready)
        for state in ready:
            self.assertTrue(state["cond_plateau"])
            self.assertTrue(state["cond_constant"])
            self.assertFalse(state["cond_ramp"])
            self.assertGreaterEqual(float(state["theta_c"]), 0.0)
            self.assertGreaterEqual(float(state["theta_delta"]), 0.0)
            self.assertEqual(state["trigger_streak"], 0)

    def test_improvement_breaks_streak_then_one_fire_and_rearm(self) -> None:
        values = [10.0] * 5 + [20.0] * 8 + [22.0 + 2.0 * index for index in range(8)]
        states = feed(StagnationMonitor(pulse_config()), values)

        self.assertFalse(any(state["fire"] for state in states[:10]))
        self.assertEqual(states[4]["trigger_streak"], 1)
        self.assertFalse(states[5]["cond_plateau"])
        self.assertEqual(states[5]["trigger_streak"], 0)
        self.assertGreater(int(states[5]["n"]), 1)
        # The jump is still inside the window: a short plateau alone cannot fire.
        self.assertTrue(states[8]["cond_plateau"])
        self.assertFalse(states[8]["cond_constant"])
        self.assertGreater(states[8]["D2_vs_const"], states[8]["theta_c"])
        self.assertEqual(states[9]["trigger_streak"], 1)
        self.assertTrue(states[10]["fire"])
        self.assertEqual(states[10]["D1_vs_ramp"], 8.0)
        self.assertEqual(states[11]["n"], 1)
        self.assertFalse(states[11]["ready"])
        self.assertEqual(fire_indices(states), [10])

    def test_patience_boundary_does_not_reset_until_streak_completes(self) -> None:
        monitor = StagnationMonitor(pulse_config(patience=11))
        states = feed(monitor, [3.0] * 16)
        self.assertEqual(states[13]["trigger_streak"], 10)
        self.assertFalse(states[13]["fire"])
        self.assertEqual(states[14]["n"], 15)
        self.assertTrue(states[14]["fire"])
        self.assertEqual(states[14]["theta_r"], 2.5)
        self.assertEqual(states[15]["n"], 1)
        self.assertFalse(states[15]["fire"])

    def test_rearm_clears_adaptive_history_before_the_next_window(self) -> None:
        states = feed(
            StagnationMonitor(pulse_config(patience=11, adapt_thresholds=True)),
            [3.0] * 20,
        )
        self.assertTrue(states[14]["fire"])
        self.assertEqual(states[14]["theta_r"], 8.0)
        self.assertEqual(states[19]["n"], 5)
        self.assertTrue(states[19]["ready"])
        self.assertFalse(states[19]["fire"])
        self.assertEqual(states[19]["theta_r"], 2.5)

    def test_config_and_sibling_monitor_stay_independent(self) -> None:
        config = pulse_config()
        before = asdict(config)
        fired = StagnationMonitor(config)
        untouched = StagnationMonitor(config)
        feed(fired, [10.0] * 8)
        self.assertEqual(asdict(config), before)
        self.assertIs(fired.config, config)
        self.assertEqual(untouched.update(10.0)["n"], 1)
        self.assertFalse(untouched.update(10.0)["fire"])


class TestVariantFireAuthority(unittest.TestCase):
    def test_d2_alone_is_not_a_fire_and_confirmed_fire_is_shared(self) -> None:
        config = StrategyConfig()
        d2_only = {
            "ready": True,
            "fire": False,
            "D2_vs_const": 0.0,
            "theta_c": 8.0,
            "delta": 8.0,
            "theta_delta": 8.0,
        }
        confirmed = {
            "ready": True,
            "fire": True,
            "D2_vs_const": 8.0,
            "theta_c": 8.0,
            "delta": -8.0,
            "theta_delta": 8.0,
        }
        for variant in VARIANT_NAMES:
            if variant == M0_NO_DTW:
                decision = AdaptationController(variant, config).decide(d2_only)
                self.assertFalse(decision.raw_fire)
                self.assertEqual(decision.mode, "base")
                continue
            blocked = AdaptationController(variant, config).decide(d2_only)
            self.assertFalse(blocked.raw_fire)
            accepted = AdaptationController(variant, config).decide(confirmed)
            self.assertTrue(accepted.raw_fire)

        self.assertEqual(
            AdaptationController(M1_FIRE_D2, config).decide(d2_only).mode,
            "exploit",
        )
        self.assertEqual(
            AdaptationController(M1_FIRE_D2, config).decide(confirmed).mode,
            "explore",
        )
        self.assertEqual(
            AdaptationController(M2_FIRE_3COND, config).decide(confirmed).mode,
            "explore",
        )
        continuous = AdaptationController(M4_D2_CONTINUOUS, config).decide(confirmed)
        self.assertLessEqual(float(continuous.intensity), 0.5)
        self.assertEqual(continuous.mode, "exploit")
        self.assertTrue(continuous.raw_fire)
        self.assertGreater(
            float(AdaptationController(M4_D2_CONTINUOUS, config).decide(d2_only).intensity),
            0.5,
        )
        self.assertFalse(
            AdaptationController(M5_SIGMOID_DELTA, config).decide(d2_only).raw_fire
        )
        self.assertTrue(
            AdaptationController(M6_HYSTERESIS_WOA_ABC, config).decide(confirmed).raw_fire
        )
        self.assertTrue(
            AdaptationController(M8_FOUR_STATE_WOA_ABC, config).decide(confirmed).raw_fire
        )
        self.assertEqual(asdict(config), asdict(StrategyConfig()))


def _constant_problem() -> Problem:
    return Problem(
        name="constant",
        dimension=3,
        lower=np.zeros(3),
        upper=np.ones(3),
        objective=lambda _position: 1.0,
        optimum=1.0,
    )


def _flat_mkp() -> MKPInstance:
    return MKPInstance(
        family="flat",
        index=0,
        profits=np.zeros(4),
        weights=np.ones((1, 4)),
        capacities=np.asarray([2.0]),
        best_known=None,
    )


def _algorithm(variant: str, iterations: int) -> WOAABCConfig:
    return WOAABCConfig(
        pop_size=4,
        iterations=iterations,
        seed=3,
        variant=variant,
        dtw=pulse_config(),
    )


class TestEngineFireCounts(unittest.TestCase):
    def test_continuous_variants_count_each_confirmed_fire_once(self) -> None:
        for variant in VARIANT_NAMES:
            with self.subTest(variant=variant):
                result = WOAABCOptimizer(_constant_problem(), _algorithm(variant, 12)).run()
                self._assert_run_contract(result, variant, iterations=12)
                if variant == M0_NO_DTW:
                    self.assertEqual(result.fire_count, 0)
                    self.assertEqual(result.control_events, [])
                    self.assertTrue(all(not row.get("raw_fire") for row in result.dtw_history))
                    self.assertTrue(all("D1_vs_ramp" not in row for row in result.dtw_history))
                    continue
                self.assertEqual(
                    [row["iteration"] for row in result.dtw_history if row.get("raw_fire")],
                    [5, 11],
                )
                self.assertEqual(result.fire_count, 2)
                self.assertEqual(result.dtw_history[6]["n"], 1)
                self.assertFalse(result.dtw_history[6]["ready"])

    def test_mkp_variants_match_continuous_cadence_and_keep_lb2_clock(self) -> None:
        for variant in VARIANT_NAMES:
            with self.subTest(variant=variant):
                result = BinaryWOAABCOptimizer(
                    _flat_mkp(),
                    BinaryWOAABCConfig(
                        algorithm=_algorithm(variant, 12),
                        local_search_interval=0,
                        early_stop_at_best_known=False,
                    ),
                ).run()
                self._assert_run_contract(result, variant, iterations=12)
                for index, row in enumerate(result.parameter_history):
                    progress = index / 11
                    self.assertAlmostEqual(row["lb2_g1"], 0.5 + 0.5 * progress)
                    self.assertAlmostEqual(row["lb2_g2"], 0.5 + 6.7 * progress)
                    self.assertAlmostEqual(row["lb2_g3"], 0.5 - 0.5 * progress)
                if variant == M0_NO_DTW:
                    self.assertEqual(result.fire_count, 0)
                    self.assertTrue(all(not row.get("raw_fire") for row in result.dtw_history))
                else:
                    self.assertEqual(result.fire_count, 2)
                    self.assertEqual(result.iterations_completed, 12)

    def test_fire_without_parameter_change_is_kept_and_not_counted_as_update(self) -> None:
        result = WOAABCOptimizer(
            _constant_problem(),
            _algorithm(M4_D2_CONTINUOUS, 12),
        ).run()
        unchanged = [
            row for row in result.dtw_history
            if row.get("raw_fire") and not row.get("parameter_changed")
        ]
        self.assertTrue(unchanged)
        self.assertEqual(
            result.parameter_update_count,
            sum(1 for row in result.dtw_history if row.get("parameter_changed")),
        )
        self.assertEqual(result.fire_count, 2)
        for row in unchanged:
            matches = [
                event for event in result.control_events
                if event.get("raw_fire") and event["detected_iteration"] == row["iteration"]
            ]
            self.assertEqual(len(matches), 1)

    def test_last_cycle_fire_is_recorded_without_a_false_applied_iteration(self) -> None:
        continuous = WOAABCOptimizer(
            _constant_problem(),
            _algorithm(M1_FIRE_D2, 6),
        ).run()
        binary = BinaryWOAABCOptimizer(
            _flat_mkp(),
            BinaryWOAABCConfig(
                algorithm=_algorithm(M2_FIRE_3COND, 6),
                local_search_interval=0,
                early_stop_at_best_known=False,
            ),
        ).run()
        for result in (continuous, binary):
            self.assertEqual(result.fire_count, 1)
            self.assertEqual(len(result.convergence_history), 6)
            fires = [event for event in result.control_events if event.get("raw_fire")]
            self.assertEqual([event["detected_iteration"] for event in fires], [5])
            self.assertIsNone(fires[0]["applied_iteration"])
            self.assertNotIn("explore", result.mode_history)

    def _assert_run_contract(self, result: object, variant: str, iterations: int) -> None:
        self.assertEqual(result.variant, variant)
        self.assertEqual(len(result.convergence_history), iterations)
        self.assertEqual(len(result.dtw_history), iterations)
        self.assertEqual(result.mode_history[0], "base")
        raw_rows = [row for row in result.dtw_history if row.get("raw_fire")]
        self.assertEqual(result.fire_count, len(raw_rows))
        self.assertEqual(
            [event["detected_iteration"] for event in result.control_events if event.get("raw_fire")],
            [row["iteration"] for row in raw_rows],
        )
        for index, row in enumerate(result.parameter_history):
            self.assertEqual(row["iteration"], index)
            self.assertAlmostEqual(row["woa_a_base"], 2.0 - 2.0 * index / (iterations - 1))


def _signal_history() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(12):
        ready = index >= 4 and index not in {6, 7, 8, 9, 10}
        rows.append({
            "iteration": index,
            "ready": ready,
            "fire": False,
            "raw_fire": index in {5, 11},
            "cond_plateau": True,
            "cond_constant": True,
            "cond_ramp": True,
            "D2_vs_const": 0.2,
            "theta_c": 1.0,
            "delta": 1.5,
            "theta_delta": 1.0,
        })
    return rows


def _parameter_row(index: int) -> dict[str, float]:
    return {
        "iteration": index,
        "woa_a_base": 1.0,
        "woa_a_effective": 1.0,
        "woa_a_scale": 1.0,
        "abc_phi_scale": 1.0,
        "abc_phi_effective": 1.0,
        "abc_limit_factor": 1.0,
        "abc_limit_effective": 2.0,
        "abc_guide_strength": 0.2,
        "lb2_g1": 0.5,
        "lb2_g2": 0.5,
        "lb2_g3": 0.5,
    }


def _vertical_event_positions(figures: list[matplotlib.figure.Figure]) -> list[float]:
    """X de las fronteras del gráfico copiado, no triángulos de convergencia.

    La frontera en 0 es el segmento inicial de la fuente. No es un fire.
    """

    positions: list[float] = []
    for figure in figures:
        for axis in figure.axes:
            for line in axis.get_lines():
                if line.get_linestyle() not in {"--", "dashed"}:
                    continue
                xdata = np.asarray(line.get_xdata(), dtype=float)
                ydata = np.asarray(line.get_ydata(), dtype=float)
                if xdata.size < 2 or not np.allclose(xdata, xdata[0]):
                    continue
                if ydata.size >= 2 and np.allclose(ydata, ydata[0]):
                    continue
                positions.append(float(xdata[0]))
    return positions


class TestFirePlotMarkers(unittest.TestCase):
    def test_public_plotters_mark_confirmed_events_not_conditions(self) -> None:
        history = _signal_history()
        continuous = OptimizationResult(
            problem="toy",
            variant=M1_FIRE_D2,
            seed=1,
            best_cost=1.0,
            best_position=[0.0],
            objective_evaluations=1,
            fire_count=2,
            transition_count=0,
            parameter_update_count=0,
            convergence_history=[1.0] * 12,
            woa_history=[1.0] * 12,
            abc_history=[1.0] * 12,
            mode_history=["exploit"] * 12,
            intensity_history=[0.0] * 12,
            dtw_history=history,
            parameter_history=[_parameter_row(index) for index in range(12)],
        )
        binary = MKPOptimizationResult(
            problem="toy",
            variant=M5_SIGMOID_DELTA,
            seed=1,
            best_profit=1.0,
            best_solution=[0, 0],
            best_known=None,
            objective_evaluations=1,
            iterations_completed=12,
            optimum_reached=False,
            stop_reason="iteration_budget",
            fire_count=2,
            transition_count=0,
            parameter_update_count=0,
            local_search_improvements=0,
            convergence_history=[1.0] * 12,
            woa_history=[1.0] * 12,
            abc_history=[1.0] * 12,
            mode_history=["exploit"] * 12,
            intensity_history=[0.0] * 12,
            dtw_history=history,
            parameter_history=[_parameter_row(index) for index in range(12)],
        )
        with tempfile.TemporaryDirectory(prefix="woa_fire_plots_") as temporary:
            root = Path(temporary)
            self.assertEqual(continuous.fire_count, 2)
            self.assertEqual(
                [row["iteration"] for row in history if row.get("raw_fire")],
                [5, 11],
            )
            self.assertEqual(self._markers(save_run_plots, root / "cec", continuous, 1.0), [0.0, 5.0, 11.0])
            self.assertEqual(self._markers(save_mkp_run_plots, root / "mkp", binary), [0.0, 5.0, 11.0])

    def test_live_optimizer_plot_markers_match_fire_count(self) -> None:
        result = WOAABCOptimizer(
            _constant_problem(),
            _algorithm(M2_FIRE_3COND, 12),
        ).run()
        with tempfile.TemporaryDirectory(prefix="woa_fire_live_") as temporary:
            markers = self._markers(save_run_plots, Path(temporary), result, 1.0)
        self.assertEqual(markers, [0.0, 5.0, 11.0])
        self.assertEqual([value for value in markers if value != 0.0], [5.0, 11.0])
        self.assertEqual(len([value for value in markers if value != 0.0]), result.fire_count)
        self.assertEqual(result.fire_count, 2)

    def _markers(self, plotter: object, directory: Path, result: object, optimum: float | None = None) -> list[float]:
        captured: list[matplotlib.figure.Figure] = []
        seen: set[int] = set()
        original = matplotlib.figure.Figure.savefig

        def spy(figure: matplotlib.figure.Figure, *args: object, **kwargs: object) -> object:
            # Count each figure once even if the save implementation changes.
            if id(figure) not in seen:
                seen.add(id(figure))
                captured.append(figure)
            return original(figure, *args, **kwargs)

        with patch.object(matplotlib.figure.Figure, "savefig", spy):
            if optimum is None:
                plotter(directory, result)
            else:
                plotter(directory, result, optimum)
            positions = _vertical_event_positions(captured)
        self.assertTrue((directory / "dtw_adaptation.pdf").is_file())
        self.assertTrue((directory / "parameters.pdf").is_file())
        self.assertEqual(list(directory.glob("*.png")), [])
        for pdf in directory.glob("*.pdf"):
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"), pdf.name)
        return positions


if __name__ == "__main__":
    unittest.main()
