"""Gráfico temporal MKP copiado de a/temporal_diagnostics.py, sin su runner."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from woa_abc_version_vieja.config import DTWConfig
from woa_abc_version_vieja.core.binary_woa_abc import MKPOptimizationResult
from woa_abc_version_vieja.domains.mkp import MKPInstance
from woa_abc_version_vieja.dtw.monitor import StagnationMonitor, ddtw_distance, dtw_distance
from woa_abc_version_vieja.reporting.mkp import save_mkp_run_artifacts, save_mkp_run_plots
from woa_abc_version_vieja.reporting.temporal_diagnostics import (
    draw_temporal_diagnostics,
    load_saved_mkp_run,
    reconstruct_temporal_series,
    save_temporal_diagnostics,
)
from woa_abc_version_vieja.runners.run_mkp import (
    _InstanceTask,
    _run_instance_task,
    parser as mkp_parser,
)


PRIOR_PDFS = (
    "convergence.pdf",
    "dtw_adaptation.pdf",
    "parameters.pdf",
    "lb2_parameters.pdf",
)
PAPER_WINDOW = 200
PAPER_SLOPE = 2.0


def _configuration(dtw: DTWConfig, variant: str, seed: int = 1) -> dict:
    return {
        "algorithm": {
            "variant": variant,
            "seed": seed,
            "dtw": asdict(dtw),
        }
    }


def _parameter_row(index: int, a_eff: float, phi_eff: float) -> dict:
    return {
        "iteration": index,
        "woa_a_base": 1.0,
        "woa_a_effective": a_eff,
        "abc_phi_effective": phi_eff,
        "abc_limit_effective": 2,
        "abc_guide_strength": 0.2,
        "lb2_g1": 0.5,
        "lb2_g2": 0.25,
        "lb2_g3": 0.75,
    }


def _result(
    values: list[float],
    rows: list[dict],
    modes: list[str],
    variant: str,
    parameters: list[dict] | None = None,
) -> MKPOptimizationResult:
    length = len(values)
    if parameters is None:
        parameters = [_parameter_row(index, 0.8, 0.4) for index in range(length)]
    return MKPOptimizationResult(
        problem="toy",
        variant=variant,
        seed=1,
        best_profit=float(values[-1]) if values else 0.0,
        best_solution=[0, 1],
        best_known=100.0,
        objective_evaluations=length,
        iterations_completed=length,
        optimum_reached=False,
        stop_reason="iteration_budget",
        fire_count=sum(bool(row.get("fire")) for row in rows),
        transition_count=0,
        parameter_update_count=0,
        local_search_improvements=0,
        convergence_history=list(values),
        woa_history=list(values),
        abc_history=list(values),
        mode_history=list(modes),
        intensity_history=[None] * length,
        dtw_history=rows,
        parameter_history=parameters,
        control_events=[],
    )


def _drive(
    values: list[float],
    dtw: DTWConfig,
    modes: list[str] | None = None,
    variant: str = "M2_fire_3cond",
    parameters: list[dict] | None = None,
    sentinel: float = 999.0,
) -> tuple[MKPOptimizationResult, dict, list[dict]]:
    monitor = StagnationMonitor(dtw)
    applied = list(modes or ["exploit"] * len(values))
    rows: list[dict] = []
    for index, value in enumerate(values):
        status = dict(monitor.update(float(value)))
        status["iteration"] = index
        status["applied_mode"] = applied[index]
        if not status.get("ready"):
            status["D1_vs_ramp"] = sentinel
            status["D2_vs_const"] = sentinel
            status["delta"] = sentinel
            status["theta_r"] = sentinel
            status["theta_c"] = sentinel
            status["theta_delta"] = sentinel
        rows.append(status)
    configuration = _configuration(dtw, variant)
    return _result(values, rows, applied, variant, parameters), configuration, rows


def _expected_references(
    values: list[float],
    index: int,
    window: int,
    min_slope: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(values[index - window + 1 : index + 1], dtype=float)
    slope = float(min_slope)
    if slope == 0.0:
        slope = 0.01 * max(1.0, abs(float(observed[-1] - observed[0]))) / window
    ramp = float(observed[0]) + slope * np.arange(window, dtype=float)
    constant = np.full(window, float(observed[0]), dtype=float)
    return observed, ramp, constant


def _axis_title(axis: plt.Axes) -> str:
    """Source titles are left-aligned; the center title stays empty."""

    return axis.get_title(loc="left") or axis.get_title()


def _texts(figure: plt.Figure) -> str:
    chunks: list[str] = []
    for axis in figure.axes:
        chunks.append(_axis_title(axis))
        chunks.append(axis.get_xlabel())
        chunks.append(axis.get_ylabel())
        chunks.extend(text.get_text() for text in axis.texts)
        chunks.extend(text.get_text() for text in axis.get_xticklabels())
        chunks.extend(text.get_text() for text in axis.get_yticklabels())
        legend = axis.get_legend()
        if legend is not None:
            chunks.extend(text.get_text() for text in legend.get_texts())
    return "\n".join(chunks)


class TemporalSeriesTests(unittest.TestCase):
    def test_ready_windows_match_monitor_and_do_not_cross_reset(self) -> None:
        dtw = DTWConfig(
            window=6,
            band=2,
            min_slope=1.5,
            plateau_max=2,
            patience=2,
            use_ddtw=True,
            adapt_thresholds=True,
        )
        values = [10.0] * 20
        result, configuration, rows = _drive(values, dtw, modes=["exploit"] * 20)
        result.mode_history[7] = "explore"
        before_values = list(result.convergence_history)
        before_rows = [dict(row) for row in result.dtw_history]
        before_config = deepcopy(configuration)

        series = reconstruct_temporal_series(result, configuration)

        self.assertEqual(result.convergence_history, before_values)
        self.assertEqual(result.dtw_history, before_rows)
        self.assertEqual(configuration, before_config)
        self.assertEqual(series["window"], 6)
        self.assertEqual(series["min_slope"], 1.5)
        self.assertNotEqual(series["window"], PAPER_WINDOW)
        self.assertEqual(series["R"].shape, (20, 6))
        distance = ddtw_distance
        fires = [index for index, row in enumerate(rows) if row.get("fire")]
        self.assertTrue(fires)
        first_ready = next(index for index, row in enumerate(rows) if row.get("ready"))
        self.assertEqual(first_ready, dtw.window - 1)
        self.assertGreaterEqual(int(series["episode_start"][first_ready]), 0)
        for index, row in enumerate(rows):
            if not row.get("ready"):
                self.assertTrue(np.all(np.isnan(series["R"][index])))
                self.assertTrue(np.all(np.isnan(series["C"][index])))
                self.assertTrue(np.isnan(series["D_R"][index]))
                self.assertTrue(np.isnan(series["D_C"][index]))
                self.assertTrue(np.isnan(series["theta_R"][index]))
                self.assertTrue(np.isnan(series["theta_C"][index]))
                self.assertTrue(np.isnan(series["theta_delta"][index]))
                self.assertNotEqual(series["D_R"][index], 999.0)
                continue
            observed, ramp, constant = _expected_references(values, index, dtw.window, dtw.min_slope)
            self.assertTrue(np.allclose(series["X"][index], observed))
            self.assertTrue(np.allclose(series["R"][index], ramp))
            self.assertTrue(np.allclose(series["C"][index], constant))
            self.assertAlmostEqual(series["D_R"][index], distance(observed, ramp, dtw.effective_band))
            self.assertAlmostEqual(series["D_C"][index], distance(observed, constant, dtw.effective_band))
            self.assertAlmostEqual(series["D_R"][index], float(row["D1_vs_ramp"]))
            self.assertAlmostEqual(series["D_C"][index], float(row["D2_vs_const"]))
            start = int(series["episode_start"][index])
            self.assertGreaterEqual(index - dtw.window + 1, start)
            self.assertTrue(np.all(series["episode_start"][index - dtw.window + 1 : index + 1] == start))
        fire_index = fires[0]
        self.assertTrue(np.isfinite(series["D_R"][fire_index]))
        self.assertTrue(np.all(np.isnan(series["R"][fire_index + 1])))
        self.assertEqual(int(series["episode_start"][fire_index + 1]), fire_index + 1)
        self.assertFalse(rows[fire_index + 1]["ready"])
        next_ready = next(index for index in range(fire_index + 1, len(rows)) if rows[index]["ready"])
        self.assertGreaterEqual(next_ready - dtw.window + 1, fire_index + 1)
        self.assertEqual(result.mode_history[fire_index + 1], "explore")
        for key in ("progress_index", "plateau_index"):
            chosen = series[key]
            if chosen is None:
                continue
            self.assertTrue(series["ready"][chosen])
            self.assertGreaterEqual(chosen - dtw.window + 1, int(series["episode_start"][chosen]))

    def test_actual_window_slope_and_band_are_not_replaced(self) -> None:
        cases = (
            DTWConfig(window=70, band=0, min_slope=0.1, plateau_max=15, patience=8, use_ddtw=True),
            DTWConfig(window=40, band=3, min_slope=0.0, plateau_max=4, patience=3, use_ddtw=False),
        )
        for dtw in cases:
            with self.subTest(window=dtw.window, slope=dtw.min_slope, ddtw=dtw.use_ddtw):
                values = [1.0, 1.0, 2.5, 2.5, 4.0]
                values.extend([4.0] * (dtw.window + 3))
                result, configuration, rows = _drive(values, dtw, variant="M4_d2_continuous")
                before = deepcopy(configuration)
                series = reconstruct_temporal_series(result, configuration)
                self.assertEqual(configuration, before)
                self.assertEqual(series["window"], dtw.window)
                self.assertEqual(series["min_slope"], dtw.min_slope)
                self.assertEqual(series["band"], dtw.band)
                self.assertEqual(series["effective_band"], dtw.effective_band)
                self.assertNotEqual(series["window"], PAPER_WINDOW)
                self.assertNotEqual((series["window"], series["min_slope"]), (20, PAPER_SLOPE))
                distance = ddtw_distance if dtw.use_ddtw else dtw_distance
                ready = [index for index, row in enumerate(rows) if row["ready"]]
                self.assertTrue(ready)
                index = ready[0]
                observed, ramp, constant = _expected_references(values, index, dtw.window, dtw.min_slope)
                self.assertTrue(np.allclose(series["R"][index], ramp))
                self.assertTrue(np.allclose(series["C"][index], constant))
                self.assertAlmostEqual(
                    float(series["D_R"][index]),
                    distance(observed, ramp, dtw.effective_band),
                )
                self.assertAlmostEqual(float(series["D_R"][index]), float(rows[index]["D1_vs_ramp"]))

    def test_stored_distance_mismatch_raises_without_writing(self) -> None:
        dtw = DTWConfig(window=5, band=1, min_slope=0.4, plateau_max=9, patience=9)
        values = [float(index) for index in range(7)]
        result, configuration, _rows = _drive(values, dtw)
        for row in result.dtw_history:
            if row.get("ready"):
                row["D1_vs_ramp"] = float(row["D1_vs_ramp"]) + 1.0
                break
        else:
            self.fail("fixture produced no ready row")
        with tempfile.TemporaryDirectory(prefix="woa_temporal_mismatch_") as temporary:
            with self.assertRaises(RuntimeError) as caught:
                save_temporal_diagnostics(Path(temporary), result, configuration)
            self.assertIn("D1", str(caught.exception))
            self.assertEqual(list(Path(temporary).glob("*")), [])

    def test_m0_skips_figure_and_does_not_invent_series(self) -> None:
        dtw = DTWConfig(window=70, band=0, min_slope=0.1)
        result, configuration, _rows = _drive([1.0, 2.0, 3.0], dtw, variant="M0_no_dtw")
        result.dtw_history = [
            {"iteration": index, "ready": False, "fire": False, "n": index + 1}
            for index in range(3)
        ]
        self.assertIsNone(reconstruct_temporal_series(result, configuration))
        self.assertIsNone(reconstruct_temporal_series(result, {}))
        self.assertIsNone(draw_temporal_diagnostics(result, configuration))
        with tempfile.TemporaryDirectory(prefix="woa_temporal_m0_") as temporary:
            run_dir = Path(temporary)
            self.assertIsNone(save_temporal_diagnostics(run_dir, result, configuration))
            save_mkp_run_plots(run_dir, result, configuration)
            self.assertEqual(sorted(path.name for path in run_dir.iterdir()), sorted(PRIOR_PDFS))
            self.assertFalse((run_dir / "temporal_diagnostics.pdf").exists())

    def test_missing_config_does_not_guess_a_window(self) -> None:
        dtw = DTWConfig(window=40, band=1, min_slope=0.0)
        result, _configuration, _rows = _drive([1.0] * 45, dtw, variant="M5_sigmoid_delta")
        with self.assertRaises(ValueError):
            reconstruct_temporal_series(result, {"algorithm": {}})


class TemporalFigureTests(unittest.TestCase):
    def test_early_ready_windows_are_selectable(self) -> None:
        dtw = DTWConfig(window=5, band=1, min_slope=0.2, plateau_max=99, patience=99)
        values = [float(index) for index in range(8)]
        result, configuration, rows = _drive(values, dtw, modes=["exploit"] * 8)
        series = reconstruct_temporal_series(result, configuration)
        ready = [index for index, row in enumerate(rows) if row["ready"]]
        self.assertTrue(ready)
        self.assertTrue(all(not series["adaptive_threshold"][index] for index in ready))
        self.assertIsNotNone(series["progress_index"])
        self.assertTrue(np.isfinite(series["R"][ready[0]]).all())

    def test_missing_representative_windows_are_annotated(self) -> None:
        short_dtw = DTWConfig(window=8, band=1, min_slope=0.1, plateau_max=9, patience=9)
        short, short_config, _rows = _drive([1.0, 1.0, 2.0], short_dtw)
        short_figure = draw_temporal_diagnostics(short, short_config)
        self.assertIsNotNone(short_figure)
        short_text = _texts(short_figure)
        self.assertGreaterEqual(short_text.count("Representative panel unavailable"), 2)
        self.assertEqual(len(short_figure.axes), 6)
        for axis in short_figure.axes:
            title = _axis_title(axis)
            if title.startswith("(b)") or title.startswith("Flat"):
                self.assertEqual(axis.get_lines(), [])
        plt.close(short_figure)

        flat_dtw = DTWConfig(window=5, band=1, min_slope=0.3, plateau_max=99, patience=99)
        flat, flat_config, _rows = _drive([4.0] * 9, flat_dtw, modes=["exploit"] * 9)
        flat_series = reconstruct_temporal_series(flat, flat_config)
        flat_figure = draw_temporal_diagnostics(flat, flat_config)
        flat_text = _texts(flat_figure)
        self.assertIsNotNone(flat_series["plateau_index"])
        self.assertIsNone(flat_series["progress_index"])
        self.assertIn("Representative panel unavailable", flat_text)
        self.assertIn("exploit", flat_series["plateau_title"])
        self.assertNotIn("Exploration", flat_series["plateau_title"])
        self.assertNotIn("BDE", flat_text)
        plt.close(flat_figure)

        rising_dtw = DTWConfig(window=4, band=1, min_slope=0.25, plateau_max=99, patience=99)
        rising_values = [float(index) for index in range(7)]
        rising, rising_config, _rows = _drive(
            rising_values,
            rising_dtw,
            modes=["explore"] * 7,
            variant="M5_sigmoid_delta",
        )
        rising_series = reconstruct_temporal_series(rising, rising_config)
        rising_figure = draw_temporal_diagnostics(rising, rising_config)
        self.assertIsNotNone(rising_series["progress_index"])
        self.assertIsNone(rising_series["plateau_index"])
        self.assertIn("explore", rising_series["progress_title"])
        self.assertIn("Representative panel unavailable", _texts(rising_figure))
        plt.close(rising_figure)

    def test_states_and_actual_parameters_are_not_bde_labels(self) -> None:
        dtw = DTWConfig(window=4, band=1, min_slope=0.2, plateau_max=99, patience=99)
        modes = [
            "base",
            "base",
            "exploit",
            "exploit_high",
            "exploit_low",
            "explore",
            "explore_low",
            "explore_high",
            "base",
        ]
        parameters = [
            _parameter_row(index, 0.25 if index % 2 == 0 else 1.75, 0.3 if index < 4 else 1.1)
            for index in range(len(modes))
        ]
        result, configuration, _rows = _drive(
            [3.0] * len(modes),
            dtw,
            modes=modes,
            variant="M8_four_state_woa_abc",
            parameters=parameters,
        )
        figure = draw_temporal_diagnostics(result, configuration)
        text = _texts(figure)
        for label in ("base", "exploit_high", "exploit_low", "explore_low", "explore_high"):
            self.assertIn(label, text)
        self.assertNotIn("BDE", text)
        self.assertNotIn("F=", text)
        self.assertNotIn("CR=", text)
        self.assertNotIn("0.5, CR", text)
        self.assertIn("0.25", text)
        self.assertIn("1.75", text)
        self.assertIn("a_eff", text)
        self.assertIn("phi_eff", text)
        ytick = [tick.get_text() for tick in figure.axes[-1].get_yticklabels()]
        self.assertNotEqual(ytick, ["0", "1"])
        self.assertNotIn("0", ytick)
        colors = {
            tuple(np.round(to_rgb(patch.get_facecolor()), 5))
            for axis in figure.axes
            for patch in axis.patches
        }
        self.assertIn(tuple(np.round(to_rgb("#B0B0B0"), 5)), colors)
        self.assertIn(tuple(np.round(to_rgb("#8ECAE6"), 5)), colors)
        self.assertIn(tuple(np.round(to_rgb("#F4A261"), 5)), colors)
        self.assertEqual(len(figure.axes), 6)
        plt.close(figure)

    def test_layout_palette_and_single_pdf(self) -> None:
        dtw = DTWConfig(window=5, band=1, min_slope=0.2, plateau_max=99, patience=99)
        values = [float(index) for index in range(6)] + [5.0] * 8
        modes = ["exploit"] * 6 + ["explore"] * 8
        result, configuration, _rows = _drive(values, dtw, modes=modes, variant="M2_fire_3cond")
        before_profit = result.best_profit
        figure = draw_temporal_diagnostics(result, configuration)
        self.assertEqual(tuple(np.round(figure.get_size_inches(), 5)), (7.4, 8.6))
        self.assertEqual(len(figure.axes), 6)
        self.assertTrue(_axis_title(figure.axes[0]).startswith("(a)"))
        self.assertTrue(any(_axis_title(axis).startswith("(b)") for axis in figure.axes))
        self.assertTrue(any(_axis_title(axis).startswith("(c)") for axis in figure.axes))
        self.assertTrue(any(_axis_title(axis).startswith("(d)") for axis in figure.axes))
        text = _texts(figure)
        self.assertNotIn("BDE", text)
        self.assertNotIn("F=", text)
        self.assertNotIn("CR=", text)
        window_axis = next(axis for axis in figure.axes if _axis_title(axis).startswith("(b)"))
        self.assertNotEqual(tuple(np.round(window_axis.get_ylim(), 5)), (-30.0, 460.0))
        plotted = np.concatenate([
            np.asarray(line.get_ydata(), dtype=float)
            for line in window_axis.get_lines()
        ])
        finite = plotted[np.isfinite(plotted)]
        lower, upper = window_axis.get_ylim()
        self.assertLessEqual(lower, float(np.min(finite)))
        self.assertGreaterEqual(upper, float(np.max(finite)))
        delta_axis = next(axis for axis in figure.axes if _axis_title(axis).startswith("(d)"))
        delta_lower, delta_upper = delta_axis.get_ylim()
        self.assertLess(delta_upper - delta_lower, 8.0)
        colors = {
            tuple(np.round(to_rgb(line.get_color()), 5))
            for axis in figure.axes
            for line in axis.get_lines()
        }
        for hex_color in ("#0072B2", "#D55E00", "#009E73"):
            self.assertIn(tuple(np.round(to_rgb(hex_color), 5)), colors)
        titled = [
            text
            for text in figure.findobj(match=lambda item: hasattr(item, "get_text"))
            if str(text.get_text()).startswith("(a)")
        ]
        self.assertEqual(titled[0].get_fontfamily(), ["serif"])
        self.assertNotIn("Intensidad", text)
        self.assertNotIn("intensity", text.lower())
        plt.close(figure)
        with tempfile.TemporaryDirectory(prefix="woa_temporal_pdf_") as temporary:
            run_dir = Path(temporary)
            saved = save_temporal_diagnostics(run_dir, result, configuration)
            self.assertEqual(saved, run_dir / "temporal_diagnostics.pdf")
            self.assertEqual([path.name for path in run_dir.iterdir()], ["temporal_diagnostics.pdf"])
            self.assertTrue(saved.read_bytes().startswith(b"%PDF"))
            self.assertGreater(saved.stat().st_size, 1000)
        self.assertEqual(result.best_profit, before_profit)

    def test_generic_variants_render_without_a_campaign(self) -> None:
        dtw = DTWConfig(window=4, band=1, min_slope=0.1, plateau_max=99, patience=99)
        values = [1.0, 1.2, 1.2, 1.4, 1.4, 1.4]
        for variant in (
            "M1_fire_d2",
            "M2_fire_3cond",
            "M4_d2_continuous",
            "M5_sigmoid_delta",
            "M6_hysteresis_woa_abc",
            "M8_four_state_woa_abc",
        ):
            with self.subTest(variant=variant):
                result, configuration, _rows = _drive(values, dtw, variant=variant)
                figure = draw_temporal_diagnostics(result, configuration)
                self.assertEqual(len(figure.axes), 6)
                self.assertNotIn("BDE", _texts(figure))
                plt.close(figure)


class TemporalSaveAndRunnerTests(unittest.TestCase):
    def _toy(self) -> MKPOptimizationResult:
        dtw = DTWConfig(window=4, band=1, min_slope=0.2, plateau_max=9, patience=9)
        result, _configuration, _rows = _drive([1.0, 1.0, 2.0, 2.0, 2.0], dtw)
        return result

    def test_omitted_configuration_keeps_the_previous_four_pdfs(self) -> None:
        result = self._toy()
        captured: dict[str, dict[str, object]] = {}
        original = plt.Figure.savefig

        def spy(self, *args, **kwargs):
            stem = Path(args[0]).stem
            axis = self.axes[0]
            captured[stem] = {
                "n_axes": len(self.axes),
                "title": axis.get_title(),
                "xlabel": axis.get_xlabel(),
                "ylabel": axis.get_ylabel(),
                "size": [round(float(value), 5) for value in self.get_size_inches()],
                "line_labels": [line.get_label() for line in axis.get_lines()],
            }
            return original(self, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="woa_temporal_prior_") as temporary:
            plt.Figure.savefig = spy
            try:
                save_mkp_run_plots(Path(temporary), result)
            finally:
                plt.Figure.savefig = original
            self.assertEqual(
                sorted(path.name for path in Path(temporary).iterdir()),
                sorted(PRIOR_PDFS),
            )
        self.assertEqual(captured["convergence"]["title"], "Convergencia MKP — M2_fire_3cond")
        self.assertEqual(captured["convergence"]["ylabel"], "Beneficio")
        self.assertEqual(captured["convergence"]["size"], [10.0, 5.5])
        self.assertEqual(
            captured["dtw_adaptation"]["title"],
            "WOA-ABC DTW Delta per Iteration  [+ = stagnation | - = active improvement]",
        )
        self.assertEqual(captured["parameters"]["n_axes"], 4)
        self.assertEqual(captured["lb2_parameters"]["title"], "Calendario LB2 (independiente de DTW)")
        self.assertEqual(captured["lb2_parameters"]["line_labels"], ["G1", "G2", "G3"])

    def test_configuration_adds_only_the_temporal_pdf(self) -> None:
        result = self._toy()
        configuration = _configuration(DTWConfig(window=4, band=1, min_slope=0.2), result.variant)
        before = list(result.convergence_history)
        with tempfile.TemporaryDirectory(prefix="woa_temporal_extra_") as temporary:
            run_dir = Path(temporary)
            with patch(
                "woa_abc_version_vieja.reporting.mkp.save_temporal_diagnostics",
                return_value=run_dir / "temporal_diagnostics.pdf",
            ) as save_temporal:
                save_mkp_run_plots(run_dir, result, configuration)
            save_temporal.assert_called_once()
            self.assertIs(save_temporal.call_args.args[1], result)
            self.assertEqual(save_temporal.call_args.args[2], configuration)
            self.assertEqual(sorted(path.name for path in run_dir.iterdir()), sorted(PRIOR_PDFS))
        self.assertEqual(result.convergence_history, before)

    def test_saved_run_reloads_without_an_optimizer(self) -> None:
        dtw = DTWConfig(window=5, band=1, min_slope=0.3, plateau_max=9, patience=9)
        values = [2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0]
        result, configuration, _rows = _drive(values, dtw, variant="M2_fire_3cond")
        instance = MKPInstance(
            family="toy",
            index=0,
            profits=np.array([1.0, 1.0]),
            weights=np.array([[1.0, 1.0]]),
            capacities=np.array([5.0]),
            best_known=4.0,
        )
        summary = result.summary()
        summary["stop_reason"] = result.stop_reason
        with tempfile.TemporaryDirectory(prefix="woa_temporal_load_") as temporary:
            run_dir = Path(temporary)
            saved_dir = run_dir / "run_01"
            save_mkp_run_artifacts(saved_dir, result, instance, summary, configuration)
            loaded, loaded_configuration = load_saved_mkp_run(saved_dir)
            direct = reconstruct_temporal_series(result, configuration)
            restored = reconstruct_temporal_series(loaded, loaded_configuration)
            self.assertEqual(loaded.variant, "M2_fire_3cond")
            self.assertEqual(loaded_configuration["algorithm"]["dtw"]["window"], 5)
            self.assertTrue(np.allclose(restored["D_R"], direct["D_R"], equal_nan=True))
            self.assertTrue(np.allclose(restored["R"], direct["R"], equal_nan=True))
            self.assertEqual(list(run_dir.glob("*.png")), [])
            self.assertEqual(list(run_dir.glob("*.npz")), [])

    def test_runner_passes_actual_configuration_without_a_campaign(self) -> None:
        args = mkp_parser().parse_args([
            "--runs", "2",
            "--variants", "M0", "M2",
            "--window", "40",
            "--band", "3",
            "--min-slope", "0",
            "--iterations", "4",
            "--pop-size", "4",
            "--seed", "7",
            "--no-statistical-analysis",
        ])
        instance = MKPInstance(
            family="toy",
            index=0,
            profits=np.array([1.0]),
            weights=np.array([[1.0]]),
            capacities=np.array([5.0]),
        )

        class FakeOptimizer:
            def __init__(self, problem: MKPInstance, config: object) -> None:
                self.config = config

            def run(self, verbose: bool = False) -> MKPOptimizationResult:
                variant = self.config.algorithm.variant
                seed = self.config.algorithm.seed
                result, _configuration, _rows = _drive(
                    [1.0, 1.0, 1.0, 1.0],
                    self.config.algorithm.dtw,
                    variant=variant,
                )
                result.seed = seed
                result.best_profit = float(seed)
                result.best_solution = [0]
                return result

        with tempfile.TemporaryDirectory(prefix="woa_temporal_runner_") as temporary:
            root = Path(temporary)
            common = dict(
                instance=instance,
                args=args,
                variants=("M0_no_dtw", "M2_fire_3cond"),
                statistics_enabled=False,
            )
            with patch(
                "woa_abc_version_vieja.runners.run_mkp.BinaryWOAABCOptimizer",
                FakeOptimizer,
            ), patch(
                "woa_abc_version_vieja.runners.run_mkp.save_mkp_run_plots",
            ) as plots, patch(
                "woa_abc_version_vieja.runners.run_mkp.save_mkp_run_artifacts",
            ) as artifacts:
                _run_instance_task(_InstanceTask(
                    instance_dir=root / "all",
                    keep_all_details=True,
                    **common,
                ))
                per_run = plots.call_args_list
                self.assertEqual(len(per_run), 4)
                for call in per_run:
                    run_dir, result, configuration = call.args
                    self.assertIn("run_0", run_dir.name)
                    self.assertEqual(configuration["algorithm"]["variant"], result.variant)
                    self.assertEqual(configuration["algorithm"]["seed"], result.seed)
                    self.assertEqual(configuration["algorithm"]["dtw"]["window"], 40)
                    self.assertEqual(configuration["algorithm"]["dtw"]["min_slope"], 0.0)
                    self.assertEqual(configuration["algorithm"]["dtw"]["band"], 3)
                self.assertEqual(artifacts.call_count, 4)
                plots.reset_mock()
                artifacts.reset_mock()
                _run_instance_task(_InstanceTask(
                    instance_dir=root / "best",
                    keep_all_details=False,
                    **common,
                ))
                self.assertEqual(plots.call_count, 2)
                seeds = []
                for call in plots.call_args_list:
                    run_dir, result, configuration = call.args
                    self.assertEqual(run_dir.name, "mejor_run")
                    self.assertEqual(configuration["algorithm"]["variant"], result.variant)
                    self.assertEqual(configuration["algorithm"]["dtw"]["window"], 40)
                    self.assertEqual(configuration["algorithm"]["seed"], result.seed)
                    seeds.append(result.seed)
                    artifact_configuration = artifacts.call_args_list[len(seeds) - 1].args[4]
                    self.assertEqual(artifact_configuration, configuration)
                self.assertEqual(seeds, [8, 8])
        self.assertNotIn("temporal", mkp_parser().format_help().lower())

    def test_source_style_is_copied_without_importing_the_script(self) -> None:
        original_path = Path("a/temporal_diagnostics.py")
        original = original_path.read_text(encoding="utf-8")
        tree = ast.parse(original)
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("plot_figure", names)
        self.assertIn("_window_end", names)
        module_path = Path("woa_abc_version_vieja/reporting/temporal_diagnostics.py")
        copied = module_path.read_text(encoding="utf-8")
        self.assertIn("a/temporal_diagnostics.py", copied)
        self.assertIn("figsize=(7.4, 8.6)", copied)
        self.assertIn("height_ratios=[1.15, 1.05, 1.05, 1.35]", copied)
        self.assertIn("height_ratios=[1.45, 0.85]", copied)
        self.assertIn('"mathtext.fontset": "stix"', copied)
        for token in ("#0072B2", "#D55E00", "#009E73", "#F4A261", "#8ECAE6"):
            self.assertIn(token, copied)
        self.assertNotIn("sys.path.insert", copied)
        self.assertNotIn("run_representative", copied)
        self.assertNotIn("savez", copied)
        self.assertNotIn(".png", copied)
        self.assertNotIn("a.temporal_diagnostics", " ".join(__import__("sys").modules))


if __name__ == "__main__":
    unittest.main()
