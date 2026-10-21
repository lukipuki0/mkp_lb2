"""Panel DTW copiado de HRES2-H2/plots/dtw_delta.py, no un restyle."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import math
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np
from matplotlib.colors import to_rgb

from woa_abc_version_vieja.core.binary_woa_abc import MKPOptimizationResult
from woa_abc_version_vieja.core.woa_abc import OptimizationResult
from woa_abc_version_vieja.reporting.mkp import draw_mkp_dtw_adaptation, save_mkp_run_plots
from woa_abc_version_vieja.reporting.plots import draw_dtw_adaptation, save_run_plots


VARIANT_COLOR = np.asarray([0x1F / 255.0, 0x77 / 255.0, 0xB4 / 255.0])
REFERENCE_TITLE = (
    "HRES2-H2 DTW Delta per Iteration  [+ = stagnation | - = active improvement]"
)
WOA_TITLE = REFERENCE_TITLE.replace("HRES2-H2", "WOA-ABC", 1)
LENGTH = 12
READY = (4, 5, 9, 10, 11)
NOT_READY_SENTINEL = 7.0
D2_VALUES = {4: 0.5, 5: 0.25, 9: 0.125, 10: 4.0, 11: 1.0}
DELTA_VALUES = {4: 3.0, 5: -1.25, 9: 0.5, 10: float("inf"), 11: -2.0}
VALID_X = (4, 5, 9, 11)
# 0 is the source initial segment boundary, not a confirmed fire.
FIRE_INDICES = (5, 11)
INITIAL_BOUNDARY = 0.0
CONVERGENCE = [float(12 - index) for index in range(LENGTH)]
ADAPTIVE = (
    "M1_fire_d2",
    "M2_fire_3cond",
    "M4_d2_continuous",
    "M5_sigmoid_delta",
    "M6_hysteresis_woa_abc",
    "M8_four_state_woa_abc",
)
NEGATIVE_EXAMPLES = (
    "M1_fire_d2",
    "M2_fire_3cond",
    "M4_d2_continuous",
    "M5_sigmoid_delta",
    "M6_hysteresis_woa_abc",
)
VARIANTS = ("M0_no_dtw",) + ADAPTIVE


def _parameter_row(index: int) -> dict[str, float]:
    return {
        "iteration": float(index),
        "woa_a_base": 1.0,
        "woa_a_effective": 1.25,
        "woa_a_scale": 1.0,
        "abc_phi_scale": 0.5,
        "abc_phi_effective": 0.5,
        "abc_limit_factor": 1.0,
        "abc_limit_effective": 2.0,
        "abc_guide_strength": 0.2,
        "lb2_g1": 0.5,
        "lb2_g2": 0.25,
        "lb2_g3": 0.75,
    }


def _history(*, monitor: bool, fires: tuple[int, ...] = FIRE_INDICES) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(LENGTH):
        ready = monitor and index in READY
        rows.append({
            "iteration": index,
            "ready": ready,
            "fire": False,
            "raw_fire": monitor and index in fires,
            "cond_plateau": True,
            "cond_constant": True,
            "cond_ramp": True,
            "D1_vs_ramp": 9.0,
            "D2_vs_const": D2_VALUES[index] if ready else NOT_READY_SENTINEL,
            "theta_c": 1.0 if ready else NOT_READY_SENTINEL,
            "delta": DELTA_VALUES[index] if ready else NOT_READY_SENTINEL,
            "theta_delta": 0.25 if ready else NOT_READY_SENTINEL,
            "applied_intensity": 1.0 if index == 6 else None,
            "next_intensity": 1.0 if index in fires else 0.0,
        })
    return rows


def _modes(*, monitor: bool) -> list[str]:
    modes = ["base"] * LENGTH
    if monitor:
        modes[6] = "explore"
    return modes


def _continuous(variant: str, history: list[dict[str, object]]) -> OptimizationResult:
    monitor = variant != "M0_no_dtw"
    return OptimizationResult(
        problem="toy",
        variant=variant,
        seed=1,
        best_cost=1.0,
        best_position=[0.0],
        objective_evaluations=1,
        fire_count=sum(bool(row["raw_fire"]) for row in history),
        transition_count=0,
        parameter_update_count=0,
        convergence_history=list(CONVERGENCE),
        woa_history=[1.0] * LENGTH,
        abc_history=[1.0] * LENGTH,
        mode_history=_modes(monitor=monitor),
        intensity_history=[1.0 if index == 6 else None for index in range(LENGTH)],
        dtw_history=history,
        parameter_history=[_parameter_row(index) for index in range(LENGTH)],
    )


def _binary(variant: str, history: list[dict[str, object]]) -> MKPOptimizationResult:
    monitor = variant != "M0_no_dtw"
    return MKPOptimizationResult(
        problem="toy",
        variant=variant,
        seed=1,
        best_profit=9.0,
        best_solution=[0, 1],
        best_known=12.0,
        objective_evaluations=1,
        iterations_completed=LENGTH,
        optimum_reached=False,
        stop_reason="iteration_budget",
        fire_count=sum(bool(row["raw_fire"]) for row in history),
        transition_count=0,
        parameter_update_count=0,
        local_search_improvements=0,
        convergence_history=list(CONVERGENCE),
        woa_history=[1.0] * LENGTH,
        abc_history=[1.0] * LENGTH,
        mode_history=_modes(monitor=monitor),
        intensity_history=[1.0 if index == 6 else None for index in range(LENGTH)],
        dtw_history=history,
        parameter_history=[_parameter_row(index) for index in range(LENGTH)],
    )


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return left == right


def _solid_lines(axis: matplotlib.axes.Axes) -> list[matplotlib.lines.Line2D]:
    lines = []
    for line in axis.get_lines():
        if line.get_linestyle() not in {"-", "solid"}:
            continue
        xdata = np.asarray(line.get_xdata(), dtype=float)
        if xdata.size == 0:
            continue
        lines.append(line)
    return lines


def _vertical_positions(axis: matplotlib.axes.Axes) -> list[float]:
    positions: list[float] = []
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


def _plotted_points(axis: matplotlib.axes.Axes) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for line in _solid_lines(axis):
        xs = np.asarray(line.get_xdata(), dtype=float)
        ys = np.asarray(line.get_ydata(), dtype=float)
        points.extend(zip(xs.tolist(), ys.tolist(), strict=True))
    return points


def _load_hres_reference():
    path = Path("HRES2-H2/plots/dtw_delta.py")
    spec = importlib.util.spec_from_file_location("hres_reference", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"no se pudo cargar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Segment:
    def __init__(self, mh_nombre: str, n_iters: int) -> None:
        self.mh_nombre = mh_nombre
        self.n_iters = n_iters


def _pixels(figure: matplotlib.figure.Figure) -> np.ndarray:
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba()).copy()


def _equalize_titles(left: matplotlib.figure.Figure, right: matplotlib.figure.Figure) -> None:
    text = "DTW Delta per Iteration  [+ = stagnation | - = active improvement]"
    left.axes[0].set_title(text, fontsize=18, fontweight="bold")
    right.axes[0].set_title(text, fontsize=18, fontweight="bold")
    left.tight_layout()
    right.tight_layout()


class TestDtwDeltaPanel(unittest.TestCase):
    def test_both_engines_draw_one_continuous_delta_panel(self) -> None:
        for engine in ("continuous", "mkp"):
            for variant in VARIANTS:
                with self.subTest(engine=engine, variant=variant):
                    self._assert_variant(engine, variant)

    def test_five_adaptive_examples_keep_negative_delta(self) -> None:
        self.assertEqual(len(NEGATIVE_EXAMPLES), 5)
        for engine in ("continuous", "mkp"):
            for variant in NEGATIVE_EXAMPLES:
                with self.subTest(engine=engine, variant=variant):
                    history = _history(monitor=True)
                    before = [dict(row) for row in history]
                    figure = self._draw(engine, variant, history=history)
                    axis = self._only_axis(figure)
                    plotted = [point[1] for point in _plotted_points(axis)]
                    self.assertIn(-1.25, plotted)
                    self.assertIn(-2.0, plotted)
                    self.assertNotIn(1.25, plotted)
                    self.assertNotIn(2.0, plotted)
                    for actual, expected in zip(history, before, strict=True):
                        self.assertEqual(actual["delta"], expected["delta"])
                    matplotlib.pyplot.close(figure)

    def test_no_confirmed_fire_keeps_initial_boundary_only(self) -> None:
        for engine in ("continuous", "mkp"):
            with self.subTest(engine=engine):
                history = _history(monitor=True, fires=())
                result = self._result(engine, "M1_fire_d2", history)
                figure = self._draw(engine, "M1_fire_d2", history=history)
                axis = self._only_axis(figure)
                self.assertEqual(result.fire_count, 0)
                self.assertFalse(any(row["raw_fire"] for row in history))
                self.assertEqual(_vertical_positions(axis), [INITIAL_BOUNDARY])
                labels = [text.get_text() for text in axis.get_legend().get_texts()]
                self.assertEqual(labels, ["M1_fire_d2"])
                self.assertNotIn("Fire DTW", labels)
                matplotlib.pyplot.close(figure)

    def test_boundary_lines_follow_source_cutoff(self) -> None:
        visible = self._cutoff_history(fires=tuple(range(1, 48)))
        hidden = self._cutoff_history(fires=tuple(range(1, 50)))
        for engine in ("continuous", "mkp"):
            with self.subTest(engine=engine, cutoff="visible"):
                figure = self._draw(engine, "M2_fire_3cond", history=visible)
                positions = _vertical_positions(self._only_axis(figure))
                self.assertEqual(positions[0], INITIAL_BOUNDARY)
                self.assertEqual(positions, [0.0, *range(1, 48)])
                matplotlib.pyplot.close(figure)
            with self.subTest(engine=engine, cutoff="hidden"):
                figure = self._draw(engine, "M2_fire_3cond", history=hidden)
                self.assertEqual(_vertical_positions(self._only_axis(figure)), [])
                matplotlib.pyplot.close(figure)

    def test_render_parity_against_hres_source(self) -> None:
        from woa_abc_version_vieja.reporting.plots import render_reference_dtw_delta

        reference = _load_hres_reference()
        deltas = ["", float("nan"), -1.25, 0.5, "", -2.0, 3.0]
        segments = [
            _Segment("PSO", 0),
            _Segment("PSO", 3),
            _Segment("SA", 2),
            _Segment("ABC", 2),
        ]
        colors = {"PSO": "#1f77b4", "SA": "#ff7f0e", "ABC": "#2ca02c"}
        before_deltas = list(deltas)
        before_lengths = [segment.n_iters for segment in segments]
        reference_figure = self._render_original(
            reference,
            deltas,
            segments,
            colors,
        )
        copied = render_reference_dtw_delta(
            deltas,
            segments,
            colors,
            title_prefix="HRES2-H2",
        )
        self.assertEqual(deltas, before_deltas)
        self.assertEqual([segment.n_iters for segment in segments], before_lengths)
        self._assert_same_artists(reference_figure, copied)
        self.assertEqual(copied.axes[0].get_title(), REFERENCE_TITLE)
        _equalize_titles(reference_figure, copied)
        self.assertTrue(np.array_equal(_pixels(reference_figure), _pixels(copied)))
        matplotlib.pyplot.close(reference_figure)
        matplotlib.pyplot.close(copied)

        hidden_segments = [_Segment("PSO", 1) for _ in range(50)]
        hidden_deltas = [1.5, *([""] * 49)]
        reference_hidden = self._render_original(
            reference,
            hidden_deltas,
            hidden_segments,
            {"PSO": "#1f77b4"},
        )
        copied_hidden = render_reference_dtw_delta(
            hidden_deltas,
            hidden_segments,
            {"PSO": "#1f77b4"},
            title_prefix="HRES2-H2",
        )
        self.assertEqual(_vertical_positions(reference_hidden.axes[0]), [])
        self.assertEqual(_vertical_positions(copied_hidden.axes[0]), [])
        self._assert_same_artists(reference_hidden, copied_hidden)
        matplotlib.pyplot.close(reference_hidden)
        matplotlib.pyplot.close(copied_hidden)

        visible_segments = [_Segment("SA", 1) for _ in range(49)]
        visible_deltas = [-0.5, *([""] * 48)]
        reference_visible = self._render_original(
            reference,
            visible_deltas,
            visible_segments,
            {"SA": "#ff7f0e"},
        )
        copied_visible = render_reference_dtw_delta(
            visible_deltas,
            visible_segments,
            {"SA": "#ff7f0e"},
            title_prefix="HRES2-H2",
        )
        self.assertEqual(_vertical_positions(reference_visible.axes[0])[0], INITIAL_BOUNDARY)
        self.assertEqual(len(_vertical_positions(reference_visible.axes[0])), 49)
        self._assert_same_artists(reference_visible, copied_visible)
        matplotlib.pyplot.close(reference_visible)
        matplotlib.pyplot.close(copied_visible)

    def test_prior_convergence_and_parameter_pdfs_keep_their_artists(self) -> None:
        continuous = _continuous("M1_fire_d2", _history(monitor=True))
        binary = _binary("M5_sigmoid_delta", _history(monitor=True))
        with tempfile.TemporaryDirectory(prefix="woa_plot_prior_") as temporary:
            root = Path(temporary)
            continuous_plots = self._render(save_run_plots, root / "continuous", continuous, 2.5)
            binary_plots = self._render(save_mkp_run_plots, root / "mkp", binary)
            self._assert_pdf_only(root / "continuous", mkp=False)
            self._assert_pdf_only(root / "mkp", mkp=True)
        convergence = continuous_plots["convergence"].axes[0]
        self.assertEqual(convergence.get_ylabel(), "Costo")
        self.assertEqual(convergence.get_title(), "Convergencia — M1_fire_d2")
        self.assertTrue(any(line.get_label().startswith("Óptimo") for line in convergence.get_lines()))
        self.assertEqual(
            [axis.get_title() for axis in continuous_plots["parameters"].axes],
            ["WOA a: base vs. efectivo", "Factor WOA a", "Factor ABC phi", "Factor ABC limit"],
        )
        binary_convergence = binary_plots["convergence"].axes[0]
        self.assertEqual(binary_convergence.get_ylabel(), "Beneficio")
        self.assertTrue(any("Mejor conocido" in line.get_label() for line in binary_convergence.get_lines()))
        self.assertEqual(
            [axis.get_title() for axis in binary_plots["parameters"].axes],
            ["WOA a: base y efectivo", "ABC phi efectivo", "ABC limit efectivo", "Guía ABC al élite"],
        )
        lb2 = binary_plots["lb2_parameters"].axes[0]
        self.assertEqual(lb2.get_title(), "Calendario LB2 (independiente de DTW)")
        self.assertEqual(
            [line.get_label() for line in lb2.get_lines() if line.get_label() and not str(line.get_label()).startswith("_")],
            ["G1", "G2", "G3"],
        )
        for stem, figure in (("continuous", continuous_plots["dtw_adaptation"]), ("mkp", binary_plots["dtw_adaptation"])):
            saved = self._only_axis(figure)
            plotted = _plotted_points(saved)
            unique_y = {point[1] for point in plotted}
            self.assertEqual(unique_y, {DELTA_VALUES[index] for index in VALID_X})
            self.assertIn(INITIAL_BOUNDARY, _vertical_positions(saved))
            self.assertEqual(
                [position for position in _vertical_positions(saved) if position != INITIAL_BOUNDARY],
                [float(index) for index in FIRE_INDICES],
            )
            self.assertFalse(any(line.get_label() == "Óptimo" for line in saved.get_lines()))
            self.assertFalse(any("Mejor conocido" in line.get_label() for line in saved.get_lines()))
            self.assertEqual(stem, stem)

    def _assert_variant(self, engine: str, variant: str) -> None:
        history = _history(monitor=variant != "M0_no_dtw")
        before_history = [dict(row) for row in history]
        result = self._result(engine, variant, history)
        before_intensity = list(result.intensity_history)
        before_fire_count = result.fire_count
        figure = draw_dtw_adaptation(result, 2.5) if engine == "continuous" else draw_mkp_dtw_adaptation(result)
        self.assertEqual(result.intensity_history, before_intensity)
        self.assertEqual(result.fire_count, before_fire_count)
        self.assertEqual(before_fire_count, sum(bool(row["raw_fire"]) for row in history))
        self.assertNotIn(0, FIRE_INDICES)
        self.assertEqual(list(result.convergence_history), CONVERGENCE)
        for actual, expected in zip(result.dtw_history, before_history, strict=True):
            for key, value in expected.items():
                self.assertTrue(_same_value(actual[key], value), key)
        axis = self._only_axis(figure)
        self.assertEqual(axis.get_xlabel(), "Accumulated Iterations")
        self.assertEqual(axis.get_ylabel(), "DTW Delta")
        self.assertEqual(axis.xaxis.label.get_fontsize(), 18)
        self.assertEqual(axis.yaxis.label.get_fontsize(), 18)
        self.assertEqual(axis.title.get_fontsize(), 18)
        self.assertEqual(axis.title.get_fontweight(), "bold")
        self.assertEqual(tuple(np.round(figure.get_size_inches(), 5)), (14.0, 5.0))
        self.assertFalse(any(axis.get_ylabel() == "Intensidad" for axis in figure.axes))
        self.assertNotIn("intensidad", axis.get_title().lower())
        self.assertFalse(self._has_zero_guide(axis))
        if variant == "M0_no_dtw":
            self.assertEqual(axis.get_title(), "DTW desactivado")
            self.assertEqual(_solid_lines(axis), [])
            self.assertEqual(_vertical_positions(axis), [])
            self.assertIsNone(axis.get_legend())
            matplotlib.pyplot.close(figure)
            return
        self.assertEqual(axis.get_title(), WOA_TITLE)
        self.assertIn("[+ = stagnation | - = active improvement]", axis.get_title())
        plotted = _plotted_points(axis)
        unique_x = sorted({point[0] for point in plotted})
        self.assertEqual(unique_x, [float(index) for index in VALID_X])
        for x_value, y_value in plotted:
            self.assertEqual(y_value, DELTA_VALUES[int(x_value)])
        self.assertTrue(all(math.isfinite(x_value) and math.isfinite(y_value) for x_value, y_value in plotted))
        self.assertNotIn(NOT_READY_SENTINEL, [point[1] for point in plotted])
        self.assertTrue(any(y_value < 0.0 for _x_value, y_value in plotted))
        self.assertFalse(any(point[1] in {0.25, 0.125, 4.0, 1.0} for point in plotted))
        for line in _solid_lines(axis):
            self.assertEqual(float(line.get_linewidth()), 2.5)
            self.assertEqual(float(line.get_alpha()), 0.85)
            self.assertTrue(np.allclose(to_rgb(line.get_color()), VARIANT_COLOR, atol=0.02))
        self.assertEqual(_vertical_positions(axis), [INITIAL_BOUNDARY, *[float(index) for index in FIRE_INDICES]])
        for line in axis.get_lines():
            if line.get_linestyle() not in {"--", "dashed"}:
                continue
            self.assertEqual(float(line.get_linewidth()), 1.5)
            self.assertEqual(float(line.get_alpha()), 0.5)
            self.assertTrue(np.allclose(to_rgb(line.get_color()), VARIANT_COLOR, atol=0.02))
        labels = [text.get_text() for text in axis.get_legend().get_texts()]
        self.assertEqual(labels, [variant])
        self.assertNotIn("Fire DTW", labels)
        self.assertNotIn("PSO", labels)
        self.assertNotIn("SA", labels)
        self.assertNotIn("D2", labels)
        self.assertIn(axis.get_legend()._loc, {4, "lower right"})
        self.assertEqual(axis.get_legend().get_texts()[0].get_fontsize(), 15)
        matplotlib.pyplot.close(figure)

    def _draw(
        self,
        engine: str,
        variant: str,
        *,
        history: list[dict[str, object]] | None = None,
        fires: tuple[int, ...] = FIRE_INDICES,
    ) -> matplotlib.figure.Figure:
        if history is None:
            history = _history(monitor=True, fires=fires)
        result = self._result(engine, variant, history)
        if engine == "continuous":
            return draw_dtw_adaptation(result, 2.5)
        return draw_mkp_dtw_adaptation(result)

    def _result(
        self,
        engine: str,
        variant: str,
        history: list[dict[str, object]],
    ) -> OptimizationResult | MKPOptimizationResult:
        if engine == "continuous":
            return _continuous(variant, history)
        return _binary(variant, history)

    def _only_axis(self, figure: matplotlib.figure.Figure) -> matplotlib.axes.Axes:
        self.assertEqual(len(figure.axes), 1, [axis.get_ylabel() for axis in figure.axes])
        return figure.axes[0]

    def _has_zero_guide(self, axis: matplotlib.axes.Axes) -> bool:
        for line in axis.get_lines():
            ydata = np.asarray(line.get_ydata(), dtype=float)
            xdata = np.asarray(line.get_xdata(), dtype=float)
            if ydata.size >= 2 and np.allclose(ydata, 0.0) and np.unique(xdata).size >= 2:
                return True
        return any(getattr(line, "get_ydata", lambda: [])() is not None and False for line in ())

    def _cutoff_history(self, fires: tuple[int, ...]) -> list[dict[str, object]]:
        total = 100
        rows: list[dict[str, object]] = []
        for index in range(total):
            ready = index == 10 or index in fires
            rows.append({
                "iteration": index,
                "ready": ready,
                "raw_fire": index in fires,
                "delta": -0.75 if ready else "",
            })
        return rows

    def _render_original(self, reference: object, deltas: list, segments: list, colors: dict) -> matplotlib.figure.Figure:
        captured: list[matplotlib.figure.Figure] = []

        def spy(figure: matplotlib.figure.Figure, *args: object, **kwargs: object) -> None:
            del args, kwargs
            captured.append(figure)

        scratch = Path("woa_abc_version_vieja/resultados/_dtw_copy_scratch")
        scratch.mkdir(parents=True, exist_ok=True)
        with patch.object(reference.plt, "close", lambda *args, **kwargs: None):
            with patch.object(matplotlib.figure.Figure, "savefig", spy):
                reference.grafico_dtw_delta(deltas, segments, colors, str(scratch))
        self.assertEqual(len(captured), 1)
        self.assertEqual(list(scratch.glob("dtw_delta.png")), [])
        return captured[0]

    def _assert_same_artists(
        self,
        reference_figure: matplotlib.figure.Figure,
        copied: matplotlib.figure.Figure,
    ) -> None:
        self.assertIsNotNone(copied)
        reference_axis = reference_figure.axes[0]
        copied_axis = copied.axes[0]
        self.assertEqual(len(reference_figure.axes), len(copied.axes))
        self.assertEqual(tuple(np.round(reference_figure.get_size_inches(), 5)), (14.0, 5.0))
        self.assertEqual(
            tuple(np.round(copied.get_size_inches(), 5)),
            tuple(np.round(reference_figure.get_size_inches(), 5)),
        )
        self.assertEqual(copied_axis.get_title(), reference_axis.get_title())
        self.assertEqual(copied_axis.title.get_fontsize(), reference_axis.title.get_fontsize())
        self.assertEqual(copied_axis.title.get_fontweight(), reference_axis.title.get_fontweight())
        self.assertEqual(copied_axis.get_xlabel(), reference_axis.get_xlabel())
        self.assertEqual(copied_axis.get_ylabel(), reference_axis.get_ylabel())
        self.assertEqual(copied_axis.xaxis.label.get_fontsize(), reference_axis.xaxis.label.get_fontsize())
        self.assertEqual(copied_axis.yaxis.label.get_fontsize(), reference_axis.yaxis.label.get_fontsize())
        self.assertEqual(copied_axis.get_lines().__len__(), len(reference_axis.get_lines()))
        self.assertEqual(len(copied_axis.collections), len(reference_axis.collections))
        for copied_line, reference_line in zip(copied_axis.get_lines(), reference_axis.get_lines(), strict=True):
            self.assertTrue(np.allclose(
                np.asarray(copied_line.get_xdata(), dtype=float),
                np.asarray(reference_line.get_xdata(), dtype=float),
            ))
            self.assertTrue(np.allclose(
                np.asarray(copied_line.get_ydata(), dtype=float),
                np.asarray(reference_line.get_ydata(), dtype=float),
            ))
            self.assertEqual(copied_line.get_linestyle(), reference_line.get_linestyle())
            self.assertEqual(copied_line.get_linewidth(), reference_line.get_linewidth())
            self.assertEqual(copied_line.get_alpha(), reference_line.get_alpha())
            self.assertTrue(np.allclose(to_rgb(copied_line.get_color()), to_rgb(reference_line.get_color())))
        reference_legend = reference_axis.get_legend()
        copied_legend = copied_axis.get_legend()
        self.assertEqual(
            [text.get_text() for text in copied_legend.get_texts()],
            [text.get_text() for text in reference_legend.get_texts()],
        )
        self.assertEqual(copied_legend._loc, reference_legend._loc)
        self.assertEqual(copied_legend.get_texts()[0].get_fontsize(), reference_legend.get_texts()[0].get_fontsize())
        self.assertEqual(
            [handle.get_facecolor() for handle in copied_legend.legend_handles],
            [handle.get_facecolor() for handle in reference_legend.legend_handles],
        )
        self.assertEqual(
            [line.get_alpha() for line in copied_axis.get_xgridlines()],
            [line.get_alpha() for line in reference_axis.get_xgridlines()],
        )
        self.assertEqual(
            [line.get_visible() for line in copied_axis.get_xgridlines()],
            [line.get_visible() for line in reference_axis.get_xgridlines()],
        )
        self.assertTrue(copied_axis.get_xgridlines())
        self.assertEqual(copied_axis.get_xgridlines()[0].get_alpha(), 0.3)
        self.assertEqual(_vertical_positions(copied_axis), _vertical_positions(reference_axis))

    def _render(
        self,
        plotter: object,
        directory: Path,
        result: object,
        *extra: object,
    ) -> dict[str, matplotlib.figure.Figure]:
        captured: dict[str, matplotlib.figure.Figure] = {}
        original = matplotlib.figure.Figure.savefig

        def spy(figure: matplotlib.figure.Figure, *args: object, **kwargs: object) -> object:
            target = args[0] if args else kwargs.get("fname")
            captured.setdefault(Path(str(target)).stem, figure)
            return original(figure, *args, **kwargs)

        with patch.object(matplotlib.figure.Figure, "savefig", spy):
            plotter(directory, result, *extra)
        return captured

    def _assert_pdf_only(self, directory: Path, *, mkp: bool) -> None:
        expected = {"convergence.pdf", "dtw_adaptation.pdf", "parameters.pdf"}
        if mkp:
            expected.add("lb2_parameters.pdf")
        self.assertEqual({path.name for path in directory.glob("*.pdf")}, expected)
        self.assertEqual(list(directory.glob("*.png")), [])
        for pdf in directory.glob("*.pdf"):
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"), pdf.name)


if __name__ == "__main__":
    unittest.main()
