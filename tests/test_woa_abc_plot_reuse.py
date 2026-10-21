"""Prove WOA-ABC plots reuse the MKP drawing code, not a restyle.

Original renderers are loaded by file path so ``hybrid_mkp/__init__.py`` is
never imported. Comparisons allow only the WOA-ABC title and hybrid-cycle
axis text to differ.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import shutil
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

import woa_abc.benchmark as bench
from woa_abc.benchmark import (
    HistoryRow,
    draw_boxplot,
    draw_convergence,
    draw_delta,
    plot_convergence,
    plot_delta,
)


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "woa_abc" / "resultados" / "_plot_reuse_unittest"
SEGMENT = "WOA\u2013ABC"
SERIES = "#1565C0"


class _Switch:
    def __init__(self, mh_nombre: str, n_iters: int) -> None:
        self.mh_nombre = mh_nombre
        self.n_iters = n_iters


def _load_plot_file(filename: str, module_name: str):
    path = ROOT / "hybrid_mkp" / "plots" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture(module, fn, args):
    holder: dict = {}
    real_subplots = module.plt.subplots

    def subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        holder["fig"] = fig
        holder["ax"] = ax
        return fig, ax

    def savefig(self, *args, **kwargs):
        holder["dpi"] = kwargs.get("dpi")
        holder["path"] = args[0] if args else kwargs.get("fname")

    stdout = StringIO()
    with patch.object(module.plt, "subplots", subplots), patch.object(
        module.plt.Figure, "savefig", savefig
    ), patch.object(module.plt, "close", lambda *a, **k: None), patch(
        "sys.stdout", stdout
    ):
        result = fn(*args)
    holder["result"] = result
    holder["stdout"] = stdout.getvalue()
    return holder


def _rgba(color) -> tuple:
    return tuple(round(channel, 5) for channel in matplotlib.colors.to_rgba(color))


def _line_signature(ax) -> list[tuple]:
    signature = []
    for line in ax.lines:
        signature.append(
            (
                _rgba(line.get_color()),
                line.get_linewidth(),
                line.get_linestyle(),
                line.get_alpha(),
                tuple(np.asarray(line.get_xdata(), dtype=float).round(6)),
                tuple(np.asarray(line.get_ydata(), dtype=float).round(6)),
            )
        )
    return signature


def _legend_signature(ax) -> dict:
    legend = ax.get_legend()
    if legend is None:
        return {"texts": [], "kinds": [], "colors": []}
    colors = []
    for handle in legend.legend_handles:
        if hasattr(handle, "get_facecolor"):
            face = handle.get_facecolor()
            colors.append(tuple(round(channel, 5) for channel in face))
        else:
            colors.append(_rgba(handle.get_color()))
    return {
        "texts": [text.get_text() for text in legend.get_texts()],
        "kinds": [type(handle).__name__ for handle in legend.legend_handles],
        "colors": colors,
        "fontsize": legend.get_texts()[0].get_fontsize() if legend.get_texts() else None,
        "frameon": legend.get_frame_on(),
        "framealpha": legend.get_frame().get_alpha(),
    }


def _assert_same_artists(testcase, original_ax, our_ax) -> None:
    testcase.assertEqual(_line_signature(original_ax), _line_signature(our_ax))
    testcase.assertEqual(len(original_ax.collections), len(our_ax.collections))
    original_legend = _legend_signature(original_ax)
    our_legend = _legend_signature(our_ax)
    testcase.assertEqual(original_legend["texts"], our_legend["texts"])
    testcase.assertEqual(original_legend["kinds"], our_legend["kinds"])
    testcase.assertEqual(original_legend["colors"], our_legend["colors"])
    testcase.assertEqual(original_legend["fontsize"], our_legend["fontsize"])
    testcase.assertEqual(original_legend["frameon"], our_legend["frameon"])
    testcase.assertEqual(original_legend["framealpha"], our_legend["framealpha"])
    testcase.assertEqual(original_ax.title.get_fontsize(), our_ax.title.get_fontsize())
    testcase.assertEqual(original_ax.title.get_fontweight(), our_ax.title.get_fontweight())
    testcase.assertNotEqual(original_ax.get_title(), our_ax.get_title())
    testcase.assertEqual(original_ax.xaxis.label.get_fontsize(), our_ax.xaxis.label.get_fontsize())
    testcase.assertEqual(original_ax.yaxis.label.get_fontsize(), our_ax.yaxis.label.get_fontsize())
    testcase.assertEqual(original_ax.get_ylabel(), our_ax.get_ylabel())
    testcase.assertNotEqual(original_ax.get_xlabel(), our_ax.get_xlabel())
    testcase.assertEqual(our_ax.get_xlabel(), "Hybrid cycles")
    testcase.assertIn("WOA", our_ax.get_title())
    testcase.assertNotIn("Pipeline", our_ax.get_title())
    if original_ax.get_xticklabels():
        testcase.assertEqual(
            original_ax.get_xticklabels()[0].get_fontsize(),
            our_ax.get_xticklabels()[0].get_fontsize(),
        )


def _sync_and_pixels(original_fig, our_fig) -> tuple[np.ndarray, np.ndarray]:
    original_ax = original_fig.axes[0]
    our_ax = our_fig.axes[0]
    # set_text keeps font size and weight. set_title() would reset them and
    # move the axes even when the drawing code matches.
    our_ax.title.set_text(original_ax.get_title())
    our_ax.xaxis.label.set_text(original_ax.get_xlabel())
    original_fig.canvas.draw()
    our_fig.canvas.draw()
    original_fig.tight_layout()
    our_fig.tight_layout()
    original_fig.canvas.draw()
    our_fig.canvas.draw()
    return (
        np.asarray(original_fig.canvas.buffer_rgba()).copy(),
        np.asarray(our_fig.canvas.buffer_rgba()).copy(),
    )


def _row(cycle, best, diagnostic) -> HistoryRow:
    applied = None if diagnostic is None else bool(diagnostic.get("recovery_applied"))
    return HistoryRow(
        cycle=cycle,
        iteration=cycle + 1,
        evaluations=10 + cycle,
        a=2.0 - cycle,
        rho=1.0,
        schedule_index=0,
        best=best,
        completeness=cycle != 1,
        recovery_applied=applied,
        diagnostic=diagnostic,
    )


class SourceReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.output = SCRATCH / self.id().split(".")[-1]
        if self.output.exists():
            shutil.rmtree(self.output)
        self.output.mkdir()
        self.convergence = _load_plot_file("convergencia.py", "_mkp_src_convergencia")
        self.delta = _load_plot_file("dtw_delta.py", "_mkp_src_dtw_delta")

    def tearDown(self) -> None:
        plt.close("all")
        if self.output.exists():
            shutil.rmtree(self.output, ignore_errors=True)

    def test_convergence_entrypoint_matches_one_source_segment(self) -> None:
        rows = [
            _row(0, 1.0, {"cycle": 0, "delta": 1.5, "recovery_applied": False, "fire": False}),
            _row(1, 2.0, None),
            _row(2, 2.0, {"cycle": 2, "delta": float("nan"), "recovery_applied": True, "fire": True}),
            _row(3, 4.0, {"cycle": 3, "delta": -2.0, "recovery_applied": False, "fire": False}),
        ]
        held = _capture(
            self.convergence,
            self.convergence.grafico_convergencia,
            ([row.best for row in rows], [_Switch(SEGMENT, len(rows))], {SEGMENT: SERIES}, 12.0, str(self.output)),
        )
        fig, ax = plt.subplots(figsize=(14, 6))
        draw_convergence(ax, rows, 12.0)
        try:
            self.assertNotIn("parameter recovery", [text.get_text() for text in ax.get_legend().get_texts()])
            self.assertFalse(any("recovery" in label.lower() for label in ax.get_legend_handles_labels()[1]))
            self.assertEqual(len(ax.collections), 0)
            _assert_same_artists(self, held["ax"], ax)
            self.assertEqual(tuple(held["fig"].get_size_inches()), (14, 6))
            self.assertEqual(held["dpi"], 150)
            self.assertIn("fitness_convergence.png", str(held["path"]))
        finally:
            plt.close(fig)

    def test_delta_entrypoint_matches_one_source_segment_and_keeps_holes(self) -> None:
        rows = [
            _row(0, 1.0, {"cycle": 0, "delta": 4.0, "recovery_applied": False}),
            _row(1, 2.0, None),
            _row(2, 3.0, {"cycle": 2, "delta": float("inf"), "recovery_applied": True, "fire": True}),
            _row(3, 5.0, {"cycle": 3, "delta": -1.0, "recovery_applied": False}),
        ]
        held = _capture(
            self.delta,
            self.delta.grafico_dtw_delta,
            ([4.0, "", float("nan"), -1.0], [_Switch(SEGMENT, 4)], {SEGMENT: SERIES}, str(self.output)),
        )
        fig, ax = plt.subplots(figsize=(14, 5))
        draw_delta(ax, rows)
        try:
            labels = [text.get_text() for text in ax.get_legend().get_texts()]
            self.assertNotIn("parameter recovery", labels)
            self.assertNotIn("delta 0", labels)
            self.assertEqual(len(ax.collections), 0)
            plotted = [line for line in ax.lines if line.get_linestyle() == "-"]
            self.assertEqual(len(plotted), 1)
            self.assertEqual(list(plotted[0].get_xdata()), [0, 3])
            self.assertEqual(list(plotted[0].get_ydata()), [4.0, -1.0])
            _assert_same_artists(self, held["ax"], ax)
            self.assertEqual(tuple(held["fig"].get_size_inches()), (14, 5))
            self.assertEqual(held["dpi"], 150)
        finally:
            plt.close(fig)

    def test_copied_renderers_match_source_pixels_on_multiple_segments(self) -> None:
        plots = importlib.import_module("woa_abc.mkp_plots")
        switches = [_Switch("ABC", 3), _Switch("WOA", 0), _Switch("GA", 2)]
        colors = {"ABC": "#1565C0", "WOA": "#E040FB", "GA": "#4CAF50"}
        historial = [1.0, 1.5, 2.0, 2.5, 4.0]
        deltas = [0.5, float("nan"), 1.25, "", -0.5]
        original_c = _capture(
            self.convergence,
            self.convergence.grafico_convergencia,
            (historial, switches, colors, 9.0, str(self.output / "orig-c")),
        )
        copied_c = _capture(
            plots,
            plots.grafico_convergencia,
            (historial, switches, colors, 9.0, str(self.output / "copy-c")),
        )
        original_d = _capture(
            self.delta,
            self.delta.grafico_dtw_delta,
            (deltas, switches, colors, str(self.output / "orig-d")),
        )
        copied_d = _capture(
            plots,
            plots.grafico_dtw_delta,
            (deltas, switches, colors, str(self.output / "copy-d")),
        )
        try:
            _assert_same_artists(self, original_c["ax"], copied_c["ax"])
            _assert_same_artists(self, original_d["ax"], copied_d["ax"])
            left_c, right_c = _sync_and_pixels(original_c["fig"], copied_c["fig"])
            left_d, right_d = _sync_and_pixels(original_d["fig"], copied_d["fig"])
            self.assertTrue(np.array_equal(left_c, right_c))
            self.assertTrue(np.array_equal(left_d, right_d))
            self.assertIn("_paint_convergencia", inspect.getsource(plots.grafico_convergencia))
            self.assertIn("dibujar_lineas = len(log_switches) < 50", inspect.getsource(plots._paint_convergencia))
            self.assertIn("ys_plot = [historial_global[offset - 1]] + seg", inspect.getsource(plots._paint_convergencia))
            self.assertIn("linewidth=3.0, alpha=0.85", inspect.getsource(plots._paint_convergencia))
            self.assertIn("ys_validos[xs_validos.index(x)]", inspect.getsource(plots._paint_dtw_delta))
            self.assertIn("linewidth=2.5, alpha=0.85", inspect.getsource(plots._paint_dtw_delta))
            convergence_source = inspect.getsource(plots.draw_convergence)
            self.assertIn("_paint_convergencia", convergence_source)
            self.assertNotIn("scatter", convergence_source)
            self.assertNotIn("recovery", convergence_source)
            self.assertIn("_paint_dtw_delta", inspect.getsource(plots.draw_delta))
        finally:
            plt.close("all")

    def test_boxplot_copies_batch_artists_for_n1_without_claiming_31(self) -> None:
        source = (ROOT / "hybrid_mkp" / "batch_benchmark.py").read_text(encoding="utf-8")
        copied = (ROOT / "woa_abc" / "mkp_plots.py").read_text(encoding="utf-8")
        for snippet in (
            'medianprops=dict(color="#FF5722", linewidth=2.5)',
            'boxprops=dict(facecolor="#1565C0", alpha=0.7)',
            'flierprops=dict(marker="o", color="#FF9800", markersize=5)',
            'whiskerprops=dict(color="#90CAF9", linewidth=1.5)',
            'capprops=dict(color="#90CAF9", linewidth=2)',
            'color="#FFEB3B", zorder=5, s=60, label=f"Mean = {mu:.1f}"',
            'color="#4CAF50", linestyle="--", linewidth=1.5',
        ):
            self.assertIn(snippet, source)
            self.assertIn(snippet, copied)
        self.assertIn("N_RUNS", source)
        self.assertNotIn("N_RUNS", copied)
        self.assertNotIn("31 Runs", copied)
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_boxplot(fig, ax, 8.0, "tiny/inst_00", 0.0)
        self.assertIn("n = 1", ax.get_title())
        self.assertIn("not inferential", ax.get_title())
        self.assertNotIn("31", ax.get_title())
        self.assertNotIn("31", ax.get_xlabel())
        self.assertIn("WOA", ax.get_xlabel())
        self.assertEqual(fig.texts[0].get_text(), bench.BOX_NOTE)
        self.assertFalse(any(line.get_linestyle() == "--" for line in ax.lines))
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(_rgba(ax.collections[0].get_facecolor()[0]), _rgba("#FFEB3B"))
        box = ax.patches[0]
        face = tuple(box.get_facecolor())
        self.assertEqual(tuple(round(channel, 5) for channel in face[:3]), _rgba("#1565C0")[:3])
        self.assertEqual(round(face[3], 5), 0.7)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_boxplot(fig, ax, 8.0, "tiny/inst_00", 10.0)
        optimum = [line for line in ax.lines if line.get_linestyle() == "--"]
        self.assertEqual(len(optimum), 1)
        self.assertEqual(_rgba(optimum[0].get_color()), _rgba("#4CAF50"))
        self.assertIn("Known Optimum = 10.0", ax.get_legend_handles_labels()[1])
        plt.close(fig)

    def test_delta_title_fits_inside_the_figure_at_150_dpi(self) -> None:
        rows = [
            _row(0, 1.0, {"cycle": 0, "delta": 4.0, "recovery_applied": False}),
            _row(1, 2.0, {"cycle": 1, "delta": -1.0, "recovery_applied": False}),
        ]
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        try:
            draw_delta(ax, rows)
            self.assertEqual(
                ax.get_title(),
                "WOA\u2013ABC DTW/DDTW Delta\n[+ = stagnation | - = active improvement]",
            )
            self.assertEqual(ax.title.get_fontsize(), 20)
            self.assertEqual(ax.xaxis.label.get_fontsize(), 18)
            self.assertEqual(ax.yaxis.label.get_fontsize(), 18)
            fig.tight_layout()
            fig.canvas.draw()
            bbox = ax.title.get_window_extent(fig.canvas.get_renderer())
            width, height = fig.canvas.get_width_height()
            self.assertEqual((width, height), (14 * 150, 5 * 150))
            self.assertGreaterEqual(bbox.x0, 0)
            self.assertGreaterEqual(bbox.y0, 0)
            self.assertLessEqual(bbox.x1, width)
            self.assertLessEqual(bbox.y1, height)
        finally:
            plt.close(fig)

    def test_plot_helpers_honor_requested_filenames(self) -> None:
        rows = [
            _row(0, 1.0, {"cycle": 0, "delta": 4.0, "recovery_applied": False}),
            _row(1, 3.0, {"cycle": 1, "delta": -2.0, "recovery_applied": True}),
        ]
        sentinel_convergence = self.output / "fitness_convergence.png"
        sentinel_delta = self.output / "dtw_delta.png"
        sentinel_convergence.write_bytes(b"keep-convergence")
        sentinel_delta.write_bytes(b"keep-delta")
        custom_convergence = self.output / "custom_convergence.png"
        custom_delta = self.output / "custom_delta.png"
        plot_convergence(rows, 0.0, custom_convergence)
        plot_delta(rows, custom_delta)
        self.assertTrue(custom_convergence.is_file())
        self.assertTrue(custom_delta.is_file())
        self.assertGreater(custom_convergence.stat().st_size, 500)
        self.assertGreater(custom_delta.stat().st_size, 500)
        self.assertEqual(sentinel_convergence.read_bytes(), b"keep-convergence")
        self.assertEqual(sentinel_delta.read_bytes(), b"keep-delta")

        standard = self.output / "standard"
        plot_convergence(rows, 12.0, standard / "fitness_convergence.png")
        plot_delta(rows, standard / "dtw_delta.png")
        self.assertTrue((standard / "fitness_convergence.png").is_file())
        self.assertTrue((standard / "dtw_delta.png").is_file())
        self.assertFalse((standard / "custom_convergence.png").exists())
        self.assertFalse((standard / "custom_delta.png").exists())

    def test_import_isolation_and_empty_delta_png(self) -> None:
        import woa_abc.mkp_plots as plots

        self.assertFalse(any(name == "hybrid_mkp" or name.startswith("hybrid_mkp.") for name in sys.modules))
        self.assertNotIn("import hybrid_mkp", (ROOT / "woa_abc" / "mkp_plots.py").read_text(encoding="utf-8"))
        self.assertIs(bench.draw_convergence, plots.draw_convergence)
        self.assertIs(bench.draw_delta, plots.draw_delta)
        held = _capture(self.delta, self.delta.grafico_dtw_delta, ([float("nan"), ""], [_Switch(SEGMENT, 2)], {SEGMENT: SERIES}, str(self.output)))
        self.assertEqual(held["result"], "")
        self.assertNotIn("fig", held)
        path = self.output / "dtw_delta.png"
        plot_delta([_row(0, 1.0, None)], path)
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 500)
        fig, ax = plt.subplots()
        draw_delta(ax, [_row(0, 1.0, {"cycle": 0, "delta": float("nan")})])
        drawn = " ".join(text.get_text() for text in ax.texts)
        self.assertIn("Empty diagnostic", drawn)
        self.assertIn("Not evidence of improvement", drawn)
        plt.close(fig)

    def test_synthetic_export_keeps_history_numbers(self) -> None:
        instance = self.output / "tiny.txt"
        instance.write_text("1\n3 1 0\n5 4 3\n2 2 2\n4\n", encoding="utf-8")
        outcome = bench.export_single_run(
            bench.ExportRequest(
                instance=instance,
                seed=7,
                population_size=2,
                iterations=2,
                output_root=self.output / "out",
                timestamp="reuse",
            )
        )
        self.assertEqual(outcome.code, 0, outcome.message)
        raw = (outcome.output_dir / "runs" / "run_01.json").read_text(encoding="utf-8")
        self.assertIn('"best_value"', raw)
        history = (outcome.output_dir / "best_run_01" / "historial_dtw.csv").read_text(encoding="utf-8")
        self.assertIn("fitness_best", history)
        self.assertNotIn("ESTANCAMIENTO", history)
        for name in ("fitness_convergence.png", "dtw_delta.png", "parameter_recovery.png"):
            self.assertTrue((outcome.output_dir / "best_run_01" / name).is_file())
        self.assertTrue((outcome.output_dir / "boxplot_runs.png").is_file())


if __name__ == "__main__":
    unittest.main()
