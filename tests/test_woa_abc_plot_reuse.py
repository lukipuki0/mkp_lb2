"""Prove the WOA-ABC plots reuse the GWO-DE drawing code, not a restyle.

gwo_de_dtw lives on another branch, but its convergence and delta plots are
copies of hybrid_mkp's renderers, which are on this branch. Those originals
are loaded by file path (``hybrid_mkp/__init__.py`` is never imported) and
must draw the same artists as woa_abc.plots for the same series.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

import woa_abc.benchmark as bench
from woa_abc.plots import plot_convergence, plot_dtw_delta
from woa_abc.profiles import PROFILE_COLORS


ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")


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


def _figure_of(fn, *args):
    """Run one plot and return the figure it would have closed."""
    kept = []
    with patch.object(plt, "close", side_effect=kept.append), patch("sys.stdout", StringIO()):
        fn(*args)
    return kept[0]


def _lines(ax) -> list[tuple]:
    return [
        (
            tuple(round(c, 5) for c in matplotlib.colors.to_rgba(line.get_color())),
            line.get_linewidth(),
            line.get_linestyle(),
            line.get_alpha(),
            tuple(np.asarray(line.get_xdata(), dtype=float).round(6)),
            tuple(np.asarray(line.get_ydata(), dtype=float).round(6)),
        )
        for line in ax.lines
    ]


def _legend(ax) -> tuple:
    legend = ax.get_legend()
    return (
        [text.get_text() for text in legend.get_texts()],
        [tuple(round(c, 5) for c in handle.get_facecolor()) for handle in legend.legend_handles],
        legend.get_texts()[0].get_fontsize(),
        legend.get_frame().get_alpha(),
    )


def _record() -> dict:
    """Three profile turns (exploit, explore, exploit) and a partly ready monitor."""
    profiles = ["exploit"] * 4 + ["explore"] * 3 + ["exploit"] * 3
    best = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 6.0]
    deltas = [NAN, NAN, 0.5, 0.7, NAN, NAN, 0.2, -0.1, 0.4, 0.9]
    monitor = [{"ready": False} if np.isnan(d) else {"ready": True, "delta": d} for d in deltas]
    return {"best_series": best, "profile_series": profiles, "known_optimum": 7.0,
            "monitor": monitor, "deltas": deltas,
            "switches": [_Switch("exploit", 4), _Switch("explore", 3), _Switch("exploit", 3)]}


def _assert_same_figure(testcase, original, ours) -> None:
    original_ax, our_ax = original.axes[0], ours.axes[0]
    testcase.assertEqual(tuple(original.get_size_inches()), tuple(ours.get_size_inches()))
    testcase.assertEqual(_lines(original_ax), _lines(our_ax))
    testcase.assertEqual(_legend(original_ax), _legend(our_ax))
    for getter in ("get_title", "get_xlabel", "get_ylabel"):
        testcase.assertEqual(getattr(original_ax, getter)(), getattr(our_ax, getter)())
    testcase.assertEqual(original_ax.title.get_fontsize(), our_ax.title.get_fontsize())
    testcase.assertEqual(original_ax.xaxis.label.get_fontsize(), our_ax.xaxis.label.get_fontsize())
    original.canvas.draw()
    ours.canvas.draw()
    testcase.assertTrue(np.array_equal(np.asarray(original.canvas.buffer_rgba()),
                                       np.asarray(ours.canvas.buffer_rgba())))


class SourceReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(plt.close, "all")
        self.record = _record()

    def test_convergence_draws_what_the_copied_original_draws(self) -> None:
        source = _load_plot_file("convergencia.py", "_mkp_src_convergencia")
        record = self.record
        original = _figure_of(source.grafico_convergencia, record["best_series"], record["switches"],
                              PROFILE_COLORS, record["known_optimum"], self.tmp.name)
        ours = _figure_of(plot_convergence, record, self.tmp.name)
        _assert_same_figure(self, original, ours)

    def test_delta_draws_what_the_copied_original_draws(self) -> None:
        source = _load_plot_file("dtw_delta.py", "_mkp_src_dtw_delta")
        record = self.record
        original = _figure_of(source.grafico_dtw_delta, record["deltas"], record["switches"],
                              PROFILE_COLORS, self.tmp.name)
        ours = _figure_of(plot_dtw_delta, record, self.tmp.name)
        _assert_same_figure(self, original, ours)

    def test_profile_colors_are_the_gwo_de_colors(self) -> None:
        self.assertEqual(PROFILE_COLORS, {"exploit": "#1f77b4", "explore": "#ff7f0e"})


class BoundaryTests(unittest.TestCase):
    def test_exporters_import_neither_hybrid_mkp_nor_the_old_plots(self) -> None:
        code = (
            "import sys, woa_abc.benchmark, woa_abc.run, woa_abc.campaign; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('hybrid_mkp', 'gwo_de_dtw') "
            "or m == 'woa_abc.mkp_plots']; print(bad)"
        )
        done = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                              text=True, check=True)
        self.assertEqual(done.stdout.strip(), "[]")

    def test_benchmark_no_longer_exposes_the_old_plot_api(self) -> None:
        for name in ("draw_boxplot", "draw_convergence", "draw_delta", "draw_parameters",
                     "plot_boxplot", "BOX_TITLE", "PARAMETER_SIZE"):
            self.assertFalse(hasattr(bench, name), name)
        self.assertNotIn("boxplot_runs.png", bench.EXPECTED_ARTIFACTS)
        self.assertIn("best_run_01/parameters.png", bench.EXPECTED_ARTIFACTS)


if __name__ == "__main__":
    unittest.main()
