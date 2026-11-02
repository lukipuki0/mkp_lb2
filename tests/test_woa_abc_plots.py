"""WOA-ABC plots are the GWO-DE plots (spec S8): files, segments, panels, group run."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from dtw_stagnation import StagnationConfig
from woa_abc import solver
from woa_abc.plots import (
    plot_convergence,
    plot_dtw_delta,
    plot_parameters,
    plot_record,
    plot_temporal,
    save_all_plots,
)
from woa_abc.plots._common import profile_turns
from woa_abc.profiles import POLICIES
from woa_abc.run import main as run_main
from woa_abc.solver import HybridConfig, solve


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "instancias" / "mknapcb1.txt"
MONITOR = StagnationConfig(window=5, band=1, min_slope=0.1, plateau_max=3, patience=2, use_ddtw=True)
TEMPORAL = ("temporal_diagnostics.png", "temporal_diagnostics.pdf", "temporal_series.npz",
            "temporal_series.csv", "temporal_metadata.json")
PLOT_FILES = ("fitness_convergence.png", "dtw_delta.png", "parameters.png") + TEMPORAL


def _solve(**overrides):
    inst, info = solver._load_local_instance(INSTANCE, 0)
    values = dict(population_size=4, iterations=50, stag_cfg=MONITOR, policy="binary_simple")
    values.update(overrides)
    return solve(inst, HybridConfig(**values), seed=1, metadata={"instance": info})


@contextlib.contextmanager
def _kept_figures():
    """Keep the figures a plot closes so the test can inspect them."""
    kept = []
    try:
        with patch.object(plt, "close", side_effect=kept.append):
            yield kept
    finally:
        for fig in kept:
            plt.close(fig)


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


class PlotRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = _solve()
        cls.res = plot_record(cls.result, instance="mknapcb1.txt[0]")

    def test_one_entry_per_complete_cycle_with_decision_after_cycle(self):
        res = self.res
        self.assertEqual(len(res["best_series"]), 50)
        for key in ("profile_series", "intensity_series", "monitor"):
            self.assertEqual(len(res[key]), 50, key)
        self.assertEqual(sorted(res["parameter_series"]), ["G3", "a", "rho"])
        self.assertEqual(res["best_series"], [record.best for record in self.result.history])
        self.assertEqual(res["profile_series"], [record.profile for record in self.result.history])
        self.assertEqual(res["parameter_series"]["rho"], [record.rho for record in self.result.history])
        self.assertEqual([entry["profile"] for entry in res["monitor"]],
                         [item["next_profile"] for item in self.result.diagnostics])
        self.assertEqual(res["changes"], self.result.profile_changes)
        self.assertGreaterEqual(len(res["changes"]), 2)
        self.assertEqual(res["monitor_config"], self.result.metadata["config"]["stagnation"])
        self.assertEqual((res["policy"], res["seed"], res["known_optimum"]), ("binary_simple", 1, 0.0))

    def test_json_form_gives_the_same_record(self):
        loaded = json.loads(json.dumps(asdict(self.result)))
        self.assertEqual(plot_record(loaded, instance="mknapcb1.txt[0]"), self.res)

    def test_partial_final_cycle_is_left_out(self):
        result = _solve(iterations=10, max_evaluations=4 + 6 * 4 * 3 + 3)
        self.assertFalse(result.history[-1].completeness)
        res = plot_record(result)
        self.assertEqual(len(res["best_series"]), result.cycles_completed)
        self.assertEqual(len(res["monitor"]), result.cycles_completed)
        self.assertEqual(res["best_fitness"], result.best_value)

    def test_complete_cycle_without_diagnostic_is_refused(self):
        broken = replace(self.result, diagnostics=self.result.diagnostics[:-1])
        with self.assertRaisesRegex(ValueError, "complete cycle 49 has no diagnostic"):
            plot_record(broken)

    def test_detector_off_skips_delta_and_temporal(self):
        res = plot_record(_solve(iterations=10, use_stagnation=False))
        self.assertNotIn("monitor_config", res)
        self.assertEqual({entry["ready"] for entry in res["monitor"]}, {False})
        with tempfile.TemporaryDirectory() as tmp:
            paths = _quiet(save_all_plots, res, tmp)
            self.assertEqual(sorted(Path(path).name for path in paths),
                             ["fitness_convergence.png", "parameters.png"])
            self.assertEqual(sorted(path.name for path in Path(tmp).iterdir()),
                             ["fitness_convergence.png", "parameters.png"])


class PlotFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = plot_record(_solve(), instance="mknapcb1.txt[0]")
        cls.turns = profile_turns(cls.res["profile_series"])

    def test_every_gwo_de_file_is_written_and_no_old_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _quiet(save_all_plots, self.res, Path(tmp) / "run")
            names = sorted(path.name for path in (Path(tmp) / "run").iterdir())
            self.assertEqual(names, sorted(PLOT_FILES))
            self.assertEqual(len(paths), 4)
            for name in PLOT_FILES:
                self.assertGreater((Path(tmp) / "run" / name).stat().st_size, 100, name)
            self.assertNotIn("boxplot_runs.png", names)
            self.assertNotIn("parameter_profiles.png", names)
            with (Path(tmp) / "run" / "temporal_series.csv").open(encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            self.assertIn("rho", header)
            self.assertNotIn("F", header)
            self.assertNotIn("CR", header)

    def test_convergence_draws_one_segment_and_change_line_per_profile_turn(self):
        self.assertGreaterEqual(len(self.turns), 3)
        with tempfile.TemporaryDirectory() as tmp, _kept_figures() as kept:
            _quiet(plot_convergence, self.res, tmp)
            ax = kept[0].axes[0]
            segments = [line for line in ax.lines if line.get_linewidth() == 3.0]
            changes = [line for line in ax.lines if line.get_linestyle() == "--"]
            self.assertEqual(len(segments), len(self.turns))
            self.assertEqual(len(changes), len(self.turns))
            starts, offset = [], 0
            for _, n in self.turns:
                starts.append(offset)
                offset += n
            self.assertEqual([line.get_xdata()[0] for line in changes], starts)
            self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()],
                             ["exploit", "explore"])
            self.assertEqual(ax.get_xlabel(), "Accumulated Iterations")

    def test_delta_draws_one_segment_per_profile_turn_with_a_ready_point(self):
        with tempfile.TemporaryDirectory() as tmp, _kept_figures() as kept:
            path = _quiet(plot_dtw_delta, self.res, tmp)
            self.assertTrue(Path(path).is_file())
            ax = kept[0].axes[0]
            ready, offset, expected = [entry["ready"] for entry in self.res["monitor"]], 0, 0
            for _, n in self.turns:
                expected += any(ready[offset:offset + n])
                offset += n
            segments = [line for line in ax.lines if line.get_linewidth() == 2.5]
            changes = [line for line in ax.lines if line.get_linestyle() == "--"]
            self.assertEqual(len(segments), expected)
            self.assertEqual(len(changes), len(self.turns))

    def test_parameters_panels_are_a_and_rho_per_cycle(self):
        with tempfile.TemporaryDirectory() as tmp, _kept_figures() as kept:
            _quiet(plot_parameters, self.res, tmp)
            fig = kept[0]
            self.assertEqual([ax.get_ylabel() for ax in fig.axes], ["a (WOA)", "rho (ABC)"])
            self.assertEqual(fig.axes[-1].get_xlabel(), "Cycle")
            self.assertEqual(fig._suptitle.get_text(), "WOA-ABC Hybrid - Parameters per Cycle")
            self.assertEqual(tuple(fig.get_size_inches()), (14.0, 7.0))
            for ax, key in zip(fig.axes, ("a", "rho")):
                self.assertEqual(list(ax.lines[0].get_ydata()), self.res["parameter_series"][key])

    def test_temporal_title_and_axis_name_woa_abc_cycles(self):
        with tempfile.TemporaryDirectory() as tmp, _kept_figures() as kept:
            _quiet(plot_temporal, self.res, tmp)
            fig = kept[0]
            self.assertTrue(fig._suptitle.get_text().startswith("WOA-ABC temporal diagnostics"))
            self.assertEqual(fig.axes[-1].get_xlabel(), "Cycle (1..T)")
            self.assertIn("rho=", fig.axes[-1].get_title())


class GroupRunTests(unittest.TestCase):
    def test_four_policy_folders_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "group.json"
            argv = ["--instance", str(INSTANCE), "--index", "0", "--seed", "1", "--pop", "4",
                    "--iterations", "20", "--window", "5", "--plateau-max", "3",
                    "--patience", "2", "--out", str(out)]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(run_main(argv), 0)
            group = Path(tmp) / "group"
            self.assertEqual(sorted(path.name for path in group.iterdir()),
                             sorted(list(POLICIES) + ["convergence_comparison.pdf",
                                                      "convergence_comparison.png"]))
            for policy in POLICIES:
                folder = group / policy
                for name in ("result.json", "fitness_convergence.png", "parameters.png") + TEMPORAL:
                    self.assertTrue((folder / name).is_file(), f"{policy}/{name}")
                saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["metadata"]["config"]["policy"], policy)
                self.assertEqual(saved["metadata"]["seed"], 1)
                self.assertEqual(saved["cycles_completed"], 20)
            self.assertIn(f"output={group}", stdout.getvalue())

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                run_main(argv)
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("output directory is not empty", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
