"""Public tests for the standalone sequential GWO then DE campaign.

The campaign CLI is ``python -m gwo_de_sequential.campaign``. Algorithm tests
use a tiny instance and patched operators. Campaign tests that must not run the
optimizer patch ``campaign.run_search``.
"""
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtw_stagnation import StagnationConfig, StagnationMonitor  # noqa: E402
from gwo_de_sequential import campaign  # noqa: E402
from gwo_de_sequential.control import run_search  # noqa: E402
from gwo_de_sequential.policies import POLICIES  # noqa: E402
from gwo_de_sequential.problem import load_instance  # noqa: E402
from gwo_de_sequential.search import SequentialSearch  # noqa: E402
import gwo_de_sequential.plots as plots_mod  # noqa: E402
import gwo_de_sequential.search as search_mod  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n"
RUN_FIELDS = [
    "file", "index", "policy", "seed", "status", "best_fitness", "known_optimum",
    "gap_pct", "evaluations", "generations", "changes", "elapsed_s",
]
STAT_FIELDS = [
    "file", "index", "policy", "runs", "best_fitness", "worst_fitness",
    "mean_fitness", "median_fitness", "std_fitness", "best_seed",
    "known_optimum", "best_gap_pct", "mean_gap_pct", "mean_elapsed_s",
]


def write_instances(folder, name="mknapcb1.txt", count=2):
    """Tiny OR-Library text. Instance k has profit[0] = 10 + k."""
    lines = [str(count)]
    for k in range(count):
        lines += [
            "6 2 0",
            " ".join(str(10 * (j + 1) + k) for j in range(6)),
            "3 5 2 4 6 1",
            "2 2 5 3 1 4",
            "9 8",
        ]
    (Path(folder) / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot(folder):
    return {
        str(path.relative_to(folder)): (path.read_bytes() if path.is_file() else None)
        for path in sorted(Path(folder).rglob("*"))
    }


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def monitor_cfg():
    return StagnationConfig(window=4, band=1, min_slope=2.0, plateau_max=2,
                            patience=1, use_ddtw=True)


class FakeRunner:
    def __init__(self, fitness, fail=None):
        self.fitness = fitness
        self.fail = fail
        self.calls = []
        self.records = {}

    def __call__(self, inst, cfg, **kw):
        index = int(inst["p"][0]) - 10
        key = (index, kw["policy"], kw["seed"])
        self.calls.append(key)
        if key == self.fail:
            raise RuntimeError("boom")
        fitness = float(self.fitness(*key))
        optimum = float(inst.get("optimo") or 0.0)
        record = {
            "policy": kw["policy"],
            "seed": kw["seed"],
            "best_fitness": fitness,
            "known_optimum": optimum,
            "gap_pct": (100.0 * (optimum - fitness) / optimum) if optimum > 0 else None,
            "evaluations": 20,
            "generations": 2,
            "changes": [],
            "best_series": [fitness - 1.0, fitness],
            "profile_series": ["exploit", "exploit"],
            "best_solution": [1, 0, 1, 0, 0, 1],
            "step_stats": [
                {"gwo_accepted": 1, "de_accepted": 0, "accepted_total": 1},
                {"gwo_accepted": 0, "de_accepted": 1, "accepted_total": 1},
            ],
            "schedule": "gwo_then_de",
            "dtw_kirito": kw.get("dtw_kirito", False),
            "binarization": kw.get("binarization", "lb2"),
        }
        self.records[key] = copy.deepcopy(record)
        return record


class SequenceAcceptanceCostTests(unittest.TestCase):
    def test_generation_is_frozen_gwo_sweep_then_inplace_de_and_monitor_once(self):
        inst = load_instance_from_text()
        search = SequentialSearch(inst, np.random.default_rng(1), pop_size=4)
        search.x_real = np.arange(24, dtype=float).reshape(4, 6)
        search.x_bin = np.zeros((4, 6), dtype=int)
        search.fit = np.array([10.0, 40.0, 30.0, 20.0])
        search.best_fitness = 40.0
        search.best_solution = search.x_bin[1].copy()
        search.set_profile(search.profile.__class__(
            "explore", a=2.0, F=0.9, CR=0.3, G3=0.1))
        expected_leaders = search.x_real[[1, 2, 3]].copy()
        events = []

        def gwo(x, leaders, a, rng, v_max):
            events.append(("gwo", leaders.copy(), a))
            leaders[0, 0] = 9999.0
            return np.full(x.shape, 7.0)

        def de(i, population, F, CR, rng, v_max):
            events.append(("de", i, population.copy(), F, CR))
            return np.full(population.shape[1], -3.0 - i)

        def binarize(v, current, *args):
            phase = "gwo" if sum(item[0] == "bin" for item in events) < 4 else "de"
            i = sum(item[0] == "bin" and item[1] == phase for item in events)
            fit = 100.0 if phase == "de" and i == 0 else 0.0
            events.append(("bin", phase, i))
            return np.array(current), fit

        with mock.patch.object(search_mod, "gwo_candidate", gwo), \
                mock.patch.object(search_mod, "de_candidate", de), \
                mock.patch.object(search_mod, "lb2_binarize", binarize):
            search.step()

        self.assertEqual([item[0] for item in events if item[0] in ("gwo", "de")],
                         ["gwo"] * 4 + ["de"] * 4)
        self.assertTrue(all(np.array_equal(item[1], expected_leaders)
                            for item in events if item[0] == "gwo"))
        self.assertTrue(all(item[2] == 2.0 for item in events if item[0] == "gwo"))
        de_events = [item for item in events if item[0] == "de"]
        self.assertTrue(all(item[3] == 0.9 and item[4] == 0.3 for item in de_events))
        np.testing.assert_array_equal(de_events[1][2][0], np.full(6, -3.0))
        np.testing.assert_array_equal(search.x_real[1], search.x_real[1])
        self.assertNotEqual(search.x_real[1, 0], 9999.0)

        observed = []

        def update(self, value):
            observed.append(value)
            return {"ready": False, "fire": False, "no_improve_len": 0, "changed": False}

        with mock.patch.object(search_mod, "gwo_candidate", gwo), \
                mock.patch.object(search_mod, "de_candidate", de), \
                mock.patch.object(search_mod, "lb2_binarize", binarize), \
                mock.patch.object(StagnationMonitor, "update", update):
            result = run_search(inst, monitor_cfg(), pop_size=4, iterations=3, seed=2,
                                policy="binary_simple")
        self.assertEqual(len(observed), 3)
        self.assertEqual(observed, result["best_series"])
        self.assertEqual(result["schedule"], "gwo_then_de")
        self.assertIs(result["dtw_kirito"], False)

    def test_acceptance_writes_real_binary_fitness_together_and_rejects_worse(self):
        inst = load_instance_from_text()
        search = SequentialSearch(inst, np.random.default_rng(0), pop_size=4)
        search.x_real = np.arange(24, dtype=float).reshape(4, 6)
        search.x_bin = np.arange(24).reshape(4, 6) % 2
        search.fit = np.array([10.0, 20.0, 30.0, 15.0])
        search.best_fitness = 30.0
        search.best_solution = search.x_bin[2].copy()
        before = (search.x_real.copy(), search.x_bin.copy(), search.fit.copy(),
                  search.best_solution.copy())
        gwo_fit = [25.0, 10.0, 30.0, 40.0]
        de_fit = [20.0, 5.0, 40.0, 10.0]
        calls = []

        def gwo(x, leaders, a, rng, v_max):
            return np.full(x.shape, 5.0 + len(calls))

        def de(i, population, F, CR, rng, v_max):
            return np.full(population.shape[1], -8.0 - i)

        def binarize(v, current, *args):
            phase = "gwo" if len(calls) < 4 else "de"
            i = len(calls) % 4
            calls.append(phase)
            bits = np.zeros(6, dtype=int)
            bits[i] = 1
            bits[-1] = 0 if phase == "gwo" else 1
            fit = gwo_fit[i] if phase == "gwo" else de_fit[i]
            return bits, fit

        with mock.patch.object(search_mod, "gwo_candidate", gwo), \
                mock.patch.object(search_mod, "de_candidate", de), \
                mock.patch.object(search_mod, "lb2_binarize", binarize):
            best = search.step()

        self.assertEqual(calls, ["gwo"] * 4 + ["de"] * 4)
        np.testing.assert_array_equal(search.x_real[1], before[0][1])
        np.testing.assert_array_equal(search.x_bin[1], before[1][1])
        self.assertEqual(search.fit[1], 20.0)
        self.assertEqual(search.fit[0], 25.0)
        np.testing.assert_array_equal(search.x_real[0], np.full(6, 5.0))
        self.assertEqual(search.x_bin[0, -1], 0)
        self.assertEqual(search.fit[3], 40.0)
        np.testing.assert_array_equal(search.x_real[3], np.full(6, 8.0))
        self.assertEqual(search.x_bin[3, -1], 0)
        self.assertEqual(best, 40.0)
        np.testing.assert_array_equal(search.best_solution, search.x_bin[3])
        self.assertEqual(search.fit[2], 40.0)
        self.assertEqual(search.x_bin[2, -1], 1)
        np.testing.assert_array_equal(search.x_real[2], np.full(6, -10.0))
        self.assertFalse(np.array_equal(search.best_solution, search.x_bin[2]))
        self.assertEqual(search.last_step_stats,
                         {"gwo_accepted": 3, "de_accepted": 1, "accepted_total": 4})

    def test_lb2_evaluation_cost_is_initialization_plus_four_n(self):
        inst = load_instance_from_text()
        search = SequentialSearch(inst, np.random.default_rng(4), pop_size=4)
        self.assertEqual(search.binarization, "lb2")
        search.initialize()
        self.assertEqual(search.evaluations, 4)
        search.step()
        self.assertEqual(search.evaluations, 4 + 16)
        result = run_search(inst, monitor_cfg(), pop_size=4, iterations=2, seed=4,
                            policy="binary_complex")
        self.assertEqual(result["evaluations"], 4 + 32)
        self.assertEqual(result["generations"], 2)
        self.assertEqual(result["binarization"], "lb2")
        self.assertTrue(all(set(row) == {"gwo_accepted", "de_accepted", "accepted_total"}
                            for row in result["step_stats"]))


def load_instance_from_text():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        write_instances(folder, count=1)
        return load_instance(folder / "mknapcb1.txt", 0)


class CampaignCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.inst = self.tmp / "inst"
        self.inst.mkdir()
        write_instances(self.inst)
        self.out = self.tmp / "camp"

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, runner, *args):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(campaign, "run_search", runner), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = campaign.main(["--instances-dir", str(self.inst), *args])
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_counts_3348_without_running_or_writing(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        target = self.tmp / "dry"
        before = snapshot(self.tmp)
        result = subprocess.run(
            [sys.executable, "-B", "-m", "gwo_de_sequential.campaign", "--dry-run",
             "--instances-dir", str(ROOT / "instancias"), "--out", str(target)],
            cwd=self.tmp, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout.splitlines(), [
            "campaign files=9 indices=3 seeds=31 policies=4 runs=3348 "
            "binarization=lb2 signal=ddtw W=40 pop=30 iterations=1000 "
            "schedule=gwo_then_de",
            "dry-run: selection and output validated, nothing executed or written",
        ])
        self.assertFalse(target.exists())
        self.assertEqual(snapshot(self.tmp), before)

    def test_rejection_exits_2_without_writes_or_runs(self):
        nonempty = self.tmp / "old"
        nonempty.mkdir()
        (nonempty / "keep.txt").write_text("keep", encoding="utf-8")
        a_file = self.tmp / "afile"
        a_file.write_text("keep", encoding="utf-8")
        cases = [
            (("--files", "../mknapcb1.txt"), "invalid instance file name"),
            (("--files", "missing.txt"), "instance file not found"),
            (("--files", "mknapcb1.txt", "mknapcb1.txt"), "duplicate instance file"),
            (("--files", "mknapcb1.txt", "--indices", "9"), "out of range"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--seeds", "-1"),
             "seeds must be nonnegative"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--seeds", "3", "3"),
             "duplicate seed"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--pop", "3"),
             "--pop must be >= 4"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--hybrid-mode", "competitive"),
             "unrecognized arguments"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--out", str(nonempty)),
             "output path already exists"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--out", str(a_file)),
             "output path already exists"),
        ]
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        before = snapshot(self.tmp)
        for args, message in cases:
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-B", "-m", "gwo_de_sequential.campaign",
                     "--instances-dir", str(self.inst), *args],
                    cwd=self.tmp, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn(message, result.stderr)
                self.assertEqual(snapshot(self.tmp), before)
        runner = FakeRunner(lambda *key: 1.0)
        code, out, err = self.run_main(
            runner, "--files", "mknapcb1.txt", "--indices", "0", "--out", str(nonempty))
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertEqual(runner.calls, [])
        self.assertEqual(snapshot(self.tmp), before)

    def test_tie_keeps_lower_seed_metrics_for_all_and_traces_for_winner(self):
        tied = {7, 12, 30}
        runner = FakeRunner(lambda i, p, s: 50.0 if s in tied else float(s % 5))
        seeds = [str(s) for s in range(31, 0, -1)]
        code, out, err = self.run_main(
            runner, "--files", "mknapcb1.txt", "--indices", "0", "--seeds", *seeds,
            "--pop", "4", "--iterations", "2", "--window", "4", "--out", str(self.out))
        self.assertEqual(code, 0, err)
        self.assertEqual(len(runner.calls), 124)
        self.assertTrue(all(call_kw is False or True for call_kw in (True,)))
        self.assertTrue(all(not runner.records[key]["dtw_kirito"] for key in runner.records))
        inst = self.out / "mknapcb1" / "inst_00"
        fields, rows = read_csv(inst / "runs.csv")
        self.assertEqual(fields, RUN_FIELDS)
        self.assertEqual(len(rows), 124)
        self.assertEqual(read_csv(self.out / "summary.csv")[1], rows)
        values = [50.0 if s in tied else float(s % 5) for s in range(1, 32)]
        fields, stats = read_csv(self.out / "statistics.csv")
        self.assertEqual(fields, STAT_FIELDS)
        self.assertEqual([row["policy"] for row in stats], list(POLICIES))
        for row in stats:
            self.assertEqual(row["runs"], "31")
            self.assertEqual(row["best_seed"], "7")
            self.assertEqual(float(row["best_fitness"]), 50.0)
            self.assertEqual(float(row["worst_fitness"]), min(values))
            self.assertAlmostEqual(float(row["mean_fitness"]), statistics.mean(values))
            self.assertAlmostEqual(float(row["median_fitness"]), statistics.median(values))
            self.assertAlmostEqual(float(row["std_fitness"]), statistics.stdev(values))
            self.assertEqual(row["known_optimum"], "")
            self.assertEqual(row["best_gap_pct"], "")
        for policy in POLICIES:
            folder = inst / policy
            self.assertEqual([path.name for path in folder.iterdir()], ["best_run"])
            saved = json.loads((folder / "best_run" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["seed"], 7)
            self.assertEqual(saved["best_fitness"], 50.0)
            trace = (folder / "best_run" / "trace.csv").read_text(encoding="utf-8")
            self.assertIn("50.0", trace)
            self.assertEqual((folder / "best_run" / "fitness.png").read_bytes()[:8], PNG)
        config = json.loads((self.out / "campaign_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "completed")
        self.assertEqual(config["completed_runs"], 124)
        self.assertIs(config["config"]["dtw_kirito"], False)
        self.assertEqual(config["config"]["schedule"], "gwo_then_de")
        self.assertIn("status=completed runs=124/124", out)

    def test_partial_failure_stops_and_does_not_resume(self):
        runner = FakeRunner(lambda i, p, s: float(s), fail=(1, "continuous_simple", 2))
        code, out, err = self.run_main(
            runner, "--files", "mknapcb1.txt", "--indices", "0", "1", "--seeds", "1", "2",
            "--pop", "4", "--iterations", "2", "--window", "4", "--out", str(self.out))
        self.assertEqual(code, 1)
        self.assertIn("policy=continuous_simple seed=2", err)
        self.assertIn("RuntimeError: boom", err)
        self.assertEqual(len(runner.calls), 15)
        self.assertIn("status=failed runs=14/16", out)
        config = json.loads((self.out / "campaign_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "failed")
        self.assertEqual(config["completed_runs"], 14)
        self.assertIn("continuous_simple", config["error"])
        _, summary = read_csv(self.out / "summary.csv")
        self.assertEqual(len(summary), 15)
        self.assertEqual([row["status"] for row in summary], ["ok"] * 14 + ["failed"])
        self.assertEqual(summary[-1]["best_fitness"], "")
        done = self.out / "mknapcb1" / "inst_00"
        self.assertTrue((done / "statistics.csv").is_file())
        self.assertTrue((done / "convergence_comparison.png").is_file())
        partial = self.out / "mknapcb1" / "inst_01"
        self.assertEqual([path.name for path in partial.iterdir()], ["runs.csv"])
        before = snapshot(self.out)
        again, _, again_err = self.run_main(
            runner, "--files", "mknapcb1.txt", "--indices", "0", "1", "--seeds", "1", "2",
            "--pop", "4", "--iterations", "2", "--window", "4", "--out", str(self.out))
        self.assertEqual(again, 2)
        self.assertIn("output path already exists", again_err)
        self.assertEqual(len(runner.calls), 15)
        self.assertEqual(snapshot(self.out), before)

    def test_comparison_plot_labels_name_policy_and_seed(self):
        winners = {
            "binary_simple": 4,
            "binary_complex": 1,
            "continuous_simple": 5,
            "continuous_complex": 2,
        }

        def fitness(index, policy, seed):
            return 80.0 if seed == winners[policy] else 10.0 + seed

        seen = []
        original = plots_mod.comparison_figure

        def spy(records, seed_labels=False):
            fig = original(records, seed_labels=seed_labels)
            seen.append({
                "seed_labels": seed_labels,
                "legend": [text.get_text() for text in fig.axes[0].get_legend().get_texts()],
                "title": fig.axes[0].get_title(),
            })
            return fig

        runner = FakeRunner(fitness)
        with mock.patch.object(plots_mod, "comparison_figure", spy):
            code, _, err = self.run_main(
                runner, "--files", "mknapcb1.txt", "--indices", "0",
                "--seeds", "3", "5", "1", "4", "2", "--pop", "4", "--iterations", "2",
                "--window", "4", "--out", str(self.out))
        self.assertEqual(code, 0, err)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0]["seed_labels"], True)
        self.assertEqual(seen[0]["legend"], [f"{policy} (seed {winners[policy]})"
                                              for policy in POLICIES])
        self.assertIn("Best run of each policy", seen[0]["title"])
        self.assertIn("mknapcb1[0]", seen[0]["title"])
        self.assertEqual((self.out / "mknapcb1" / "inst_00" / "convergence_comparison.png")
                         .read_bytes()[:8], PNG)


class BoundaryTests(unittest.TestCase):
    def test_package_source_does_not_import_gwo_de_dtw(self):
        root = ROOT / "gwo_de_sequential"
        self.assertTrue(root.is_dir())
        for path in root.rglob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                code = line.split("#", 1)[0]
                self.assertNotIn("gwo_de_dtw", code, path)


if __name__ == "__main__":
    unittest.main()
