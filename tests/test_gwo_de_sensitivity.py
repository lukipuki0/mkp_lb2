"""Public checks for the MKP-only descriptive OAT sensitivity campaign."""

import contextlib
import csv
import io
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / "instancias" / "mknapcb1.txt"
DRY = "dry-run: selection and output validated, nothing executed or written"
NAMES = (
    "baseline,window_20,window_80,window_160,band_1,band_2,band_8,"
    "p_low_20,p_low_30,p_low_50,p_high_50,p_high_70,p_high_80,"
    "plateau_max_5,plateau_max_10,plateau_max_30,patience_2,patience_4,patience_16"
)


def _plan(indices, seeds, configurations, pop, iterations):
    runs = len(indices) * len(seeds) * len(configurations) * 2
    names = ",".join(configurations)
    return (
        f"sensitivity file=instancias/mknapcb1.txt indices={len(indices)} "
        f"seeds={len(seeds)} policies=2 configurations={len(configurations)} "
        f"runs={runs} pop={pop} iterations={iterations} binarization=lb2 "
        f"signal=ddtw hybrid_mode=competitive names={names}"
    )


class DesignTests(unittest.TestCase):
    def test_nineteen_oat_configs_keep_absolute_band_and_count_3534(self):
        from gwo_de_dtw.sensitivity.config import (
            CONFIGURATIONS, FULL_RUNS, INDICES, POLICIES, SEEDS, oat_problems,
        )
        from gwo_de_dtw import config as hybrid_config

        before = (hybrid_config.WINDOW, hybrid_config.BAND, hybrid_config.P_LOW,
                  hybrid_config.POLICY)
        self.assertEqual(POLICIES, ("binary_simple", "binary_complex"))
        self.assertEqual(INDICES, (0, 15, 29))
        self.assertEqual(SEEDS, tuple(range(1, 32)))
        self.assertEqual(len(CONFIGURATIONS), 19)
        self.assertEqual([item.name for item in CONFIGURATIONS], NAMES.split(","))
        self.assertEqual(FULL_RUNS, 3534)
        self.assertEqual(19 * 2 * 3 * 31, 3534)
        baseline = CONFIGURATIONS[0]
        self.assertEqual(oat_problems(baseline, baseline), [])
        seen = set()
        alive = []
        for spec in CONFIGURATIONS:
            self.assertEqual(oat_problems(spec, baseline), [])
            cfg = spec.stagnation_config()
            alive.append(cfg)
            self.assertIs(cfg.use_ddtw, True)
            self.assertIs(cfg.adapt_thresholds, True)
            self.assertEqual(cfg.min_slope, 2.0)
            self.assertEqual(cfg.improvement_tol, 0.0)
            self.assertNotIn(id(cfg), seen)
            seen.add(id(cfg))
            if spec.parameter == "window" or spec.name == "baseline":
                self.assertEqual(cfg.band, 4)
            if spec.name == "window_20":
                self.assertEqual(cfg.window, 20)
                self.assertEqual(cfg.band, 4)
            if spec.name == "window_160":
                self.assertEqual(cfg.window, 160)
                self.assertNotEqual(cfg.band, 16)
            if spec.name == "band_1":
                self.assertEqual((cfg.band, cfg.window), (1, 40))
            if spec.name == "p_low_50":
                self.assertEqual(cfg.p_low, 50.0)
                self.assertEqual(cfg.p_high, 60.0)
        self.assertEqual(
            (hybrid_config.WINDOW, hybrid_config.BAND, hybrid_config.P_LOW,
             hybrid_config.POLICY),
            before,
        )
        text = (ROOT / "gwo_de_dtw" / "sensitivity" / "config.py").read_text(encoding="utf-8")
        self.assertNotIn("gwo_de_dtw.config", text)

    def test_descriptive_summaries_do_not_infer_or_pool(self):
        from gwo_de_dtw.sensitivity.analysis import comparison_table, summarize

        rows = [
            {"best_fitness": 10.0, "seed": 3, "gap_pct": 1.0, "elapsed_s": 2.0},
            {"best_fitness": 30.0, "seed": 1, "gap_pct": 3.0, "elapsed_s": 4.0},
            {"best_fitness": 20.0, "seed": 2, "gap_pct": 2.0, "elapsed_s": 6.0},
        ]
        stats = summarize(rows)
        self.assertEqual(stats["best_fitness"], 30.0)
        self.assertEqual(stats["worst_fitness"], 10.0)
        self.assertEqual(stats["best_seed"], 1)
        self.assertAlmostEqual(stats["mean_fitness"], 20.0)
        self.assertAlmostEqual(stats["median_fitness"], 20.0)
        self.assertAlmostEqual(stats["std_fitness"], statistics.stdev([10.0, 30.0, 20.0]))
        self.assertAlmostEqual(stats["mean_gap_pct"], 2.0)
        tied = [
            {"best_fitness": 5.0, "seed": 9, "gap_pct": 1.0, "elapsed_s": 1.0},
            {"best_fitness": 5.0, "seed": 2, "gap_pct": 1.0, "elapsed_s": 1.0},
            {"best_fitness": 4.0, "seed": 1, "gap_pct": 1.0, "elapsed_s": 1.0},
        ]
        self.assertEqual(summarize(tied)["best_seed"], 2)
        lone = summarize([{"best_fitness": 7.0, "seed": 4, "gap_pct": None, "elapsed_s": 1.5}])
        self.assertEqual(lone["std_fitness"], "")
        self.assertEqual(lone["mean_gap_pct"], "")
        self.assertEqual(lone["best_gap_pct"], "")
        table = comparison_table([
            {"instance": "mknapcb1.txt", "index": 0, "policy": "binary_simple",
             "configuration": "baseline", "parameter": "baseline", "parameter_value": "",
             "mean_fitness": 10.0},
            {"instance": "mknapcb1.txt", "index": 0, "policy": "binary_simple",
             "configuration": "window_20", "parameter": "window", "parameter_value": 20,
             "mean_fitness": 14.0},
            {"instance": "mknapcb1.txt", "index": 29, "policy": "binary_simple",
             "configuration": "baseline", "parameter": "baseline", "parameter_value": "",
             "mean_fitness": 100.0},
            {"instance": "mknapcb1.txt", "index": 29, "policy": "binary_simple",
             "configuration": "window_20", "parameter": "window", "parameter_value": 20,
             "mean_fitness": 90.0},
        ])
        by_index = {row["index"]: row for row in table if row["configuration"] == "window_20"}
        self.assertEqual(by_index[0]["mean_fitness_delta_vs_baseline"], 4.0)
        self.assertEqual(by_index[29]["mean_fitness_delta_vs_baseline"], -10.0)
        self.assertNotEqual(by_index[0]["baseline_mean_fitness"], by_index[29]["baseline_mean_fitness"])
        missing = comparison_table([
            {"instance": "mknapcb1.txt", "index": 0, "policy": "binary_complex",
             "configuration": "window_20", "parameter": "window", "parameter_value": 20,
             "mean_fitness": 3.0},
        ])
        self.assertEqual(missing[0]["mean_fitness_delta_vs_baseline"], "")
        source = (ROOT / "gwo_de_dtw" / "sensitivity" / "analysis.py").read_text(encoding="utf-8")
        self.assertNotIn("scipy", source)
        self.assertNotIn("p_value", source)
        self.assertNotIn("wilcoxon", source)

    def test_sensitivity_plot_is_baseline_zero_and_closes_only_its_figure(self):
        from gwo_de_dtw.sensitivity import analysis
        from gwo_de_dtw.sensitivity.analysis import save_sensitivity_plot, sensitivity_figure

        rows = [
            {"policy": "binary_simple", "parameter": "window", "parameter_value": 20,
             "index": 0, "mean_fitness_delta_vs_baseline": 1.0},
            {"policy": "binary_simple", "parameter": "window", "parameter_value": 20,
             "index": 29, "mean_fitness_delta_vs_baseline": -2.0},
            {"policy": "binary_simple", "parameter": "baseline", "parameter_value": "",
             "index": 0, "mean_fitness_delta_vs_baseline": 0.0},
        ]
        fig = sensitivity_figure(rows, "binary_simple")
        try:
            title = fig._suptitle.get_text()
            self.assertIn("Not a global-robustness claim", title)
            self.assertNotIn("optimum", title.lower())
            labels = []
            window_ax = next(ax for ax in fig.axes if ax.get_title() == "window")
            zero = [line for line in window_ax.get_lines()
                    if set(np.asarray(line.get_ydata()).tolist()) == {0.0}]
            self.assertTrue(zero)
            legend = window_ax.get_legend()
            self.assertIsNotNone(legend)
            labels.extend(text.get_text() for text in legend.get_texts())
            self.assertIn("mknapcb1.txt[0]", labels)
            self.assertIn("mknapcb1.txt[29]", labels)
        finally:
            analysis.plt.close(fig)
        held = analysis.plt.figure()
        before = set(analysis.plt.get_fignums())
        with tempfile.TemporaryDirectory() as tmp:
            path = save_sensitivity_plot(rows, "binary_simple", Path(tmp))
            self.assertEqual(Path(path).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            saved = {}
            real = analysis.plt.Figure.savefig

            def spy(self, *args, **kwargs):
                saved["dpi"] = kwargs.get("dpi")
                raise RuntimeError("disk")

            with mock.patch.object(analysis.plt.Figure, "savefig", spy):
                with self.assertRaises(RuntimeError):
                    save_sensitivity_plot(rows, "binary_simple", Path(tmp))
            self.assertEqual(saved["dpi"], 150)
        self.assertEqual(set(analysis.plt.get_fignums()), before)
        self.assertIn(held.number, analysis.plt.get_fignums())
        analysis.plt.close(held)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.out = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, runner, *args):
        from gwo_de_dtw.sensitivity import campaign

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(campaign, "run_hybrid", runner), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = campaign.main(list(args))
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_dry_run_and_preflight_reject_before_writes(self):
        from gwo_de_dtw.sensitivity import campaign
        from gwo_de_dtw.sensitivity.config import CONFIGURATIONS

        env = dict(os.environ, PYTHONPATH=str(ROOT))
        target = self.tmp / "dry"
        before = set(Path("gwo_de_dtw/results").glob("sensitivity_*")) if Path("gwo_de_dtw/results").exists() else set()
        result = subprocess.run(
            [sys.executable, "-B", "-m", "gwo_de_dtw.sensitivity.campaign", "--dry-run",
             "--out", str(target)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout.splitlines(), [_plan((0, 15, 29), tuple(range(1, 32)), NAMES.split(","), 30, 1000), DRY])
        self.assertFalse(target.exists())
        after = set(Path("gwo_de_dtw/results").glob("sensitivity_*")) if Path("gwo_de_dtw/results").exists() else set()
        self.assertEqual(before, after)
        idx = subprocess.run(
            [sys.executable, "-B", "-m", "gwo_de_dtw.sensitivity.campaign", "--dry-run",
             "--indices", "29", "--seeds", "1", "--configurations", "baseline", "window_20",
             "--pop", "4", "--iterations", "2", "--out", str(self.tmp / "idx29")],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(idx.returncode, 0, idx.stderr)
        self.assertIn("runs=4", idx.stdout)
        self.assertFalse((self.tmp / "idx29").exists())
        self.assertTrue(INSTANCE.is_file())

        nonempty = self.tmp / "kept"
        nonempty.mkdir()
        (nonempty / "keep.txt").write_text("keep", encoding="utf-8")
        a_file = self.tmp / "afile"
        a_file.write_text("keep", encoding="utf-8")
        calls = []

        def runner(*args, **kwargs):
            calls.append(args)
            raise AssertionError("optimizer must not run")

        cases = [
            (["--indices", "0", "0", "--seeds", "1", "--pop", "4", "--iterations", "1",
              "--out", str(self.tmp / "a")], "duplicate index"),
            (["--indices", "0", "--seeds", "-1", "--pop", "4", "--iterations", "1",
              "--out", str(self.tmp / "b")], "seeds must be nonnegative"),
            (["--indices", "0", "--seeds", "1", "1", "--pop", "4", "--iterations", "1",
              "--out", str(self.tmp / "c")], "duplicate seed"),
            (["--indices", "0", "--seeds", "1", "--pop", "3", "--iterations", "1",
              "--out", str(self.tmp / "d")], "--pop must be >= 4"),
            (["--indices", "0", "--seeds", "1", "--pop", "4", "--iterations", "0",
              "--out", str(self.tmp / "e")], "--iterations must be positive"),
            (["--indices", "30", "--seeds", "1", "--pop", "4", "--iterations", "1",
              "--out", str(self.tmp / "f")], "out of range"),
            (["--indices", "29", "--seeds", "1", "--configurations", "baseline", "baseline",
              "--pop", "4", "--iterations", "1", "--out", str(self.tmp / "g")],
             "duplicate configuration"),
            (["--indices", "29", "--seeds", "1", "--configurations", "nope",
              "--pop", "4", "--iterations", "1", "--out", str(self.tmp / "h")],
             "unknown configuration"),
            (["--indices", "29", "--seeds", "1", "--pop", "4", "--iterations", "1",
              "--window", "10", "--out", str(self.tmp / "i")], "unrecognized arguments"),
            (["--indices", "29", "--seeds", "1", "--pop", "4", "--iterations", "1",
              "--out", str(nonempty)], "output directory is not empty"),
            (["--indices", "29", "--seeds", "1", "--pop", "4", "--iterations", "1",
              "--out", str(a_file)], "output path is not a directory"),
        ]
        for args, message in cases:
            with self.subTest(message=message):
                code, out, err = self.run_main(runner, *args)
                self.assertEqual(code, 2, err)
                self.assertEqual(out, "")
                self.assertIn(message, err)
                self.assertEqual(calls, [])
        self.assertEqual((nonempty / "keep.txt").read_text(encoding="utf-8"), "keep")

        class Bad:
            name = "window_20"
            parameter = "window"
            value = 20
            window = 20
            band = 8
            p_low = 40.0
            p_high = 60.0
            plateau_max = 15
            patience = 8

            def stagnation_config(self):
                raise AssertionError("optimizer must not run")

        broken = (CONFIGURATIONS[0], Bad())
        with mock.patch.object(campaign, "CONFIGURATIONS", broken):
            code, out, err = self.run_main(
                runner, "--dry-run", "--indices", "29", "--seeds", "1",
                "--configurations", "window_20", "--pop", "4", "--iterations", "1",
                "--out", str(self.tmp / "oat"),
            )
        self.assertEqual(code, 2, err)
        self.assertIn("OAT", err)
        self.assertFalse((self.tmp / "oat").exists())
        self.assertEqual(calls, [])

    def _runner(self, fitness, gap=1.5, fail_at=None, interrupt_at=None):
        from dataclasses import asdict

        calls = []

        def runner(inst, monitor_cfg, **kwargs):
            calls.append((inst, monitor_cfg, kwargs))
            if fail_at is not None and len(calls) == fail_at:
                raise RuntimeError("boom")
            if interrupt_at is not None and len(calls) == interrupt_at:
                raise KeyboardInterrupt()
            seed = kwargs["seed"]
            value = fitness(kwargs["policy"], kwargs.get("configuration_name"), seed, monitor_cfg)
            return {
                "policy": kwargs["policy"],
                "seed": seed,
                "best_fitness": value,
                "best_solution": np.array([1, 0, seed], dtype=int),
                "known_optimum": 0.0 if gap is None else inst["optimo"],
                "gap_pct": gap,
                "evaluations": 12,
                "changes": [3, 4],
                "monitor": [{"ready": False, "token": "genuine"}],
                "parameter_series": {"a": [0.5], "F": [0.5], "CR": [0.9], "G3": [0.1]},
                "best_series": [float(value)],
                "profile_series": ["exploit"],
                "monitor_config": asdict(monitor_cfg),
                "dtw_kirito": kwargs["dtw_kirito"],
                "hybrid_mode": kwargs["hybrid_mode"],
                "binarization": kwargs["binarization"],
            }

        runner.calls = calls
        return runner

    def test_winners_records_and_failure_keep_completed_work(self):
        from gwo_de_dtw.sensitivity import campaign

        def fitness(policy, name, seed, cfg):
            if seed == 8:
                return 50.0
            if seed == 2:
                return 50.0
            return float(seed)

        runner = self._runner(fitness, gap=None)
        plots = []
        held = campaign.plt.figure()

        def leaky(res, output_dir):
            campaign.plt.figure()
            plots.append((res["policy"], res["configuration"], res["instance"], res["monitor"], output_dir))
            (Path(output_dir) / "marker.txt").write_text(res["policy"], encoding="utf-8")

        with mock.patch.object(campaign, "PLOTS", (leaky,)):
            code, out, err = self.run_main(
                runner, "--indices", "29", "--seeds", "8", "2", "1",
                "--configurations", "window_20", "baseline",
                "--pop", "4", "--iterations", "2", "--out", str(self.out),
            )
        self.assertEqual(code, 0, err)
        self.assertIn("status=completed runs=12/12", out)
        self.assertEqual(len(runner.calls), 12)
        self.assertEqual(len({id(cfg) for _, cfg, _ in runner.calls}), 12)
        windows = [cfg.window for _, cfg, kw in runner.calls if kw["policy"] == "binary_simple"]
        self.assertEqual(windows.count(40), 3)
        self.assertTrue(all(cfg.band == 4 for _, cfg, _ in runner.calls))
        for _, _, kw in runner.calls:
            self.assertEqual(kw["G1"], 0.5)
            self.assertEqual(kw["G2"], 0.5)
            self.assertEqual(kw["g3_exploit"], 0.1)
            self.assertEqual(kw["g3_explore"], 0.1)
            self.assertEqual(kw["binarization"], "lb2")
            self.assertIs(kw["dtw_kirito"], False)
            self.assertEqual(kw["hybrid_mode"], "competitive")
            self.assertEqual(kw["pop_size"], 4)
            self.assertEqual(kw["iterations"], 2)
            self.assertIn(kw["policy"], ("binary_simple", "binary_complex"))
        first_plot_configs = {item[1] for item in plots}
        self.assertEqual(first_plot_configs, {"baseline", "window_20"})
        self.assertTrue(all(item[2] == "mknapcb1.txt[29]" for item in plots))
        self.assertEqual(plots[0][3], [{"ready": False, "token": "genuine"}])
        self.assertIn(held.number, campaign.plt.get_fignums())
        campaign.plt.close(held)
        self.assertEqual(campaign.plt.get_fignums(), [])

        with self.out.joinpath("runs.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["gap_pct"] == "" for row in rows))
        self.assertTrue(all(row["known_optimum"] == "" for row in rows))
        self.assertTrue(all(row["profile_changes"] == "2" for row in rows))
        self.assertTrue(all(row["elapsed_s"] != "" for row in rows))
        self.assertEqual(sum(row["configuration"] == "baseline" for row in rows), 6)
        jsonl = [json.loads(line) for line in self.out.joinpath("runs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(jsonl), 12)
        self.assertNotIn("monitor", jsonl[0])
        self.assertIsNone(jsonl[0]["gap_pct"])
        winner = json.loads((self.out / "mknapcb1" / "inst_29" / "binary_simple" / "baseline" / "best_run" / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(winner["seed"], 2)
        self.assertEqual(winner["best_fitness"], 50.0)
        self.assertEqual(winner["best_solution"], [1, 0, 2])
        self.assertEqual(winner["policy"], "binary_simple")
        self.assertEqual(winner["monitor"], [{"ready": False, "token": "genuine"}])
        self.assertEqual(winner["monitor_config"]["band"], 4)
        self.assertTrue((self.out / "mknapcb1" / "inst_29" / "binary_simple" / "baseline" / "best_run" / "marker.txt").is_file())
        self.assertTrue((self.out / "plots" / "sensitivity_binary_simple.png").is_file())
        self.assertTrue((self.out / "plots" / "sensitivity_binary_complex.png").is_file())
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["effective_config"]["binarization"], "lb2")
        self.assertEqual(manifest["effective_config"]["hybrid_mode"], "competitive")
        self.assertIs(manifest["effective_config"]["dtw_kirito"], False)
        self.assertEqual(manifest["effective_config"]["monitor_by_configuration"]["window_20"]["band"], 4)
        self.assertEqual(manifest["effective_config"]["monitor_by_configuration"]["window_20"]["window"], 20)
        self.assertEqual(manifest["provenance"]["seed_paired_initial_conditions"], "not_claimed")
        self.assertIs(manifest["provenance"]["inferential_tests"], False)
        self.assertIn("numpy", manifest["runtime"])
        self.assertIn("python", manifest["runtime"])
        evidence = {item["index"]: item for item in manifest["optimum_evidence"]}
        self.assertIn(29, evidence)
        self.assertIn(evidence[29]["reference"], ("known", "unknown"))
        self.assertIsInstance(manifest["timing"]["elapsed_s"], float)
        text = (self.out / "resumen.txt").read_text(encoding="utf-8")
        self.assertIn("descriptive", text.lower())
        self.assertIn("seconds", text.lower())
        self.assertNotIn("global robustness", text.lower())
        with (self.out / "statistics.csv").open(encoding="utf-8", newline="") as fh:
            stats = list(csv.DictReader(fh))
        baseline_row = next(row for row in stats if row["policy"] == "binary_simple" and row["configuration"] == "baseline")
        self.assertEqual(baseline_row["best_seed"], "2")
        self.assertAlmostEqual(float(baseline_row["std_fitness"]), statistics.stdev([50.0, 50.0, 1.0]))
        self.assertEqual(baseline_row["mean_gap_pct"], "")
        self.assertEqual(baseline_row["mean_fitness_delta_vs_baseline"], "0.0")
        package = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "gwo_de_dtw" / "sensitivity").glob("*.py"))
        self.assertNotIn("close('all'", package)
        self.assertNotIn('close("all"', package)
        self.assertNotIn("HRES2", package)
        self.assertNotIn("import sensitivity", package)

        again, _, again_err = self.run_main(
            runner, "--indices", "29", "--seeds", "8", "2", "1",
            "--configurations", "baseline", "--pop", "4", "--iterations", "2",
            "--out", str(self.out),
        )
        self.assertEqual(again, 2)
        self.assertIn("output directory is not empty", again_err)
        self.assertEqual(len(runner.calls), 12)

    def test_single_seed_sd_blank_and_plot_failure_keeps_every_winner(self):
        from gwo_de_dtw.sensitivity import campaign

        runner = self._runner(lambda policy, name, seed, cfg: 3.0, gap=4.0)
        seen = {"calls": 0}

        def plotting(res, output_dir):
            seen["calls"] += 1
            if seen["calls"] == 1:
                for name in ("baseline", "window_20"):
                    path = self.out / "mknapcb1" / "inst_29" / "binary_simple" / name / "best_run" / "result.json"
                    self.assertTrue(path.is_file(), name)
                state = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "running")
                raise RuntimeError("plot broke")

        with mock.patch.object(campaign, "PLOTS", (plotting,)):
            code, out, err = self.run_main(
                runner, "--indices", "29", "--seeds", "1",
                "--configurations", "baseline", "window_20",
                "--pop", "4", "--iterations", "1", "--out", str(self.out),
            )
        self.assertEqual(code, 1)
        self.assertIn("RuntimeError: plot broke", err)
        self.assertIn("status=failed", out)
        self.assertNotIn("status=completed", out)
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("plot broke", manifest["error"])
        for policy in ("binary_simple", "binary_complex"):
            for name in ("baseline", "window_20"):
                self.assertTrue((self.out / "mknapcb1" / "inst_29" / policy / name / "best_run" / "result.json").is_file())
        self.assertFalse((self.out / "plots" / "sensitivity_binary_simple.png").exists())
        with (self.out / "statistics.csv").open(encoding="utf-8", newline="") as fh:
            stats = list(csv.DictReader(fh))
        self.assertTrue(stats)
        self.assertTrue(all(row["std_fitness"] == "" for row in stats))
        self.assertTrue(all(row["mean_gap_pct"] != "" for row in stats))

    def test_run_failure_does_not_retry_plot_or_resume(self):
        from gwo_de_dtw.sensitivity import campaign

        runner = self._runner(lambda *args: 1.0, fail_at=2)
        with mock.patch.object(campaign, "PLOTS", (lambda *a, **k: (_ for _ in ()).throw(AssertionError("plot")),)):
            code, out, err = self.run_main(
                runner, "--indices", "29", "--seeds", "1", "2",
                "--configurations", "baseline", "--pop", "4", "--iterations", "1",
                "--out", str(self.out),
            )
        self.assertEqual(code, 1)
        self.assertIn("RuntimeError: boom", err)
        self.assertEqual(len(runner.calls), 2)
        self.assertIn("status=failed runs=1/4", out)
        with self.out.joinpath("runs.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([row["status"] for row in rows], ["ok", "failed"])
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed_runs"], 1)
        self.assertIn("boom", manifest["error"])
        self.assertFalse((self.out / "plots").exists())
        again, _, again_err = self.run_main(
            runner, "--indices", "29", "--seeds", "1", "--configurations", "baseline",
            "--pop", "4", "--iterations", "1", "--out", str(self.out),
        )
        self.assertEqual(again, 2)
        self.assertIn("output directory is not empty", again_err)
        self.assertEqual(len(runner.calls), 2)

    def test_interruption_is_recorded_and_reraised(self):
        from gwo_de_dtw.sensitivity import campaign

        runner = self._runner(lambda *args: 1.0, interrupt_at=2)
        with self.assertRaises(KeyboardInterrupt):
            self.run_main(
                runner, "--indices", "29", "--seeds", "1", "2",
                "--configurations", "baseline", "--pop", "4", "--iterations", "1",
                "--out", str(self.out),
            )
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "interrupted")
        with self.out.joinpath("runs.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(len(runner.calls), 2)
        self.assertFalse((self.out / "plots").exists())

    def test_default_directory_name_and_readme_smoke_names(self):
        from gwo_de_dtw.sensitivity import campaign

        with mock.patch.object(campaign, "RESULTS_DIR", self.tmp):
            root = campaign.new_default_root()
        self.assertTrue(root.name.startswith("sensitivity_"))
        self.assertEqual(root.parent, self.tmp)
        readme = (ROOT / "gwo_de_dtw" / "sensitivity" / "README.md").read_text(encoding="utf-8")
        for name in NAMES.split(","):
            self.assertIn(name, readme)
        self.assertIn("--configurations baseline window_20", readme)
        self.assertIn("3534", readme)
        self.assertIn("not claimed", readme.lower())
        help_text = campaign.build_parser().format_help()
        self.assertIn("window_20", help_text)
        self.assertIn("baseline", help_text)


OPTIMIZER_ENV = "GWO_DE_SENSITIVITY_OPTIMIZER"
TRACE_ENV = "GWO_DE_SENSITIVITY_TRACE"
OUT_ENV = "GWO_DE_SENSITIVITY_OUT"
MODE_ENV = "GWO_DE_SENSITIVITY_FAKE_MODE"
FAKE_OPT = '''
import json
import os
import time
from pathlib import Path

def run(inst, monitor_cfg, **kwargs):
    index = int(os.environ["GWO_DE_SENSITIVITY_JOB_INDEX"])
    mode = os.environ.get("GWO_DE_SENSITIVITY_FAKE_MODE", "equiv")
    trace = Path(os.environ["GWO_DE_SENSITIVITY_TRACE"])
    trace.parent.mkdir(parents=True, exist_ok=True)
    events = trace.parent / (trace.name + ".events")
    events.mkdir(parents=True, exist_ok=True)
    def emit(event):
        payload = {
            "event": event, "pid": os.getpid(), "index": index,
            "policy": kwargs["policy"], "seed": kwargs["seed"],
            "t": time.time(),
        }
        (events / f"{time.time_ns()}-{os.getpid()}-{event}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    def started_pids():
        found = set()
        for path in events.glob("*.json"):
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("event") == "start":
                found.add(item["pid"])
        return found
    emit("start")
    if mode == "spawn":
        deadline = time.time() + 8
        while time.time() < deadline:
            if len(started_pids()) >= 2:
                emit("overlap")
                break
            time.sleep(0.02)
        else:
            emit("barrier-timeout")
    elif mode == "fail" and index == 15:
        out = Path(os.environ["GWO_DE_SENSITIVITY_OUT"])
        marker = out / "mknapcb1" / "inst_00" / "binary_simple" / "baseline" / "best_run" / "result.json"
        deadline = time.time() + 10
        while not marker.is_file() and time.time() < deadline:
            time.sleep(0.02)
        time.sleep(0.2)
        emit("fail")
        raise RuntimeError("boom")
    elif mode == "cancel" and index == 15:
        time.sleep(30)
        emit("end")
        return _result(index, monitor_cfg, kwargs)
    elif mode == "cancel" and index == 0:
        emit("fail")
        raise RuntimeError("boom")
    emit("end")
    return _result(index, monitor_cfg, kwargs)

def _result(index, monitor_cfg, kwargs):
    policy_bias = 0 if kwargs["policy"] == "binary_simple" else 10
    value = float(index) * 100 + kwargs["seed"] + policy_bias + monitor_cfg.window / 100.0
    return {
        "policy": kwargs["policy"],
        "seed": kwargs["seed"],
        "best_fitness": value,
        "best_solution": [1, 0, kwargs["seed"]],
        "known_optimum": 100.0,
        "gap_pct": 1.5,
        "evaluations": 4,
        "changes": [1],
        "monitor": [{"ready": False, "token": "spawn-fake"}],
        "parameter_series": {"a": [0.5], "F": [0.5], "CR": [0.9], "G3": [0.1]},
        "best_series": [value],
        "profile_series": ["exploit"],
        "monitor_config": {"window": monitor_cfg.window, "band": monitor_cfg.band},
        "dtw_kirito": kwargs["dtw_kirito"],
        "hybrid_mode": kwargs["hybrid_mode"],
        "binarization": kwargs["binarization"],
        "token": "spawn-fake",
    }
'''


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.out = self.tmp / "out"
        self._env = {}
        self._fake_dir = self.tmp / "fakepath"
        self._fake_dir.mkdir()
        (self._fake_dir / "fake_opt.py").write_text(FAKE_OPT, encoding="utf-8")
        self.trace = self.tmp / "trace.jsonl"
        for key in (OPTIMIZER_ENV, TRACE_ENV, OUT_ENV, MODE_ENV, "PYTHONPATH"):
            self._env[key] = os.environ.get(key)
        os.environ[OPTIMIZER_ENV] = "fake_opt.run"
        os.environ[TRACE_ENV] = str(self.trace)
        os.environ[OUT_ENV] = str(self.out)
        previous = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = str(self._fake_dir) + (os.pathsep + previous if previous else "")
        sys.path.insert(0, str(self._fake_dir))

    def tearDown(self):
        sys.path[:] = [item for item in sys.path if item != str(self._fake_dir)]
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def run_main(self, runner, *args):
        from gwo_de_dtw.sensitivity import campaign

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(campaign, "run_hybrid", runner), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = campaign.main(list(args))
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def _events(self):
        folder = self.trace.parent / (self.trace.name + ".events")
        if not folder.exists():
            return []
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]

    def test_workers_rejected_before_any_write(self):
        from gwo_de_dtw.sensitivity import campaign

        calls = []

        def runner(*args, **kwargs):
            calls.append(args)
            raise AssertionError("optimizer must not run")

        for workers in ("0", "-1"):
            with self.subTest(workers=workers):
                target = self.tmp / f"bad-{workers}"
                code, out, err = self.run_main(
                    runner, "--workers", workers, "--indices", "0", "15",
                    "--seeds", "1", "--configurations", "baseline",
                    "--pop", "4", "--iterations", "1", "--out", str(target),
                )
                self.assertEqual(code, 2, err)
                self.assertEqual(out, "")
                self.assertIn("--workers must be >= 1", err)
                self.assertFalse(target.exists())
                self.assertEqual(calls, [])
                self.assertFalse(self.trace.exists())
        code, out, err = self.run_main(
            runner, "--workers", "nope", "--dry-run",
        )
        self.assertEqual(code, 2, err)
        self.assertEqual(out, "")
        self.assertIn("invalid", err.lower())
        help_text = campaign.build_parser().format_help()
        self.assertIn("--workers", help_text)
        self.assertIn("default: 3", help_text)
        readme = (ROOT / "gwo_de_dtw" / "sensitivity" / "README.md").read_text(encoding="utf-8")
        self.assertIn("--workers", readme)
        self.assertIn("--workers 1", readme)

    def test_spawn_two_indices_are_distinct_processes(self):
        """Spawn-safe path: the fake optimizer is imported by the child, not patched in the parent."""
        from gwo_de_dtw.sensitivity import campaign

        os.environ[MODE_ENV] = "spawn"
        plots = []

        def plotting(res, output_dir):
            self.assertEqual(os.getpid(), plotting.parent_pid)
            plots.append(res.get("token"))
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "marker.txt").write_text("parent", encoding="utf-8")

        plotting.parent_pid = os.getpid()
        started = time.time()
        with mock.patch.object(campaign, "PLOTS", (plotting,)):
            code, out, err = self.run_main(
                campaign.run_hybrid, "--workers", "5", "--indices", "0", "15",
                "--seeds", "1", "--configurations", "baseline",
                "--pop", "4", "--iterations", "1", "--out", str(self.out),
            )
        elapsed = time.time() - started
        self.assertLess(elapsed, 30, err)
        self.assertEqual(code, 0, err)
        self.assertIn("status=completed", out)
        events = self._events()
        overlap = [item for item in events if item["event"] == "overlap"]
        pids = {item["pid"] for item in overlap}
        self.assertGreaterEqual(len(pids), 2, events)
        self.assertNotIn(os.getpid(), pids)
        self.assertTrue(all(item["index"] in (0, 15) for item in overlap))
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["selection"]["workers_requested"], 5)
        self.assertEqual(manifest["selection"]["workers_effective"], 2)
        self.assertEqual(manifest["completed_runs"], 4)
        self.assertEqual(manifest["status"], "completed")
        with self.out.joinpath("runs.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), manifest["completed_runs"])
        self.assertEqual([row["index"] for row in rows], ["0", "0", "15", "15"])
        for line in self.out.joinpath("runs.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            self.assertNotIn("best_series", item)
            self.assertNotIn("monitor", item)
            self.assertNotIn("best_solution", item)
        for index in (0, 15):
            winner = self.out / "mknapcb1" / f"inst_{index:02d}" / "binary_simple" / "baseline" / "best_run" / "result.json"
            self.assertEqual(json.loads(winner.read_text(encoding="utf-8"))["token"], "spawn-fake")
        self.assertTrue(plots)
        self.assertTrue(all(item == "spawn-fake" for item in plots))
        self.assertFalse((self.out / "mknapcb1" / "inst_00" / "_journal").exists())

    def test_serial_and_parallel_public_results_match(self):
        from gwo_de_dtw.sensitivity import campaign

        def plotting(res, output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "marker.txt").write_text(res["policy"], encoding="utf-8")

        args = [
            "--indices", "15", "0", "--seeds", "2", "1",
            "--configurations", "window_20", "baseline",
            "--pop", "4", "--iterations", "1",
        ]

        def once(workers, folder):
            os.environ[MODE_ENV] = "equiv"
            os.environ[OUT_ENV] = str(folder)
            self.trace = folder.parent / f"trace-{workers}.jsonl"
            os.environ[TRACE_ENV] = str(self.trace)
            with mock.patch.object(campaign, "PLOTS", (plotting,)):
                code, out, err = self.run_main(
                    campaign.run_hybrid, "--workers", str(workers), *args, "--out", str(folder),
                )
            self.assertEqual(code, 0, err)
            self.assertIn("status=completed", out)
            return self._events()

        serial_events = once(1, self.tmp / "serial")
        parallel_events = once(2, self.tmp / "parallel")
        self.assertTrue(serial_events)
        self.assertTrue(all(item["pid"] == os.getpid() for item in serial_events))
        self.assertGreaterEqual(len({item["pid"] for item in parallel_events}), 2)
        self.assertEqual(self._public_tree(self.tmp / "serial"), self._public_tree(self.tmp / "parallel"))

    def test_parallel_failure_preserves_completed_work_and_cancels_pending(self):
        from gwo_de_dtw.sensitivity import campaign

        os.environ[MODE_ENV] = "fail"
        with mock.patch.object(campaign, "PLOTS", (lambda *a, **k: (_ for _ in ()).throw(AssertionError("plot")),)):
            code, out, err = self.run_main(
                campaign.run_hybrid, "--workers", "2", "--indices", "0", "15",
                "--seeds", "1", "--configurations", "baseline",
                "--pop", "4", "--iterations", "1", "--out", str(self.out),
            )
        self.assertEqual(code, 1, err)
        self.assertIn("RuntimeError: boom", err)
        self.assertIn("status=failed", out)
        self.assertNotIn("status=completed", out)
        events = self._events()
        self.assertEqual(sum(item["index"] == 15 and item["event"] == "fail" for item in events), 1)
        manifest = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("boom", manifest["error"])
        self.assertFalse(manifest["provenance"].get("retried", False))
        with self.out.joinpath("runs.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(manifest["completed_runs"], sum(row["status"] == "ok" for row in rows))
        self.assertGreaterEqual(manifest["completed_runs"], 1)
        self.assertEqual(sum(row["status"] == "failed" for row in rows), 1)
        self.assertTrue(any(row["index"] == "0" and row["status"] == "ok" for row in rows))
        winner = self.out / "mknapcb1" / "inst_00" / "binary_simple" / "baseline" / "best_run" / "result.json"
        self.assertTrue(winner.is_file())
        self.assertFalse((self.out / "plots").exists())
        again, _, again_err = self.run_main(
            campaign.run_hybrid, "--workers", "1", "--indices", "0",
            "--seeds", "1", "--configurations", "baseline",
            "--pop", "4", "--iterations", "1", "--out", str(self.out),
        )
        self.assertEqual(again, 2)
        self.assertIn("output directory is not empty", again_err)

        cancel_out = self.tmp / "cancel"
        os.environ[MODE_ENV] = "cancel"
        os.environ[OUT_ENV] = str(cancel_out)
        self.trace = self.tmp / "cancel-trace.jsonl"
        os.environ[TRACE_ENV] = str(self.trace)
        started = time.time()
        with mock.patch.object(campaign, "PLOTS", (lambda *a, **k: (_ for _ in ()).throw(AssertionError("plot")),)):
            code, out, err = self.run_main(
                campaign.run_hybrid, "--workers", "2", "--indices", "0", "15",
                "--seeds", "1", "--configurations", "baseline",
                "--pop", "4", "--iterations", "1", "--out", str(cancel_out),
            )
        elapsed = time.time() - started
        self.assertEqual(code, 1, err)
        self.assertLess(elapsed, 20, self._events())
        self.assertIn("RuntimeError: boom", err)
        self.assertFalse(any(item["event"] == "end" and item["index"] == 15 for item in self._events()))
        self.assertEqual(sum(item["index"] == 0 and item["event"] == "fail" for item in self._events()), 1)

    def _public_tree(self, root: Path):
        def rows(path, drop):
            with path.open(encoding="utf-8", newline="") as fh:
                loaded = list(csv.DictReader(fh))
            for row in loaded:
                for key in drop:
                    row.pop(key, None)
            return loaded

        runs = rows(root / "runs.csv", {"elapsed_s"})
        stats = rows(root / "statistics.csv", {"mean_elapsed_s"})
        compared = rows(root / "comparisons.csv", set())
        jsonl = []
        for line in (root / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            item.pop("elapsed_s", None)
            jsonl.append(item)
        winners = {}
        for path in sorted((root / "mknapcb1").glob("inst_*/*/*/best_run/result.json")):
            winners[str(path.relative_to(root))] = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        public_manifest = {
            "status": manifest["status"],
            "total_runs": manifest["total_runs"],
            "completed_runs": manifest["completed_runs"],
            "error": manifest["error"],
            "indices": manifest["selection"]["indices"],
            "seeds": manifest["selection"]["seeds"],
            "configurations": manifest["selection"]["configurations"],
            "effective_config": manifest["effective_config"],
        }
        self.assertEqual(public_manifest["completed_runs"], len(runs))
        self.assertEqual(len(jsonl), len(runs))
        text = (root / "resumen.txt").read_text(encoding="utf-8")
        text = "\n".join(line.split(" mean_elapsed_s=")[0] for line in text.splitlines())
        plots = sorted(path.name for path in (root / "plots").glob("*.png"))
        return {
            "runs": runs,
            "jsonl": jsonl,
            "stats": stats,
            "compared": compared,
            "winners": winners,
            "manifest": public_manifest,
            "resumen": text,
            "plots": plots,
            "markers": sorted(str(path.relative_to(root)) for path in root.rglob("marker.txt")),
        }


if __name__ == "__main__":
    unittest.main()
