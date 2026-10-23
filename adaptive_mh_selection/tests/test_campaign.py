"""Grouped MKP campaign: plan by loaded size, keep every outcome, store only the winner."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEMP = Path(__file__).resolve().parent / ".tmp"
INSTANCES = ROOT / "instancias"


def temporary():
    TEMP.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEMP)


def write_instances(directory, family, specs):
    lines = [str(len(specs))]
    for spec in specs:
        n, m = spec["n"], spec["m"]
        profit = spec.get("p", [1] * n)
        weights = spec.get("r", [[1] * n for _ in range(m)])
        capacity = spec.get("b", [n] * m)
        lines.append(f"{n} {m} {spec.get('valor_optimo', 0)}")
        lines.append(" ".join(map(str, profit)))
        for row in weights:
            lines.append(" ".join(map(str, row)))
        lines.append(" ".join(map(str, capacity)))
    path = Path(directory) / f"mknapcb{family}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def invoke(*args, timeout=120):
    return subprocess.run(
        [sys.executable, "-B", "-m", "adaptive_mh_selection.campaign", *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout,
    )


def solution_for(instance, fitness):
    fitness = int(fitness)
    if not 0 <= fitness <= instance.n:
        raise AssertionError("fixture fitness must be a count of selected unit-profit items")
    return [1] * fitness + [0] * (instance.n - fitness)


def fake_result(instance, config, fitness, *, actual=None):
    from hybrid_mkp.orchestrator import PipelineResult, SwitchLog

    actual = config.max_iters if actual is None else actual
    fitness = float(fitness)
    solution = solution_for(instance, fitness)
    history = [fitness] * actual
    native = PipelineResult(
        fitness, solution, history, list(history), [0.0] * actual,
        log_switches=[SwitchLog("PSO", "poblacional", fitness, 0.0, 0.01, actual, [], [])],
        valor_optimo=float(instance.valor_optimo),
    )
    gap = None if instance.valor_optimo == 0 else 100.0 * (instance.valor_optimo - fitness) / instance.valor_optimo
    summary = {
        "status": "success", "seed": config.seed, "selector": config.selector,
        "use_ddtw": config.use_ddtw, "best_fitness": fitness, "best_solution": solution,
        "requested_iterations": config.max_iters, "actual_iterations": actual,
        "iteration_overshoot": max(0, actual - config.max_iters), "gap_pct": gap,
        "n_blocks": 1, "transitions": 0, "elapsed_seconds": 0.01,
    }
    return {"native_result": native, "selection_records": [{"block_index": 0, "solver": "PSO", "seed": config.seed}],
            "summary": summary}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = temporary()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_public_dry_run_plans_27_cases_without_solvers_or_output(self):
        from adaptive_mh_selection.campaign import main

        output = self.root / "planned"
        before = set(Path("resultados/adaptive_mh_selection").glob("*"))
        with patch("adaptive_mh_selection.campaign.run_pipeline") as pipeline, \
                patch("hybrid_mkp.orchestrator.ejecutar_pipeline") as native:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["--dry-run", "--output", str(output)])
        self.assertEqual(code, 0)
        pipeline.assert_not_called()
        native.assert_not_called()
        self.assertFalse(output.exists())
        protocol = json.loads(stdout.getvalue())
        self.assertEqual(protocol["schema_version"], 3)
        self.assertEqual(protocol["budget_unit"], "native_iterations")
        self.assertEqual(protocol["kind"], "grouped_mkp_campaign")
        self.assertEqual(protocol["planned_cases"], 27)
        self.assertEqual(protocol["planned_runs"], 837)
        self.assertEqual(protocol["groups"], {"100": 9, "250": 9, "500": 9})
        self.assertNotIn("240", protocol["groups"])
        self.assertEqual(protocol["families"], list(range(1, 10)))
        self.assertEqual(protocol["indices"], [0, 1, 2])
        self.assertEqual(protocol["seeds"], list(range(42, 73)))
        from adaptive_mh_selection.campaign import make_config
        from adaptive_mh_selection.config import RunConfig, default_monitor
        study, source = RunConfig(), default_monitor()
        self.assertEqual(protocol["conditions"], ["adaptive_ddtw"])
        self.assertEqual(protocol["modes"], ["ddtw"])
        self.assertEqual(protocol["selectors"], [study.selector])
        self.assertEqual(protocol["iterations"], study.max_iters)
        self.assertIsNone(protocol["max_epoch_iters"])
        self.assertEqual((protocol["alpha"], protocol["epsilon"]), (study.alpha, study.epsilon))
        self.assertEqual(protocol["pop_injection_mode"], study.pop_injection_mode)
        for name in ("window", "band", "min_slope", "plateau_max", "patience",
                     "adapt_thresholds", "p_low", "p_high", "improvement_tol", "use_ddtw"):
            self.assertEqual(protocol["monitor"][name], getattr(source, name), name)
        built = make_config(protocol, protocol["seeds"][0], "adaptive", "ddtw")
        self.assertTrue(built.use_ddtw and built.monitor.use_ddtw)
        self.assertEqual(built.monitor.band, source.band)
        self.assertEqual((built.monitor.p_low, built.monitor.p_high), (source.p_low, source.p_high))
        self.assertEqual(built.monitor.adapt_thresholds, source.adapt_thresholds)
        self.assertEqual(built.monitor.improvement_tol, source.improvement_tol)
        self.assertEqual(built.pop_injection_mode, study.pop_injection_mode)
        self.assertIsNone(built.max_epoch_iters)
        self.assertEqual(len(protocol["cases"]), 27)
        self.assertEqual([case["n"] for case in protocol["cases"]], [100] * 9 + [250] * 9 + [500] * 9)
        self.assertEqual(
            [(case["family"], case["index"], case["m"]) for case in protocol["cases"] if case["n"] == 100],
            [(family, index, {1: 5, 4: 10, 7: 30}[family]) for family in (1, 4, 7) for index in (0, 1, 2)],
        )
        for case in protocol["cases"]:
            self.assertEqual(case["file"], f"mknapcb{case['family']}.txt")
            self.assertEqual(case["relative_output_path"], f"items_{case['n']}/mknapcb{case['family']}/inst_{case['index']:02d}")
            self.assertEqual(case["reference_value"], 0)
            self.assertIn("zero means unknown", case["reference_source"])
            self.assertEqual(case["conditions"][0]["relative_output_path"],
                             f"{case['relative_output_path']}/adaptive_ddtw")
        self.assertNotIn("max_evaluations", stdout.getvalue())
        self.assertEqual(before, set(Path("resultados/adaptive_mh_selection").glob("*")))

        cli = invoke("--dry-run", "--output", output / "cli")
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertFalse((output / "cli").exists())
        cli_protocol = json.loads(cli.stdout)
        self.assertEqual(cli_protocol["planned_cases"], 27)
        self.assertEqual(cli_protocol["planned_runs"], 837)
        self.assertEqual(cli_protocol["seeds"][0], 42)
        self.assertEqual(cli_protocol["seeds"][-1], 72)
        self.assertEqual(cli_protocol["conditions"], ["adaptive_ddtw"])
        self.assertEqual(cli_protocol["monitor"]["band"], source.band)
        self.assertEqual(cli_protocol["monitor"]["p_low"], source.p_low)
        self.assertEqual(cli_protocol["monitor"]["p_high"], source.p_high)
        self.assertEqual(cli_protocol["monitor"]["adapt_thresholds"], source.adapt_thresholds)

    def test_invalid_input_writes_nothing(self):
        from adaptive_mh_selection.campaign import main

        real = self.root / "real"
        real.mkdir()
        write_instances(real, 1, [{"n": 240, "m": 5}] + [{"n": 100, "m": 5} for _ in range(29)])
        missing = self.root / "missing"
        missing.mkdir()
        write_instances(missing, 1, [{"n": 100, "m": 4, "valor_optimo": 10}])
        bad = self.root / "bad"
        bad.mkdir()
        (bad / "mknapcb1.txt").write_text("this is not an instance\n", encoding="utf-8")
        short = self.root / "short"
        short.mkdir()
        write_instances(short, 1, [{"n": 100, "m": 2}])
        existing = self.root / "existing"
        existing.mkdir()
        cases = [
            ["--families", "1", "1"],
            ["--families", "10"],
            ["--families", "0"],
            ["--indices", "0", "0"],
            ["--indices", "30"],
            ["--indices", "-1"],
            ["--indices", "2", "--instances-dir", str(short)],
            ["--runs", "0"],
            ["--iterations", "0"],
            ["--max-epoch-iters", "0"],
            ["--seed", "-1"],
            ["--seed", "4294967295", "--runs", "2"],
            ["--selectors", "adaptive", "adaptive"],
            ["--modes", "dtw", "dtw"],
            ["--max-evaluations", "10"],
            ["--instances-dir", str(missing.parent / "absent")],
            ["--instances-dir", str(bad)],
            ["--instances-dir", str(real), "--families", "1", "--indices", "0"],
        ]
        with patch("adaptive_mh_selection.campaign.run_pipeline", side_effect=AssertionError("solver must not start")):
            for args in cases:
                output = self.root / f"out-{cases.index(args)}"
                with self.subTest(args=args):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                        main([*args, "--output", str(output)])
                    self.assertEqual(caught.exception.code, 2, stderr.getvalue())
                    self.assertFalse(output.exists(), stderr.getvalue())
                    if str(real) in args:
                        self.assertIn("240", stderr.getvalue())
                        self.assertIn("250", stderr.getvalue())
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                main(["--output", str(existing), "--families", "1", "--indices", "0", "--dry-run"])
            self.assertEqual(caught.exception.code, 2, stderr.getvalue())
        self.assertEqual(list(existing.iterdir()), [])

    def test_groups_follow_loaded_n_and_do_not_copy_index_zero_reference(self):
        from adaptive_mh_selection.campaign import main

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 500, "m": 7, "valor_optimo": 24381},
                                     {"n": 500, "m": 7, "valor_optimo": 0},
                                     {"n": 500, "m": 7, "valor_optimo": 0}])
        write_instances(library, 9, [{"n": 100, "m": 3, "valor_optimo": 0},
                                     {"n": 100, "m": 3, "valor_optimo": 0}])
        output = self.root / "grouped"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["--dry-run", "--output", str(output), "--instances-dir", str(library),
                         "--families", "9", "1", "--indices", "0", "1", "--runs", "1"])
        self.assertEqual(code, 0)
        self.assertFalse(output.exists())
        protocol = json.loads(stdout.getvalue())
        self.assertEqual([(case["family"], case["index"], case["n"], case["m"], case["reference_value"])
                          for case in protocol["cases"]],
                         [(9, 0, 100, 3, 0), (9, 1, 100, 3, 0), (1, 0, 500, 7, 24381), (1, 1, 500, 7, 0)])
        self.assertEqual(protocol["groups"], {"100": 2, "250": 0, "500": 2})
        self.assertTrue(all(case["relative_output_path"].startswith(f"items_{case['n']}/") for case in protocol["cases"]))
        self.assertNotIn("items_240", stdout.getvalue())

    def test_condition_counts_and_canonical_order(self):
        from adaptive_mh_selection.campaign import main

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["--dry-run", "--families", "1", "--indices", "0", "--runs", "31",
                         "--selectors", "uniform", "adaptive", "--modes", "ddtw", "dtw",
                         "--output", str(self.root / "multi")])
        self.assertEqual(code, 0)
        protocol = json.loads(stdout.getvalue())
        self.assertEqual(protocol["planned_cases"], 1)
        self.assertEqual(protocol["conditions"], ["adaptive_dtw", "uniform_dtw", "adaptive_ddtw", "uniform_ddtw"])
        self.assertEqual(protocol["planned_runs"], 124)
        self.assertFalse((self.root / "multi").exists())

    def test_execution_retains_31_rows_original_winner_and_analysis_metrics(self):
        from adaptive_mh_selection.campaign import main
        from adaptive_mh_selection.exports import write_run

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 100, "m": 4, "valor_optimo": 0}])
        output = self.root / "run31"
        calls = []

        def fake(instance, config):
            if calls:
                rows = read_csv(output / "items_100/mknapcb1/inst_00/adaptive_dtw/runs_resultados.csv")
                self.assertEqual(len(rows), len(calls))
                estado = read_csv(output / "estado_ejecucion.csv")
                self.assertEqual(estado[0]["status"], "running")
                self.assertGreaterEqual(int(estado[0]["succeeded_runs"]) + int(estado[0]["failed_runs"]), 1)
            calls.append(config.seed)
            ordinal = config.seed - 41
            if ordinal == 1:
                raise RuntimeError("first solver failed")
            fitness = 50 if ordinal == 17 else 10
            return fake_result(instance, config, fitness)

        with patch("adaptive_mh_selection.campaign.run_pipeline", side_effect=fake), \
                patch("adaptive_mh_selection.campaign.write_run", wraps=write_run) as writer, \
                patch("adaptive_mh_selection.plotting.plot_run") as plot_run, \
                patch("adaptive_mh_selection.plotting.plot_distribution") as plot_distribution:
            code = main(["--output", str(output), "--instances-dir", str(library),
                         "--families", "1", "--indices", "0", "--runs", "31", "--iterations", "2",
                         "--modes", "dtw", "--selection-trace"])
        self.assertEqual(code, 1)
        self.assertEqual(calls, list(range(42, 73)))
        self.assertEqual(writer.call_count, 1)
        self.assertTrue(writer.call_args.kwargs["best"])
        condition = output / "items_100/mknapcb1/inst_00/adaptive_dtw"
        rows = read_csv(condition / "runs_resultados.csv")
        self.assertEqual(len(rows), 31)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["run"], "1")
        self.assertEqual(rows[0]["seed"], "42")
        self.assertEqual(rows[0]["final_fitness"], "")
        self.assertEqual(rows[0]["gap_pct"], "")
        self.assertEqual(rows[0]["actual_iterations"], "")
        self.assertNotEqual(rows[0]["final_fitness"], "0")
        self.assertEqual(rows[16]["run"], "17")
        self.assertEqual(float(rows[16]["final_fitness"]), 50)
        self.assertEqual([path.name for path in condition.iterdir() if path.name.startswith("best_run_")],
                         ["best_run_17"])
        self.assertFalse(any(path.name.startswith("run_") for path in condition.iterdir()))
        self.assertEqual([path.relative_to(output).as_posix() for path in output.rglob("selection_trace.csv")],
                         ["items_100/mknapcb1/inst_00/adaptive_dtw/best_run_17/selection_trace.csv"])
        self.assertTrue(all("best_run_17" in path.as_posix() for path in output.rglob("native_result.json")))
        self.assertTrue(all("best_run_17" in path.as_posix() for path in output.rglob("historial_dtw.csv")))
        self.assertEqual(len(list(output.rglob("summary.json"))), 1)
        plot_run.assert_called_once()
        plotted = [float(row["final_fitness"]) for row in rows if row["status"] == "success"]
        self.assertEqual(plot_distribution.call_args.args[1], plotted)
        self.assertEqual(len(plotted), 30)
        estado = read_csv(output / "estado_ejecucion.csv")
        self.assertEqual(estado[0]["status"], "partial_failure")
        self.assertEqual(estado[0]["winner_run"], "17")
        self.assertEqual(int(estado[0]["planned_runs"]), 31)
        self.assertEqual(int(estado[0]["succeeded_runs"]), 30)
        self.assertEqual(int(estado[0]["failed_runs"]), 1)
        summary = read_csv(output / "resumen_global.csv")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["status"], "partial_failure")
        self.assertEqual(read_csv(output / "items_100/resumen_grupo.csv"), summary)
        values = [float(row["final_fitness"]) for row in rows if row["status"] == "success"]
        self.assertAlmostEqual(float(summary[0]["best_fitness"]), max(values))
        self.assertAlmostEqual(float(summary[0]["mean_fitness"]), float(np.mean(values)))
        self.assertAlmostEqual(float(summary[0]["std_fitness"]), float(np.std(values, ddof=1)))
        self.assertAlmostEqual(float(summary[0]["median_fitness"]), float(np.median(values)))
        text = (output / "resumen_global.txt").read_text(encoding="utf-8")
        self.assertIn("no pooled", text.lower())
        self.assertIn("partial_failure", text)
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["schema_version"], 3)
        self.assertEqual(protocol["kind"], "grouped_mkp_campaign")
        environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
        source = ROOT / "adaptive_mh_selection/campaign.py"
        digest = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
        self.assertEqual(environment["source_sha256"]["adaptive_mh_selection/campaign.py"], digest)
        from adaptive_mh_selection.analysis import analyze_campaign
        before = sorted(path.relative_to(output).as_posix() for path in output.rglob("*"))
        report = analyze_campaign(output)
        after = sorted(path.relative_to(output).as_posix() for path in output.rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(len(report["metrics"]), 31)
        self.assertEqual(report["successful_runs"], 30)
        self.assertEqual(report["failed_runs"], 1)
        self.assertEqual(report["distinct_cases"], 1)
        self.assertIsInstance(report["metrics"][0]["seed"], int)
        self.assertIsNone(report["metrics"][0]["fitness"])
        self.assertEqual(report["metrics"][16]["fitness"], 50)
        self.assertIsInstance(report["metrics"][16]["iterations"], int)
        self.assertNotEqual(len(report["metrics"]), 1)

    def test_tie_keeps_first_ordinal_and_failed_ordinal_is_not_reused(self):
        from adaptive_mh_selection.campaign import main

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 100, "m": 2}])

        def fake(instance, config):
            if config.seed == 42:
                raise RuntimeError("missing")
            return fake_result(instance, config, 10)

        output = self.root / "ties"
        with patch("adaptive_mh_selection.campaign.run_pipeline", side_effect=fake):
            code = main(["--output", str(output), "--instances-dir", str(library),
                         "--families", "1", "--indices", "0", "--runs", "4", "--iterations", "2",
                         "--modes", "dtw", "--no-plots", "--selection-trace"])
        self.assertEqual(code, 1)
        condition = output / "items_100/mknapcb1/inst_00/adaptive_dtw"
        self.assertTrue((condition / "best_run_02").is_dir())
        self.assertFalse((condition / "best_run_01").exists())
        self.assertFalse((condition / "best_run_04").exists())
        self.assertEqual(list(output.rglob("selection_trace.csv")),
                         [condition / "best_run_02" / "selection_trace.csv"])
        self.assertEqual(read_csv(output / "estado_ejecucion.csv")[0]["winner_run"], "2")

    def test_unknown_reference_gap_stays_blank_and_real_zero_gap_is_kept(self):
        from adaptive_mh_selection.campaign import main

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 100, "m": 1, "valor_optimo": 4},
                                     {"n": 100, "m": 1, "valor_optimo": 0}])

        def fake(instance, config):
            return fake_result(instance, config, 0 if instance.valor_optimo == 0 else 4)

        output = self.root / "gaps"
        with patch("adaptive_mh_selection.campaign.run_pipeline", side_effect=fake):
            code = main(["--output", str(output), "--instances-dir", str(library),
                         "--families", "1", "--indices", "0", "1", "--runs", "1", "--iterations", "2",
                         "--modes", "dtw", "--no-plots"])
        self.assertEqual(code, 0)
        known = read_csv(output / "items_100/mknapcb1/inst_00/adaptive_dtw/runs_resultados.csv")[0]
        unknown = read_csv(output / "items_100/mknapcb1/inst_01/adaptive_dtw/runs_resultados.csv")[0]
        self.assertEqual(float(known["gap_pct"]), 0.0)
        self.assertEqual(float(known["final_fitness"]), 4.0)
        self.assertEqual(unknown["gap_pct"], "")
        self.assertEqual(float(unknown["final_fitness"]), 0.0)
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual([case["reference_value"] for case in protocol["cases"]], [4, 0])

    def test_overshoot_is_recorded_when_the_native_epoch_continues(self):
        from adaptive_mh_selection.campaign import main

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 100, "m": 1}])

        def fake(instance, config):
            self.assertIsNone(config.max_epoch_iters)
            return fake_result(instance, config, 4, actual=5)

        output = self.root / "overshoot"
        with patch("adaptive_mh_selection.campaign.run_pipeline", side_effect=fake):
            code = main(["--output", str(output), "--instances-dir", str(library),
                         "--families", "1", "--indices", "0", "--runs", "1", "--iterations", "2",
                         "--modes", "dtw", "--no-plots"])
        self.assertEqual(code, 0)
        row = read_csv(output / "items_100/mknapcb1/inst_00/adaptive_dtw/runs_resultados.csv")[0]
        self.assertEqual(row["requested_iterations"], "2")
        self.assertEqual(row["actual_iterations"], "5")
        self.assertEqual(row["iteration_overshoot"], "3")
        self.assertEqual(row["status"], "success")

    def test_invalid_native_outcome_is_failed_and_cannot_become_best(self):
        from adaptive_mh_selection.campaign import main

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 100, "m": 1, "b": [0]}])

        def fake(instance, config):
            if config.seed == 42:
                result = fake_result(instance, config, 0)
                result["summary"]["best_fitness"] = float("inf")
                return result
            if config.seed == 43:
                result = fake_result(instance, config, 0)
                result["summary"]["best_solution"] = [1] + [0] * (instance.n - 1)
                result["native_result"].mejor_solucion_global = result["summary"]["best_solution"]
                return result
            return fake_result(instance, config, 0)

        output = self.root / "invalid"
        with patch("adaptive_mh_selection.campaign.run_pipeline", side_effect=fake):
            code = main(["--output", str(output), "--instances-dir", str(library),
                         "--families", "1", "--indices", "0", "--runs", "3", "--iterations", "2",
                         "--modes", "dtw", "--no-plots"])
        self.assertEqual(code, 1)
        rows = read_csv(output / "items_100/mknapcb1/inst_00/adaptive_dtw/runs_resultados.csv")
        self.assertEqual([row["status"] for row in rows], ["failed", "failed", "success"])
        self.assertEqual(rows[0]["final_fitness"], "")
        self.assertEqual(rows[1]["error"].split(":")[0], "ValueError")
        self.assertTrue((output / "items_100/mknapcb1/inst_00/adaptive_dtw/best_run_03").is_dir())
        self.assertEqual(read_csv(output / "estado_ejecucion.csv")[0]["status"], "partial_failure")

    def test_all_failed_has_no_winner_and_later_case_still_runs(self):
        from adaptive_mh_selection.campaign import main

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 100, "m": 2}])
        write_instances(library, 2, [{"n": 250, "m": 6}])

        def fake(instance, config):
            if instance.n == 100:
                raise RuntimeError("unavailable")
            return fake_result(instance, config, 3)

        output = self.root / "failed"
        with patch("adaptive_mh_selection.campaign.run_pipeline", side_effect=fake) as pipeline, \
                patch("adaptive_mh_selection.plotting.plot_run") as plot_run, \
                patch("adaptive_mh_selection.plotting.plot_distribution") as plot_distribution:
            code = main(["--output", str(output), "--instances-dir", str(library),
                         "--families", "1", "2", "--indices", "0", "--runs", "2", "--iterations", "2",
                         "--modes", "dtw"])
        self.assertEqual(code, 1)
        self.assertEqual(pipeline.call_count, 4)
        failed = output / "items_100/mknapcb1/inst_00/adaptive_dtw"
        later = output / "items_250/mknapcb2/inst_00/adaptive_dtw"
        self.assertEqual(list(failed.glob("best_run_*")), [])
        self.assertEqual(list(failed.glob("run_*")), [])
        self.assertEqual(list(failed.rglob("*.png")), [])
        failed_report = (failed / "resumen_pipeline.txt").read_text(encoding="utf-8")
        self.assertIn("Status: failed", failed_report)
        self.assertIn("Successful runs: 0", failed_report)
        self.assertIn("Failed runs: 2", failed_report)
        self.assertIn("Winner original run: --", failed_report)
        self.assertTrue((later / "best_run_01").is_dir())
        estado = {row["n"]: row for row in read_csv(output / "estado_ejecucion.csv")}
        self.assertEqual(estado["100"]["status"], "failed")
        self.assertEqual(estado["100"]["winner_run"], "")
        self.assertEqual(estado["250"]["status"], "complete")
        rows = read_csv(failed / "runs_resultados.csv")
        self.assertEqual([row["final_fitness"] for row in rows], ["", ""])
        summary = {row["n"]: row for row in read_csv(output / "resumen_global.csv")}
        self.assertEqual(summary["100"]["best_fitness"], "")
        self.assertEqual(summary["100"]["mean_fitness"], "")
        self.assertEqual(summary["100"]["std_fitness"], "")
        self.assertEqual(len(read_csv(output / "items_100/resumen_grupo.csv")), 1)
        self.assertEqual(len(read_csv(output / "items_250/resumen_grupo.csv")), 1)
        plot_run.assert_called_once()
        plot_distribution.assert_called_once()
        text = (output / "resumen_global.txt").read_text(encoding="utf-8")
        self.assertIn("no pooled", text.lower())
        self.assertIn("failed", text)

    def test_export_failure_is_separate_and_does_not_retry(self):
        from adaptive_mh_selection.campaign import main
        from adaptive_mh_selection.exports import write_run

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 100, "m": 2}])
        write_instances(library, 2, [{"n": 250, "m": 2}])

        def wrapped(result, folder, **kwargs):
            if "mknapcb1" in Path(folder).as_posix():
                raise OSError("export failed")
            return write_run(result, folder, **kwargs)

        output = self.root / "export"
        with patch("adaptive_mh_selection.campaign.run_pipeline",
                   side_effect=lambda instance, config: fake_result(instance, config, 4)), \
                patch("adaptive_mh_selection.campaign.write_run", side_effect=wrapped) as writer:
            code = main(["--output", str(output), "--instances-dir", str(library),
                         "--families", "1", "2", "--indices", "0", "--runs", "1", "--iterations", "2",
                         "--modes", "dtw", "--no-plots", "--selection-trace"])
        self.assertEqual(code, 1)
        self.assertEqual(writer.call_count, 2)
        estado = {row["family"]: row for row in read_csv(output / "estado_ejecucion.csv")}
        self.assertEqual(estado["1"]["status"], "export_failed")
        self.assertIn("Status: export_failed", (output / "items_100/mknapcb1/inst_00/adaptive_dtw/resumen_pipeline.txt").read_text(encoding="utf-8"))
        self.assertEqual(estado["2"]["status"], "complete")
        self.assertEqual(list(output.rglob("selection_trace.csv")),
                         [output / "items_250/mknapcb2/inst_00/adaptive_dtw/best_run_01/selection_trace.csv"])
        failed_rows = read_csv(output / "items_100/mknapcb1/inst_00/adaptive_dtw/runs_resultados.csv")
        self.assertEqual(failed_rows[0]["status"], "success")
        self.assertEqual(float(failed_rows[0]["final_fitness"]), 4)

    def test_interrupt_keeps_partial_metrics_and_does_not_mark_completed(self):
        from adaptive_mh_selection.campaign import main

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 100, "m": 2}])
        write_instances(library, 2, [{"n": 250, "m": 2}])
        seen = []

        def fake(instance, config):
            seen.append((instance.n, config.seed))
            if instance.n == 100 and config.seed == 43:
                raise KeyboardInterrupt()
            return fake_result(instance, config, 4)

        output = self.root / "interrupt"
        with patch("adaptive_mh_selection.campaign.run_pipeline", side_effect=fake):
            code = main(["--output", str(output), "--instances-dir", str(library),
                         "--families", "1", "2", "--indices", "0", "--runs", "3", "--iterations", "2",
                         "--modes", "dtw", "--no-plots"])
        self.assertEqual(code, 1)
        self.assertEqual(seen, [(100, 42), (100, 43)])
        estado = read_csv(output / "estado_ejecucion.csv")
        self.assertEqual([row["status"] for row in estado], ["interrupted", "pending"])
        self.assertEqual(estado[0]["succeeded_runs"], "1")
        self.assertEqual(estado[0]["failed_runs"], "0")
        self.assertEqual(estado[0]["planned_runs"], "3")
        self.assertEqual(estado[0]["winner_run"], "")
        self.assertNotEqual(estado[0]["status"], "complete")
        rows = read_csv(output / "items_100/mknapcb1/inst_00/adaptive_dtw/runs_resultados.csv")
        self.assertEqual([row["status"] for row in rows], ["success", "interrupted"])
        self.assertEqual(rows[1]["final_fitness"], "")
        report = (output / "items_100/mknapcb1/inst_00/adaptive_dtw/resumen_pipeline.txt").read_text(encoding="utf-8")
        self.assertIn("Status: interrupted", report)
        self.assertIn("Planned runs: 3", report)
        self.assertIn("Successful runs: 1", report)
        self.assertEqual(list(output.rglob("best_run_*")), [])
        self.assertIn("interrupted", (output / "resumen_global.txt").read_text(encoding="utf-8"))

    def test_compared_conditions_share_seeds_and_are_not_pooled(self):
        from adaptive_mh_selection.campaign import main

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 250, "m": 8}])
        seen = []

        def fake(instance, config):
            seen.append((config.selector, config.use_ddtw, config.seed, config.alpha, instance.m))
            fitness = 10 if config.selector == "adaptive" else 40
            return fake_result(instance, config, fitness)

        output = self.root / "compared"
        with patch("adaptive_mh_selection.campaign.run_pipeline", side_effect=fake):
            code = main(["--output", str(output), "--instances-dir", str(library),
                         "--families", "1", "--indices", "0", "--runs", "2", "--iterations", "2",
                         "--selectors", "uniform", "adaptive", "--modes", "dtw", "--alpha", "0.4",
                         "--no-plots"])
        self.assertEqual(code, 0)
        self.assertEqual(seen, [("adaptive", False, 42, 0.4, 8), ("adaptive", False, 43, 0.4, 8),
                                ("uniform", False, 42, 0.4, 8), ("uniform", False, 43, 0.4, 8)])
        self.assertTrue((output / "items_250/mknapcb1/inst_00/adaptive_dtw/best_run_01").is_dir())
        self.assertTrue((output / "items_250/mknapcb1/inst_00/uniform_dtw/best_run_01").is_dir())
        summary = read_csv(output / "resumen_global.csv")
        self.assertEqual([row["condition"] for row in summary], ["adaptive_dtw", "uniform_dtw"])
        self.assertEqual([float(row["mean_fitness"]) for row in summary], [10, 40])
        text = (output / "resumen_global.txt").read_text(encoding="utf-8")
        self.assertIn("no pooled", text.lower())
        self.assertIn("adaptive_dtw", text)
        self.assertIn("uniform_dtw", text)

    def test_public_cli_short_native_case_records_overshoot_without_loser_directories(self):
        output = self.root / "native"
        result = invoke("--output", output, "--families", "1", "--indices", "0", "--runs", "2",
                        "--iterations", "2", "--max-epoch-iters", "1", "--modes", "dtw", "--no-plots")
        self.assertEqual(result.returncode, 0, result.stderr)
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["schema_version"], 3)
        self.assertEqual(protocol["cases"][0]["n"], 100)
        self.assertEqual(protocol["cases"][0]["m"], 5)
        self.assertEqual(protocol["cases"][0]["reference_value"], 0)
        condition = output / "items_100/mknapcb1/inst_00/adaptive_dtw"
        rows = read_csv(condition / "runs_resultados.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["seed"] for row in rows], ["42", "43"])
        self.assertTrue(all(row["status"] == "success" for row in rows))
        self.assertTrue(all(row["gap_pct"] == "" for row in rows))
        self.assertTrue(all(int(row["requested_iterations"]) == 2 for row in rows))
        self.assertTrue(all(int(row["actual_iterations"]) >= 2 for row in rows))
        self.assertTrue(all(int(row["iteration_overshoot"]) == int(row["actual_iterations"]) - 2 for row in rows))
        self.assertFalse(list(condition.glob("run_*")))
        winners = list(condition.glob("best_run_*"))
        self.assertEqual(len(winners), 1)
        winner_ordinal = int(winners[0].name.rsplit("_", 1)[1])
        best = max(rows, key=lambda row: float(row["final_fitness"]))
        self.assertEqual(winner_ordinal, int(best["run"]))
        self.assertEqual(read_csv(output / "estado_ejecucion.csv")[0]["status"], "complete")
        report_path = condition / "resumen_pipeline.txt"
        self.assertTrue(report_path.is_file())
        report = report_path.read_text(encoding="utf-8")
        self.assertIn("Planned runs: 2", report)
        self.assertIn("Successful runs: 2", report)
        self.assertIn("Failed runs: 0", report)
        self.assertIn(f"Winner original run: {winner_ordinal}", report)
        self.assertIn("Status: complete", report)
        self.assertEqual(list(output.rglob("*.png")), [])
        self.assertTrue((winners[0] / "native_result.json").is_file())
        self.assertTrue((winners[0] / "historial_dtw.csv").is_file())

    def test_cli_overrides_keep_unspecified_monitor_fields(self):
        from adaptive_mh_selection.campaign import main, make_config
        from adaptive_mh_selection.config import default_monitor

        library = self.root / "library"
        library.mkdir()
        write_instances(library, 1, [{"n": 100, "m": 2}])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["--dry-run", "--output", str(self.root / "override"),
                         "--instances-dir", str(library), "--families", "1", "--indices", "0",
                         "--runs", "1", "--window", "12", "--min-slope", "0.25",
                         "--plateau-max", "9", "--patience", "4", "--modes", "dtw",
                         "--alpha", "0.4", "--epsilon", "0.5", "--iterations", "8"])
        self.assertEqual(code, 0)
        self.assertFalse((self.root / "override").exists())
        protocol = json.loads(stdout.getvalue())
        source = default_monitor()
        self.assertEqual(protocol["conditions"], ["adaptive_dtw"])
        self.assertEqual(protocol["monitor"]["window"], 12)
        self.assertEqual(protocol["monitor"]["min_slope"], 0.25)
        self.assertEqual(protocol["monitor"]["plateau_max"], 9)
        self.assertEqual(protocol["monitor"]["patience"], 4)
        self.assertEqual(protocol["monitor"]["band"], source.band)
        self.assertEqual(protocol["monitor"]["p_low"], source.p_low)
        self.assertEqual(protocol["monitor"]["p_high"], source.p_high)
        self.assertEqual(protocol["monitor"]["adapt_thresholds"], source.adapt_thresholds)
        self.assertEqual(protocol["monitor"]["improvement_tol"], source.improvement_tol)
        built = make_config(protocol, 42, "adaptive", "dtw")
        self.assertFalse(built.use_ddtw)
        self.assertFalse(built.monitor.use_ddtw)
        self.assertEqual(built.monitor.band, source.band)
        self.assertEqual((built.monitor.p_low, built.monitor.p_high), (source.p_low, source.p_high))
        self.assertEqual(built.monitor.adapt_thresholds, source.adapt_thresholds)
        self.assertEqual(built.monitor.improvement_tol, source.improvement_tol)
        self.assertEqual(built.alpha, 0.4)
        self.assertEqual(built.epsilon, 0.5)
        self.assertEqual(built.max_iters, 8)
        self.assertEqual(built.pop_injection_mode, "mixed")
        self.assertIsNone(built.max_epoch_iters)

    def test_custom_band_and_percentiles_survive_make_config(self):
        from adaptive_mh_selection.campaign import make_config

        protocol = {
            "iterations": 11, "max_epoch_iters": None, "alpha": 0.3, "epsilon": 0.2,
            "pop_injection_mode": "mixed",
            "monitor": {
                "window": 70, "band": 9, "min_slope": 2, "plateau_max": 25, "patience": 15,
                "use_ddtw": False, "adapt_thresholds": False, "p_low": 11.0, "p_high": 88.0,
                "improvement_tol": 0.01,
            },
        }
        built = make_config(protocol, 7, "uniform", "ddtw")
        self.assertEqual(built.seed, 7)
        self.assertEqual(built.selector, "uniform")
        self.assertEqual(built.monitor.band, 9)
        self.assertEqual(built.monitor.p_low, 11.0)
        self.assertEqual(built.monitor.p_high, 88.0)
        self.assertFalse(built.monitor.adapt_thresholds)
        self.assertEqual(built.monitor.improvement_tol, 0.01)
        self.assertEqual(built.monitor.window, 70)
        self.assertTrue(built.use_ddtw)
        self.assertTrue(built.monitor.use_ddtw)
        self.assertEqual(built.pop_injection_mode, "mixed")
