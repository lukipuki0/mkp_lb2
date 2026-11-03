"""Public CLI and campaign-record contracts for the HRES2 paper rerun wrapper.

Helpers are mocked. These tests must not launch the nominal simulation.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "HRES2-H2" / "reproduce_campaign.py"
ORCHESTRATOR = ROOT / "HRES2-H2" / "orchestrator.py"


def load_module():
    if not ENTRY.is_file():
        raise AssertionError(f"missing public entrypoint: {ENTRY}")
    spec = importlib.util.spec_from_file_location("hres2_reproduce_campaign", ENTRY)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load public entrypoint: {ENTRY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_result(value=0.25, solution=None, history=3, feasible=True):
    solution = [0.1, 2.0, 5.0, 1.0] if solution is None else solution
    return SimpleNamespace(
        mejor_valor_global=value,
        mejor_valor=value,
        mejor_solucion_global=list(solution),
        mejor_solucion=list(solution),
        historial_global=[value] * history,
        historial=[value] * history,
        feasible=feasible,
    )


class PublicCliTests(unittest.TestCase):
    def test_public_cli_help_does_not_launch_experiments(self):
        with tempfile.TemporaryDirectory() as scratch:
            completed = subprocess.run(
                [sys.executable, str(ENTRY), "--help"],
                cwd=scratch,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            help_text = (completed.stdout + completed.stderr).lower()
            self.assertIn("usage", help_text)
            self.assertIn("nonpaper", help_text)
            self.assertEqual(list(Path(scratch).iterdir()), [])


class CampaignContractTests(unittest.TestCase):
    def test_paper_config_roles_seeds_and_unpaired_plan(self):
        module = load_module()
        config = module.paper_config()
        self.assertEqual(config["window"], 40)
        self.assertEqual(config["band"], 4)
        self.assertEqual(config["patience"], 8)
        self.assertEqual(config["min_slope"], 0.1)
        self.assertEqual(config["plateau_max"], 15)
        self.assertIs(config["adapt_thresholds"], True)
        self.assertEqual(config["p_low"], 30.0)
        self.assertEqual(config["p_high"], 70.0)
        self.assertEqual(config["nominal_iterations"], 1000)
        self.assertEqual(config["seeds"], list(range(43, 74)))
        self.assertEqual(config["distance_modes"], ["dtw", "ddtw"])
        self.assertEqual(
            config["baselines"],
            ["PSO", "GWO", "WOA", "EHO", "ACO", "ILS", "SA", "TS", "VNS"],
        )
        self.assertNotIn("ABC", config["baselines"])
        self.assertTrue(config["paper_nominal"])
        self.assertFalse(config["nonpaper"])

        jobs = module.build_jobs(config)
        hybrid = [job for job in jobs if job["role"] == "hybrid"]
        baselines = [job for job in jobs if job["role"] == "baseline"]
        self.assertEqual(len(hybrid), 62)
        self.assertEqual(len(baselines), 9)
        for mode in ("dtw", "ddtw"):
            selected = [job for job in hybrid if job["distance_mode"] == mode]
            self.assertEqual([job["seed"] for job in selected], list(range(43, 74)))
            self.assertEqual([job["run"] for job in selected], list(range(1, 32)))
        self.assertEqual([job["algorithm"] for job in baselines], config["baselines"])
        self.assertEqual(len({job["seed"] for job in baselines}), 9)
        self.assertTrue(set(config["baseline_seeds"].values()).isdisjoint(config["seeds"]))
        self.assertTrue(all(job["paired"] is False for job in jobs))
        self.assertIn("not an evaluation budget", module.ITERATION_SEMANTICS)

    def test_incremental_records_weather_before_seed_and_existing_output_refused(self):
        module = load_module()
        events = []
        calls = {"pipeline": 0}

        def factory():
            events.append("weather")
            return SimpleNamespace(
                df_year=None,
                get_info=lambda solution: {"feasible": True, "lcoe_cny_per_kwh": 0.125},
            )

        def seed_globals(seed):
            events.append(("seed", seed))

        def pipeline(**kwargs):
            events.append("pipeline")
            calls["pipeline"] += 1
            self.assertNotIn("max_epoch_iters", kwargs)
            self.assertEqual(kwargs["max_iters"], 2)
            self.assertEqual(kwargs["stag_cfg"].window, 40)
            self.assertEqual(kwargs["stag_cfg"].band, 4)
            self.assertEqual(kwargs["stag_cfg"].patience, 8)
            self.assertEqual(kwargs["stag_cfg"].min_slope, 0.1)
            self.assertEqual(kwargs["stag_cfg"].plateau_max, 15)
            self.assertIs(kwargs["stag_cfg"].adapt_thresholds, True)
            self.assertEqual(kwargs["stag_cfg"].p_low, 30.0)
            self.assertEqual(kwargs["stag_cfg"].p_high, 70.0)
            self.assertEqual(kwargs["stag_cfg"].use_ddtw, calls["pipeline"] == 2)
            if calls["pipeline"] == 2:
                raise RuntimeError("interrupted after first completed record")
            return fake_result(value=1.0 / 3.0, history=5)

        standalone_calls = []

        def standalone(func, name, max_iters):
            standalone_calls.append((name, max_iters))
            return fake_result()

        config = module.paper_config()
        config.update({
            "paper_nominal": False,
            "nonpaper": True,
            "label": "nonpaper_test",
            "seeds": [43],
            "nominal_iterations": 2,
            "distance_modes": ["dtw", "ddtw"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fresh"
            with self.assertRaises(RuntimeError):
                module.run_campaign(
                    output,
                    config,
                    function_factory=factory,
                    pipeline=pipeline,
                    standalone=standalone,
                    seed_globals=seed_globals,
                )
            self.assertEqual(events[0], "weather")
            self.assertLess(events.index("weather"), events.index(("seed", 43)))
            records_path = output / "runs.jsonl"
            self.assertTrue(records_path.is_file())
            lines = records_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["algorithm"], "HRES2")
            self.assertEqual(record["role"], "hybrid")
            self.assertEqual(record["run"], 1)
            self.assertEqual(record["seed"], 43)
            self.assertEqual(record["objective"]["repr"], repr(1.0 / 3.0))
            self.assertIs(record["feasible"], True)
            self.assertEqual(record["solution"]["repr"], [repr(v) for v in [0.1, 2.0, 5.0, 1.0]])
            self.assertEqual(record["actual_history_length"], 5)
            self.assertFalse(record["paired"])
            self.assertFalse(record["evaluation_budget_equivalent"])
            self.assertFalse((output / "completion.json").exists())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("git_sha", manifest["source_revision"])
            self.assertIn("weather", manifest)
            self.assertIn("runtime_versions", manifest)
            self.assertFalse(manifest["effective_configuration"]["paper_nominal"])
            self.assertEqual(standalone_calls, [])

            sentinel = output / "runs.jsonl"
            before = sentinel.read_bytes()
            def refuse_factory():
                raise AssertionError("existing destination must not start work")

            with self.assertRaises(FileExistsError):
                module.run_campaign(
                    output,
                    config,
                    function_factory=refuse_factory,
                    pipeline=pipeline,
                    standalone=standalone,
                    seed_globals=seed_globals,
                )
            self.assertEqual(sentinel.read_bytes(), before)

    def test_completed_smoke_calls_each_baseline_once_and_writes_completion(self):
        module = load_module()
        calls = []

        def pipeline(**kwargs):
            calls.append(("hybrid", kwargs["stag_cfg"].use_ddtw, kwargs["max_iters"]))
            self.assertNotIn("max_epoch_iters", kwargs)
            return fake_result(history=kwargs["max_iters"] + 2)

        def standalone(func, name, max_iters):
            calls.append(("baseline", name, max_iters))
            return fake_result(history=max_iters)

        config = module.nonpaper_config(runs=1, iterations=1)
        config["baselines"] = ["PSO"]
        config["baseline_seeds"] = {"PSO": 10043}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "done"
            completion = module.run_campaign(
                output,
                config,
                function_factory=lambda: SimpleNamespace(df_year=None),
                pipeline=pipeline,
                standalone=standalone,
                seed_globals=lambda seed: None,
            )
            self.assertTrue(completion["completed"])
            self.assertTrue(completion["nonpaper"])
            self.assertFalse(completion["paper_nominal"])
            self.assertEqual(calls, [
                ("hybrid", False, 1),
                ("hybrid", True, 1),
                ("baseline", "PSO", 1),
            ])
            rows = [
                json.loads(line)
                for line in (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["role"] for row in rows], ["hybrid", "hybrid", "baseline"])
            self.assertEqual([row["algorithm"] for row in rows], ["HRES2", "HRES2", "PSO"])
            self.assertTrue(all(row["actual_history_length"] == 3 for row in rows[:2]))
            self.assertTrue((output / "completion.json").is_file())

    def test_smoke_cli_is_marked_nonpaper_and_help_path_does_not_call_runner(self):
        module = load_module()
        seen = {}

        def runner(output, config, **kwargs):
            seen["config"] = config
            seen["output"] = output

        original = module.run_campaign
        module.run_campaign = runner
        try:
            with self.assertRaises(SystemExit) as exited:
                module.main(["--help"])
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(seen, {})
            with tempfile.TemporaryDirectory() as tmp:
                code = module.main([
                    "--output", str(Path(tmp) / "smoke"),
                    "--nonpaper",
                    "--runs", "1",
                    "--iterations", "1",
                ])
            self.assertEqual(code, 0)
        finally:
            module.run_campaign = original
        self.assertFalse(seen["config"]["paper_nominal"])
        self.assertTrue(seen["config"]["nonpaper"])
        self.assertEqual(seen["config"]["seeds"], [43])
        self.assertEqual(seen["config"]["nominal_iterations"], 1)
        self.assertEqual(seen["config"]["baselines"], module.BASELINES)
        self.assertIn("nonpaper", seen["config"]["label"])

    def test_helper_signatures_remain_the_existing_contracts(self):
        tree = ast.parse(ORCHESTRATOR.read_text(encoding="utf-8"))
        signatures = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "ejecutar_pipeline_hres2",
                "ejecutar_mh_standalone_hres2",
            }:
                signatures[node.name] = [arg.arg for arg in node.args.args]
        self.assertEqual(
            signatures["ejecutar_pipeline_hres2"],
            [
                "func", "max_iters", "tiempo_max", "stag_cfg", "verbose",
                "on_epoch_callback", "pool_poblacional", "pool_trayectoria",
                "max_epoch_iters",
            ],
        )
        self.assertEqual(
            signatures["ejecutar_mh_standalone_hres2"],
            ["func", "mh_nombre", "max_iters"],
        )
        module = load_module()
        self.assertEqual(module.PIPELINE_HELPER, "ejecutar_pipeline_hres2")
        self.assertEqual(module.STANDALONE_HELPER, "ejecutar_mh_standalone_hres2")


if __name__ == "__main__":
    unittest.main()
