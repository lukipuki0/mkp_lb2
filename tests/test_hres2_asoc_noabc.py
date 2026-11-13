"""Campaign wiring contracts; no optimization is launched."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_hres2_benchmark_records import ROOT, load_module, fake_result, physical_info

ENTRY = ROOT / "HRES2-H2" / "run_asoc_ddtw_noabc.py"
POPULATION = ["PSO", "GWO", "WOA", "EHO", "ACO"]
TRAJECTORY = ["ILS", "SA", "TS", "VNS"]


class AsocNoAbcTests(unittest.TestCase):
    def test_campaign_pool_manifest_seeds_and_nominal_protocol(self):
        module = load_module(ENTRY, "asoc_noabc_under_test")
        bench = module.benchmark
        calls, seeds = [], []
        defaults = list(bench._orch_mod.POOL_POBLACIONAL_HRES2)

        def pipeline(**kwargs):
            self.assertEqual(kwargs["pool_poblacional"], POPULATION)
            self.assertNotIn("max_epoch_iters", kwargs)
            self.assertEqual(kwargs["max_iters"], 1000)
            self.assertEqual(vars(kwargs["stag_cfg"]), vars(bench.make_stag(True)))
            calls.append("Hybrid DDTW")
            return fake_result()

        def standalone(_func, name, iterations):
            self.assertEqual(iterations, 1000)
            calls.append(name)
            return fake_result()

        def factory():
            self.assertEqual(seeds, [])  # weather precedes campaign seeding
            return SimpleNamespace(df_year=None, get_info=lambda _s: physical_info())

        with tempfile.TemporaryDirectory() as scratch, contextlib.ExitStack() as stack:
            for name in ("grafico_convergencia_hres2", "grafico_dtw_delta", "grafico_switches",
                         "grafico_boxplot_hres2", "realizar_analisis_estadistico_completo"):
                stack.enter_context(patch.object(bench, name, return_value={}))
            stack.enter_context(patch.object(bench, "ejecutar_pipeline_hres2", pipeline))
            stack.enter_context(patch.object(bench, "ejecutar_mh_standalone_hres2", standalone))
            model = stack.enter_context(patch.object(bench, "HRES2Function", side_effect=factory))
            stack.enter_context(patch.object(bench, "_seed_globals", seeds.append))
            stdout, stderr = io.StringIO(), io.StringIO()
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            output = Path(scratch) / "fresh"
            self.assertEqual(module.main(["--output", str(output)]), 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("Hybrid DDTW", stdout.getvalue())
            records = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
            methods = ["Hybrid DDTW"]
            self.assertEqual(calls, [name for name in methods for _ in range(31)])
            self.assertEqual(seeds, [42 + run + 1000 * i for i in range(1) for run in range(1, 32)])
            self.assertEqual([r["method"] for r in records], calls)
            self.assertEqual([r["seed"] for r in records], seeds)
            manifest = json.loads((output / "manifest.json").read_text())
            config = manifest["effective_configuration"]
            self.assertEqual(config["hybrid_population"], POPULATION)
            self.assertEqual(config["hybrid_trajectory"], TRAJECTORY)
            self.assertEqual(config["modes"], ["ddtw"])
            self.assertEqual(config["baselines"], [])
            self.assertEqual((config["n_runs"], config["nominal_iterations"], config["seed_base"]), (31, 1000, 42))
            self.assertTrue(manifest["weather"]["default_factory"])
            self.assertEqual(json.loads((output / "completion.json").read_text())["records"], 31)
            before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite existing output destination"):
                module.main(["--output", str(output)])
            self.assertEqual(model.call_count, 1)
            self.assertEqual(len(calls), 31)
            self.assertEqual(len(seeds), 31)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()})
        self.assertEqual(bench._orch_mod.POOL_POBLACIONAL_HRES2, defaults)
        self.assertIn("ABC", defaults)

    def test_one_run_no_baselines_exports_real_plots_and_telemetry(self):
        module = load_module(ENTRY, "asoc_noabc_under_test")
        bench = module.benchmark
        result = fake_result()
        result.dtw_deltas_global = [0.1, -0.2, 0.3]
        result.log_switches = [SimpleNamespace(n_iters=3, mh_nombre="PSO", tipo="poblacional")]
        seeds = []
        with tempfile.TemporaryDirectory() as scratch, contextlib.ExitStack() as stack:
            pipeline = stack.enter_context(patch.object(bench, "ejecutar_pipeline_hres2", return_value=result))
            standalone = stack.enter_context(patch.object(bench, "ejecutar_mh_standalone_hres2", side_effect=AssertionError("baseline invoked")))
            stack.enter_context(patch.object(bench, "HRES2Function", return_value=SimpleNamespace(df_year=None, get_info=lambda _s: physical_info())))
            stack.enter_context(patch.object(bench, "grafico_switches"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            output = Path(scratch) / "trace"
            bench.ejecutar_benchmark_hres2(n_runs=1, max_iters=1000, seed_base=42,
                modes=["ddtw"], baselines=[], pool_poblacional=POPULATION,
                seed_globals=seeds.append, output_dir=str(output))
            self.assertEqual(seeds, [43])
            standalone.assert_not_called()
            self.assertEqual(pipeline.call_count, 1)
            self.assertEqual(pipeline.call_args.kwargs["pool_poblacional"], POPULATION)
            self.assertTrue(pipeline.call_args.kwargs["stag_cfg"].use_ddtw)
            self.assertEqual(pipeline.call_args.kwargs["max_iters"], 1000)
            for name in ("fitness_convergence.png", "dtw_delta.png", "historial_dtw.csv", "historial_dtw_detalle.txt"):
                self.assertGreater((output / "best_run_01" / name).stat().st_size, 0)
            self.assertEqual(len((output / "best_run_01/historial_dtw.csv").read_text().splitlines()), 4)
            self.assertEqual(json.loads((output / "completion.json").read_text())["records"], 1)

    def test_help_does_not_launch_campaign(self):
        module = load_module(ENTRY, "asoc_noabc_under_test")
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(module.benchmark, "ejecutar_benchmark_hres2") as runner:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    module.main(["--help"])
            self.assertEqual(caught.exception.code, 0)
            runner.assert_not_called()
        self.assertIn("--output", stdout.getvalue())
        self.assertIn("ABC", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")
