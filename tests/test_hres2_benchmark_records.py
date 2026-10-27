"""Focused contracts for HRES2 canonical records. Optimization is mocked."""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "HRES2-H2" / "benchmark_hres2.py"
ANALYSIS = ROOT / "HRES2-H2" / "analisis_estadistico_hres2.py"
_LOADED = {}


def load_module(path: Path, name: str):
    if name not in _LOADED:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise AssertionError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOADED[name] = module
    return _LOADED[name]


def fake_result(value=0.2, solution=None, history=3):
    solution = [0.1, 12.0, 10.0, 1.0] if solution is None else list(solution)
    return SimpleNamespace(
        mejor_valor=value,
        mejor_valor_global=value,
        mejor_solucion=list(solution),
        mejor_solucion_global=list(solution),
        historial=[value] * history,
        historial_global=[value] * history,
        historial_inst_global=[],
        dtw_deltas_global=[],
        dtw_info_global=[],
        log_switches=[],
        n_switches=0,
        iteraciones=history,
    )


def physical_info(feasible=True, lcoe=0.2, lcoh=1.5, agsr=0.1):
    return {
        "feasible": feasible,
        "lcoe_cny_per_kwh": lcoe,
        "lcoh_cny_per_kg": lcoh,
        "agsr": agsr,
        "wind_mw": 10.0,
        "pv_mw": 190.0,
        "n_el_units": 12,
        "electrolyzer_mw": 60.0,
        "battery_mw": 10.0,
        "battery_duration_h": 2.0,
        "battery_mwh": 20.0,
        "total_h2_kg": 1234.5,
        "electrolyzer_cf": 0.4,
    }


class PublicPersistenceTests(unittest.TestCase):
    def test_run_button_defaults_and_explicit_mode_are_preserved(self):
        module = load_module(BENCHMARK, "hres2_benchmark_records_under_test")
        calls = []
        module.ejecutar_benchmark_hres2 = lambda *args, **kwargs: calls.append((args, kwargs))
        self.assertEqual(module.main([]), 0)
        args, kwargs = calls[-1]
        self.assertEqual(args, (31, 1000, None))
        self.assertEqual(kwargs["modes"], ["dtw", "ddtw"])
        self.assertIsNone(kwargs["baselines"])
        self.assertFalse(kwargs["nonpaper"])
        self.assertEqual(module.main(["--modes", "ddtw"]), 0)
        self.assertEqual(calls[-1][1]["modes"], ["ddtw"])
        self.assertEqual(module.main(["--nonpaper"]), 0)
        self.assertEqual(calls[-1][0], (1, 1, None))
        self.assertEqual(calls[-1][1]["modes"], ["dtw"])
        self.assertEqual(calls[-1][1]["baselines"], ["PSO"])

    def test_public_cli_help_does_not_start_the_runner(self):
        module = load_module(BENCHMARK, "hres2_benchmark_records_under_test")
        with self.assertRaises(SystemExit) as caught:
            module.main(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_help_subprocess_is_safe_and_exposes_nonpaper(self):
        with tempfile.TemporaryDirectory() as scratch:
            completed = subprocess.run(
                [sys.executable, str(BENCHMARK), "--help"],
                cwd=scratch,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            help_text = " ".join((completed.stdout + completed.stderr).lower().split())
            self.assertIn("usage", help_text)
            self.assertIn("--nonpaper", help_text)
            self.assertIn("--modes", help_text)
            self.assertIn("--baselines", help_text)
            self.assertIn("honors --runs/--iterations", help_text)
            self.assertIn("not meaningful", help_text)
            self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_incremental_records_seeds_refusal_and_full_precision(self):
        module = load_module(BENCHMARK, "hres2_benchmark_records_under_test")
        events = []
        seen = {}

        def factory():
            events.append("weather")
            return SimpleNamespace(df_year=None, get_info=lambda _solution: physical_info(lcoe=1.0 / 3.0))

        def seed_globals(seed):
            events.append(("seed", seed))

        def pipeline(**kwargs):
            events.append("pipeline")
            seen["stag"] = kwargs["stag_cfg"]
            self.assertNotIn("max_epoch_iters", kwargs)
            self.assertEqual(kwargs["max_iters"], 2)
            return fake_result(value=1.0 / 3.0, history=5)

        def standalone(*_args, **_kwargs):
            raise AssertionError("baselines were disabled")

        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "fresh"
            result = module.ejecutar_benchmark_hres2(
                1, 2, str(output), modes=["dtw"], baselines=[],
                pipeline=pipeline, standalone=standalone,
                function_factory=factory, seed_globals=seed_globals, seed_base=42,
            )
            self.assertEqual(events[0], "weather")
            self.assertLess(events.index("weather"), events.index(("seed", 43)))
            self.assertEqual(seen["stag"].window, 40)
            self.assertEqual(seen["stag"].band, 4)
            self.assertEqual(seen["stag"].patience, 8)
            self.assertEqual(seen["stag"].min_slope, 0.1)
            self.assertEqual(seen["stag"].plateau_max, 15)
            self.assertIs(seen["stag"].adapt_thresholds, True)
            self.assertEqual(seen["stag"].p_low, 30.0)
            self.assertEqual(seen["stag"].p_high, 70.0)
            self.assertIs(seen["stag"].use_ddtw, False)
            record = json.loads((output / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["run"], 1)
            self.assertEqual(record["method"], "Hybrid DTW")
            self.assertEqual(record["mode"], "dtw")
            self.assertEqual(record["seed"], 43)
            self.assertEqual(record["objective_repr"], repr(1.0 / 3.0))
            self.assertEqual(record["solution_repr"], [repr(v) for v in [0.1, 12.0, 10.0, 1.0]])
            self.assertEqual(record["physical_lcoe_repr"], repr(1.0 / 3.0))
            self.assertIs(record["objective_is_physical_lcoe"], True)
            self.assertEqual(record["decoded"]["total_h2_kg"], 1234.5)
            self.assertEqual(record["nominal_iterations"], 2)
            self.assertEqual(record["actual_iterations"], 5)
            self.assertEqual(record["overshoot_iterations"], 3)
            self.assertIs(record["paired_initial_conditions"], False)
            self.assertIn("does not by itself prove paired", record["seed_note"])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("effective_monitor", manifest)
            self.assertIn("weather", manifest)
            self.assertIn("runtime", manifest)
            self.assertIn("git_sha", manifest["source"])
            self.assertIn("media_lcoe", result)
            self.assertIn("best_config", result)
            self.assertIn("stat_results", result)
            before = (output / "runs.jsonl").read_bytes()

            def refuse_factory():
                raise AssertionError("existing destination must not start work")

            with self.assertRaises(FileExistsError) as caught:
                module.ejecutar_benchmark_hres2(
                    1, 2, str(output), modes=["dtw"], baselines=[],
                    function_factory=refuse_factory, pipeline=pipeline,
                    standalone=standalone, seed_globals=seed_globals,
                )
            self.assertIn("refusing to overwrite", str(caught.exception))
            self.assertEqual((output / "runs.jsonl").read_bytes(), before)

    def test_partial_failure_keeps_flushed_record_without_completion(self):
        module = load_module(BENCHMARK, "hres2_benchmark_records_under_test")
        calls = {"n": 0}

        def pipeline(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("interrupted after first completed record")
            return fake_result()

        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "partial"
            with self.assertRaises(RuntimeError):
                module.ejecutar_benchmark_hres2(
                    1, 1, str(output), modes=["dtw", "ddtw"], baselines=[],
                    pipeline=pipeline, standalone=lambda *_a, **_k: fake_result(),
                    function_factory=lambda: SimpleNamespace(df_year=None, get_info=lambda _s: physical_info()),
                    seed_globals=lambda _seed: None,
                )
            lines = (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["method"], "Hybrid DTW")
            self.assertFalse((output / "completion.json").exists())

    def test_penalty_and_nonfinite_metrics_stay_null_and_distinct(self):
        module = load_module(BENCHMARK, "hres2_benchmark_records_under_test")

        def info(_solution):
            return physical_info(feasible=False, lcoe=math.inf, lcoh=math.inf, agsr=math.nan)

        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "penalty"
            module.ejecutar_benchmark_hres2(
                1, 1, str(output), modes=["ddtw"], baselines=[],
                pipeline=lambda **_kwargs: fake_result(value=110.0),
                standalone=lambda *_a, **_k: fake_result(),
                function_factory=lambda: SimpleNamespace(df_year=None, get_info=info),
                seed_globals=lambda _seed: None,
            )
            record = json.loads((output / "runs.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["method"], "Hybrid DDTW")
            self.assertNotEqual(record["method"], "Hybrid DTW")
            self.assertEqual(record["objective"], 110.0)
            self.assertIs(record["feasible"], False)
            self.assertIsNone(record["physical_lcoe"])
            self.assertIsNone(record["physical_lcoh"])
            self.assertIsNone(record["physical_agsr"])
            self.assertIs(record["objective_is_physical_lcoe"], False)
            text = (output / "resumen_hres2.txt").read_text(encoding="utf-8")
            self.assertIn("Mediana", text)
            self.assertIn("min", text.lower())

    def test_baselines_run_once_for_shared_protocol(self):
        module = load_module(BENCHMARK, "hres2_benchmark_records_under_test")
        calls = []

        def pipeline(**kwargs):
            calls.append(("hybrid", kwargs["stag_cfg"].use_ddtw))
            return fake_result()

        def standalone(_func, name, max_iters):
            calls.append(("baseline", name, max_iters))
            return fake_result(value=0.4)

        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "shared"
            module.ejecutar_benchmark_hres2(
                1, 1, str(output), modes=["dtw", "ddtw"], baselines=["PSO"],
                pipeline=pipeline, standalone=standalone,
                function_factory=lambda: SimpleNamespace(df_year=None, get_info=lambda _s: physical_info()),
                seed_globals=lambda _seed: None,
            )
            self.assertEqual(calls, [("hybrid", False), ("hybrid", True), ("baseline", "PSO", 1)])
            methods = [
                json.loads(line)["method"]
                for line in (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(methods, ["Hybrid DTW", "Hybrid DDTW", "PSO"])
            self.assertNotIn("ABC", methods)

    def test_combined_modes_keep_independent_holm_families_and_one_baseline_pass(self):
        module = load_module(BENCHMARK, "hres2_benchmark_records_under_test")
        baseline_calls = []

        def pipeline(**kwargs):
            base = 10.0 if kwargs["stag_cfg"].use_ddtw else 1.0
            value = base + 0.01 * len(baseline_calls)
            return fake_result(value=value, solution=[value, 12.0, 10.0, 1.0])

        def standalone(_func, name, max_iters):
            baseline_calls.append((name, max_iters))
            value = 5.0 + 0.01 * len(baseline_calls)
            return fake_result(value=value, solution=[value, 12.0, 10.0, 1.0])

        def get_info(solution):
            return physical_info(lcoe=float(solution[0]))

        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "combined"
            result = module.ejecutar_benchmark_hres2(
                8, 1, str(output), modes=["dtw", "ddtw"], baselines=["PSO", "GWO"],
                pipeline=pipeline, standalone=standalone,
                function_factory=lambda: SimpleNamespace(df_year=None, get_info=get_info),
                seed_globals=lambda _seed: None,
            )
            records = [json.loads(line) for line in (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(item[0] == "PSO" for item in baseline_calls), 8)
            self.assertEqual(sum(item[0] == "GWO" for item in baseline_calls), 8)
            self.assertEqual(sum(record["method"] == "PSO" for record in records), 8)
            self.assertEqual(sum(record["method"] == "GWO" for record in records), 8)
            self.assertEqual(sum(record["method"] == "Hybrid DTW" for record in records), 8)
            self.assertEqual(sum(record["method"] == "Hybrid DDTW" for record in records), 8)
            self.assertIn("stat_results", result)
            self.assertEqual(set(result["stat_results_by_hybrid"]), {"Hybrid DTW", "Hybrid DDTW"})
            dtw_rows = {row["algoritmo"]: row for row in result["stat_results_by_hybrid"]["Hybrid DTW"]["tabla_resumen"]}
            ddtw_rows = {row["algoritmo"]: row for row in result["stat_results_by_hybrid"]["Hybrid DDTW"]["tabla_resumen"]}
            self.assertEqual(set(dtw_rows), {"Hybrid DTW", "PSO", "GWO"})
            self.assertEqual(set(ddtw_rows), {"Hybrid DDTW", "PSO", "GWO"})
            self.assertEqual(dtw_rows["PSO"]["wtl"], "W")
            self.assertEqual(ddtw_rows["PSO"]["wtl"], "L")
            self.assertLess(dtw_rows["PSO"]["holm_p"], 0.05)
            self.assertLess(ddtw_rows["PSO"]["holm_p"], 0.05)
            self.assertFalse((output / "analisis_estadistico_pvalues.csv").exists())
            with (output / "analisis_dtw" / "analisis_estadistico_pvalues.csv").open(encoding="utf-8", newline="") as handle:
                dtw_names = [row["algoritmo"] for row in csv.DictReader(handle)]
            with (output / "analisis_ddtw" / "analisis_estadistico_pvalues.csv").open(encoding="utf-8", newline="") as handle:
                ddtw_names = [row["algoritmo"] for row in csv.DictReader(handle)]
            self.assertEqual(dtw_names, ["Hybrid DTW", "PSO", "GWO"])
            self.assertEqual(ddtw_names, ["Hybrid DDTW", "PSO", "GWO"])
            self.assertIs(result["stat_results_by_hybrid"]["Hybrid DTW"]["paired_inference_valid"], False)
            self.assertIs(result["stat_results_by_hybrid"]["Hybrid DDTW"]["paired_inference_valid"], False)

    def test_ejecutar_benchmark_hres2_preserves_prior_return_contract(self):
        module = load_module(BENCHMARK, "hres2_benchmark_records_under_test")
        signature = inspect.signature(module.ejecutar_benchmark_hres2)
        self.assertEqual(list(signature.parameters), [
            "n_runs", "max_iters", "output_dir", "modes", "baselines", "nonpaper",
            "pipeline", "standalone", "function_factory", "seed_globals", "seed_base",
        ])
        self.assertEqual(signature.parameters["modes"].kind, inspect.Parameter.KEYWORD_ONLY)
        with tempfile.TemporaryDirectory() as scratch:
            result = module.ejecutar_benchmark_hres2(
                1, 1, str(Path(scratch) / "legacy"),
                pipeline=lambda **_kwargs: fake_result(),
                standalone=lambda *_a, **_k: fake_result(),
                function_factory=lambda: SimpleNamespace(df_year=None, get_info=lambda _s: physical_info()),
                seed_globals=lambda _seed: None,
                baselines=[],
            )
        self.assertEqual(
            {"media_lcoe", "std_lcoe", "mejor_lcoe", "best_config", "stat_results"},
            set(result).intersection({"media_lcoe", "std_lcoe", "mejor_lcoe", "best_config", "stat_results"}),
        )


class AnalysisRecordTests(unittest.TestCase):
    def test_realizar_analisis_preserves_prior_keys_and_separates_best_from_median(self):
        module = load_module(ANALYSIS, "hres2_analysis_records_under_test")
        signature = inspect.signature(module.realizar_analisis_estadistico_completo)
        self.assertEqual(list(signature.parameters)[:3], ["resultados_dict", "output_dir", "algoritmo_referencia"])
        data = {
            "Hybrid DDTW": [3.0, 1.0, 2.0, 4.0],
            "PSO": [3.5, 3.5, 3.5, 3.5],
        }
        with tempfile.TemporaryDirectory() as scratch:
            result = module.realizar_analisis_estadistico_completo(data, scratch, "Hybrid DDTW")
            self.assertIn("friedman_stat", result)
            self.assertIn("friedman_p", result)
            self.assertIn("tabla_resumen", result)
            self.assertTrue(result["tabla_resumen"])
            with open(Path(scratch) / "analisis_estadistico_pvalues.csv", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            hybrid = next(row for row in rows if row["algoritmo"] == "Hybrid DDTW")
            self.assertNotIn("Hybrid DTW", [row["algoritmo"] for row in rows])
            self.assertEqual(float(hybrid["best_lcoe"]), 1.0)
            self.assertEqual(float(hybrid["median_lcoe"]), 2.5)
            self.assertNotEqual(hybrid["best_lcoe"], hybrid["median_lcoe"])
            self.assertIs(result["paired_inference_valid"], False)

    def test_infeasible_records_are_counted_and_excluded_from_lcoe_inference(self):
        module = load_module(ANALYSIS, "hres2_analysis_records_under_test")
        records = []
        for value in (1.0, 1.2, 1.1, 1.3):
            records.append({
                "method": "Hybrid DTW", "feasible": True, "physical_lcoe": value, "objective": value,
            })
        records.append({
            "method": "Hybrid DTW", "feasible": False, "physical_lcoe": None, "objective": 110.0,
        })
        for value in (2.0, 2.2, 2.1, 2.4):
            records.append({
                "method": "PSO", "feasible": True, "physical_lcoe": value, "objective": value,
            })
        records.append({
            "method": "SA", "feasible": False, "physical_lcoe": None, "objective": 999.0,
        })
        floats = {
            "Hybrid DTW": [1.0, 1.2, 1.1, 1.3, 110.0],
            "PSO": [2.0, 2.2, 2.1, 2.4],
            "SA": [999.0],
        }
        with tempfile.TemporaryDirectory() as scratch:
            result = module.realizar_analisis_estadistico_completo(
                floats, scratch, "Hybrid DTW", records=records,
            )
            saved = result["tabla_resumen"]
            names = [row["algoritmo"] for row in saved]
            self.assertEqual(names.count("SA"), 1)
            sa = next(row for row in saved if row["algoritmo"] == "SA")
            self.assertEqual(sa["n_records"], 1)
            self.assertIn("no finite feasible physical LCOE", sa["inference_reason"])
            self.assertIsNone(sa["best"])
            hybrid = next(row for row in saved if row["algoritmo"] == "Hybrid DTW")
            self.assertEqual(hybrid["n_records"], 5)
            self.assertEqual(hybrid["n_finite_feasible"], 4)
            self.assertAlmostEqual(hybrid["mean"], 1.15)
            pso = next(row for row in saved if row["algoritmo"] == "PSO")
            self.assertIsNone(pso["holm_p"])
            self.assertTrue(math.isnan(pso["mwu_p"]))
            self.assertEqual(pso["wtl"], "skipped")
            self.assertIn("incomplete reference", pso["inference_reason"].lower())
            self.assertNotIn(pso["wtl"], ("W", "T", "L"))
            self.assertEqual(result["wtl_counts"], {"W": 0, "T": 0, "L": 0})
            text = (Path(scratch) / "analisis_estadistico_pvalues.md").read_text(encoding="utf-8")
            self.assertIn("not valid", text.lower())
            self.assertIn("Mann-Whitney", text)

    def test_incomplete_reference_skips_every_comparison_without_invented_p(self):
        module = load_module(ANALYSIS, "hres2_analysis_records_under_test")
        records = [
            {"method": "Hybrid DTW", "feasible": True, "physical_lcoe": value}
            for value in (1.0, 1.1, 1.2)
        ]
        records.append({"method": "Hybrid DTW", "feasible": False, "physical_lcoe": None, "objective": 50.0})
        records.extend(
            {"method": "PSO", "feasible": True, "physical_lcoe": value}
            for value in (2.0, 2.1, 2.2, 2.3)
        )
        with tempfile.TemporaryDirectory() as scratch:
            result = module.realizar_analisis_estadistico_completo(
                {"Hybrid DTW": [1.0, 1.1, 1.2, 50.0], "PSO": [2.0, 2.1, 2.2, 2.3]},
                scratch, "Hybrid DTW", records=records, expected_n=4,
            )
            rows = {row["algoritmo"]: row for row in result["tabla_resumen"]}
            self.assertEqual(rows["Hybrid DTW"]["n_records"], 4)
            self.assertEqual(rows["Hybrid DTW"]["n_finite_feasible"], 3)
            self.assertAlmostEqual(rows["Hybrid DTW"]["mean"], 1.1)
            self.assertIsNone(rows["PSO"]["holm_p"])
            self.assertTrue(math.isnan(rows["PSO"]["mwu_p"]))
            self.assertEqual(rows["PSO"]["wtl"], "skipped")
            self.assertIn("all comparisons skipped", rows["PSO"]["inference_reason"].lower())
            self.assertNotIn("1.000", rows["PSO"]["inference_reason"])
            self.assertEqual(result["wtl_counts"], {"W": 0, "T": 0, "L": 0})

    def test_missing_reference_runs_skip_all_comparisons(self):
        module = load_module(ANALYSIS, "hres2_analysis_records_under_test")
        records = [
            {"method": "Hybrid DTW", "feasible": True, "physical_lcoe": value}
            for value in (1.0, 1.1, 1.2)
        ]
        records.extend(
            {"method": "PSO", "feasible": True, "physical_lcoe": value}
            for value in (3.0, 3.1, 3.2, 3.3)
        )
        with tempfile.TemporaryDirectory() as scratch:
            result = module.realizar_analisis_estadistico_completo(
                {"Hybrid DTW": [1.0, 1.1, 1.2], "PSO": [3.0, 3.1, 3.2, 3.3]},
                scratch, "Hybrid DTW", records=records, expected_n=4,
            )
            pso = next(row for row in result["tabla_resumen"] if row["algoritmo"] == "PSO")
            self.assertEqual(next(row for row in result["tabla_resumen"] if row["algoritmo"] == "Hybrid DTW")["n_records"], 3)
            self.assertIsNone(pso["holm_p"])
            self.assertTrue(math.isnan(pso["mwu_p"]))
            self.assertEqual(pso["wtl"], "skipped")
            self.assertIn("missing runs", pso["inference_reason"].lower())
            self.assertEqual(result["wtl_counts"], {"W": 0, "T": 0, "L": 0})

    def test_incomplete_comparator_skips_only_that_comparison(self):
        module = load_module(ANALYSIS, "hres2_analysis_records_under_test")
        records = [
            {"method": "Hybrid DTW", "feasible": True, "physical_lcoe": value}
            for value in (1, 2, 3, 4, 5, 6, 7, 8)
        ]
        records.extend(
            {"method": "PSO", "feasible": True, "physical_lcoe": value}
            for value in (10, 11, 12, 13, 14, 15, 16, 17)
        )
        records.extend(
            {"method": "SA", "feasible": True, "physical_lcoe": value}
            for value in (4, 5, 6)
        )
        records.append({"method": "SA", "feasible": False, "physical_lcoe": None, "objective": 80.0})
        with tempfile.TemporaryDirectory() as scratch:
            result = module.realizar_analisis_estadistico_completo(
                {
                    "Hybrid DTW": [1, 2, 3, 4, 5, 6, 7, 8],
                    "PSO": [10, 11, 12, 13, 14, 15, 16, 17],
                    "SA": [4, 5, 6, 80.0],
                },
                scratch, "Hybrid DTW", records=records, expected_n=8,
            )
            rows = {row["algoritmo"]: row for row in result["tabla_resumen"]}
            self.assertEqual(rows["SA"]["n_records"], 4)
            self.assertEqual(rows["SA"]["n_finite_feasible"], 3)
            self.assertAlmostEqual(rows["SA"]["mean"], 5.0)
            self.assertIsNone(rows["SA"]["holm_p"])
            self.assertTrue(math.isnan(rows["SA"]["mwu_p"]))
            self.assertEqual(rows["SA"]["wtl"], "skipped")
            self.assertIn("missing runs", rows["SA"]["inference_reason"].lower())
            self.assertEqual(rows["PSO"]["wtl"], "W")
            self.assertLess(rows["PSO"]["holm_p"], 0.05)
            self.assertEqual(result["wtl_counts"], {"W": 1, "T": 0, "L": 0})

    def test_complete_physical_samples_keep_directional_wtl(self):
        module = load_module(ANALYSIS, "hres2_analysis_records_under_test")
        records = []
        for method, values in {
            "Hybrid DTW": [1, 2, 3, 4, 5, 6, 7, 8],
            "PSO": [10, 11, 12, 13, 14, 15, 16, 17],
            "WOA": [0, 0, 0, 0, 1, 1, 1, 1],
        }.items():
            records.extend(
                {"method": method, "feasible": True, "physical_lcoe": value} for value in values
            )
        with tempfile.TemporaryDirectory() as scratch:
            result = module.realizar_analisis_estadistico_completo(
                {
                    "Hybrid DTW": [1, 2, 3, 4, 5, 6, 7, 8],
                    "PSO": [10, 11, 12, 13, 14, 15, 16, 17],
                    "WOA": [0, 0, 0, 0, 1, 1, 1, 1],
                },
                scratch, "Hybrid DTW", records=records, expected_n=8,
            )
            rows = {row["algoritmo"]: row for row in result["tabla_resumen"]}
            self.assertEqual(rows["PSO"]["wtl"], "W")
            self.assertEqual(rows["WOA"]["wtl"], "L")
            self.assertLess(rows["PSO"]["holm_p"], 0.05)
            self.assertLess(rows["WOA"]["holm_p"], 0.05)
            self.assertEqual(result["wtl_counts"], {"W": 1, "T": 0, "L": 1})
            self.assertIs(result["paired_inference_valid"], False)

    def test_wtl_requires_lower_lcoe_and_holm_adjusted_p(self):
        module = load_module(ANALYSIS, "hres2_analysis_records_under_test")
        reference = [1, 2, 3, 4, 5, 6, 7, 8]
        with tempfile.TemporaryDirectory() as scratch:
            adjusted = module.realizar_analisis_estadistico_completo(
                {
                    "Hybrid DTW": reference,
                    "PSO": [4, 5, 6, 7, 8, 9, 10, 11],
                    "GWO": list(reference),
                },
                scratch,
                "Hybrid DTW",
            )
            rows = {row["algoritmo"]: row for row in adjusted["tabla_resumen"]}
            self.assertLess(rows["PSO"]["mwu_p"], 0.05)
            self.assertGreaterEqual(rows["PSO"]["holm_p"], 0.05)
            self.assertEqual(rows["PSO"]["wtl"], "T")
            self.assertEqual(rows["GWO"]["wtl"], "T")
            self.assertIs(adjusted["paired_inference_valid"], False)
        with tempfile.TemporaryDirectory() as scratch:
            clear = module.realizar_analisis_estadistico_completo(
                {
                    "Hybrid DTW": reference,
                    "PSO": [10, 11, 12, 13, 14, 15, 16, 17],
                    "WOA": [0, 0, 0, 0, 1, 1, 1, 1],
                },
                scratch,
                "Hybrid DTW",
            )
            rows = {row["algoritmo"]: row for row in clear["tabla_resumen"]}
            self.assertEqual(rows["PSO"]["wtl"], "W")
            self.assertEqual(rows["WOA"]["wtl"], "L")
            self.assertLess(rows["PSO"]["holm_p"], 0.05)
            self.assertLess(rows["WOA"]["holm_p"], 0.05)
            self.assertEqual(clear["wtl_counts"], {"W": 1, "T": 0, "L": 1})


if __name__ == "__main__":
    unittest.main()
