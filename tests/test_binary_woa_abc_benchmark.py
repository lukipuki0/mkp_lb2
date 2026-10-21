"""Focused checks for the single-run WOA-ABC exporter. No 1000-cycle runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import matplotlib

import woa_abc.solver as hybrid
import woa_abc.benchmark as bench
from woa_abc.solver import CycleRecord, HybridResult
from woa_abc.benchmark import (
    BOX_NOTE,
    BOX_TITLE,
    CONVERGENCE_SIZE,
    DELTA_SIZE,
    DPI,
    EMPTY_DELTA_LABEL,
    EXPECTED_ARTIFACTS,
    PARAMETER_SIZE,
    RUN_CSV_FIELDS,
    ExportRequest,
    HistoryRow,
    build_hybrid_config,
    build_parser,
    draw_boxplot,
    draw_convergence,
    draw_delta,
    draw_parameters,
    export_single_run,
    inherited_stagnation_config,
    join_history,
    main,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_instance(path: Path, optimum: float = 0.0, instances: int = 1) -> None:
    body = ["1" if instances == 1 else str(instances)]
    for offset in range(instances):
        body.extend(
            [
                f"3 1 {optimum}",
                f"{5 + offset} 4 3",
                "2 2 2",
                "4",
            ]
        )
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _request(path: Path, root: Path, **overrides) -> ExportRequest:
    values = dict(
        instance=path,
        instance_index=0,
        seed=7,
        population_size=2,
        iterations=2,
        output_root=root,
        timestamp="tiny",
    )
    values.update(overrides)
    return ExportRequest(**values)


def _png_size(path: Path) -> tuple[int, int, float | None]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"{path} is not a PNG")
    if data[12:16] != b"IHDR":
        raise AssertionError("missing IHDR")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    dpi = None
    marker = data.find(b"pHYs")
    if marker >= 0:
        ppm = int.from_bytes(data[marker + 4 : marker + 8], "big")
        dpi = ppm * 0.0254
    return width, height, dpi


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _synthetic_result() -> HybridResult:
    history = [
        CycleRecord(0, 0, 2.0, 1.0, 5.0, 10, True),
        CycleRecord(1, 1, 1.5, 1.0, 8.0, 20, True),
        CycleRecord(2, 2, 1.0, 1.0, 8.0, 22, False),
    ]
    diagnostics = [
        {
            "cycle": 0,
            "ready": True,
            "fire": True,
            "recovery_applied": True,
            "delta": 4.0,
            "D1_vs_ramp": 6.0,
            "D2_vs_const": 2.0,
            "theta_c": 1.0,
            "theta_r": 2.0,
            "theta_delta": 0.5,
            "no_improve_len": 3,
            "trigger_streak": 2,
            "n": 40,
        },
        {
            "cycle": 2,
            "ready": False,
            "fire": True,
            "recovery_applied": False,
            "delta": 9.0,
            "D1_vs_ramp": 1.0,
            "D2_vs_const": 1.0,
            "n": 1,
        },
    ]
    return HybridResult(
        best_solution=[1, 0, 1],
        best_value=8.0,
        gap=None,
        evaluations=22,
        evaluations_by_phase={"initialization": 2},
        cycles_completed=2,
        termination_reason="evaluation_budget",
        history=history,
        phases_executed=["initialization"],
        source_trials=[1, 0],
        archive_size=22,
        scout_limit=4,
        diagnostics=diagnostics,
        metadata={
            "seed": 7,
            "instance": {"reference": 0.0, "index": 0},
            "config": {
                "population_size": 2,
                "iterations": 2,
                "max_evaluations": 40,
                "scout_limit": 4,
                "use_stagnation": True,
                "stagnation": {
                    "window": 40,
                    "band": 4,
                    "min_slope": 0.1,
                    "plateau_max": 15,
                    "patience": 8,
                    "use_ddtw": True,
                    "adapt_thresholds": True,
                    "p_low": 30.0,
                    "p_high": 70.0,
                    "improvement_tol": 0.0,
                },
            },
        },
    )


class InheritedConfigTests(unittest.TestCase):
    def test_copied_constants_match_batch_source_and_auto_band(self) -> None:
        source = (ROOT / "hybrid_mkp" / "batch_benchmark.py").read_text(encoding="utf-8")
        expected = {
            "STAG_WINDOW": "40",
            "STAG_BAND": "0",
            "STAG_MIN_SLOPE": "0.1",
            "STAG_PLATEAU_MAX": "15",
            "STAG_PATIENCE": "8",
            "STAG_USE_DDTW": "True",
            "STAG_ADAPT": "True",
            "STAG_P_LOW": "30.0",
            "STAG_P_HIGH": "70.0",
        }
        for name, value in expected.items():
            self.assertRegex(source, rf"{name}\s*=\s*{re.escape(value)}\b")
        block = source[source.index("stag_cfg = StagnationConfig(") : source.index("dtw_mode")]
        self.assertNotIn("improvement_tol", block)

        config = inherited_stagnation_config()
        self.assertEqual(config.window, 40)
        self.assertEqual(config.band, 4)
        self.assertEqual(config.min_slope, 0.1)
        self.assertEqual(config.plateau_max, 15)
        self.assertEqual(config.patience, 8)
        self.assertTrue(config.use_ddtw)
        self.assertTrue(config.adapt_thresholds)
        self.assertEqual(config.p_low, 30.0)
        self.assertEqual(config.p_high, 70.0)
        self.assertEqual(config.improvement_tol, 0.0)

        request = ExportRequest(instance=Path("unused.txt"))
        hybrid_config = build_hybrid_config(request)
        self.assertTrue(hybrid_config.use_stagnation)
        self.assertEqual(hybrid_config.stag_cfg.band, 4)
        self.assertTrue(hybrid_config.stag_cfg.use_ddtw)
        self.assertIsNone(hybrid_config.max_evaluations)
        self.assertEqual(hybrid._resolved_max_evaluations(hybrid_config), 210030)
        self.assertEqual(hybrid_config.population_size, 30)
        self.assertEqual(hybrid_config.iterations, 1000)

    def test_parser_defaults_reject_runs_and_help_is_one_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--instance", "instancias/mknapcb1.txt"])
        self.assertEqual(args.seed, 43)
        self.assertEqual(args.population_size, 30)
        self.assertEqual(args.iterations, 1000)
        self.assertEqual(args.instance_index, 0)
        self.assertIsNone(args.max_evaluations)
        self.assertIsNone(args.scout_limit)
        self.assertFalse(hasattr(args, "runs"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["--instance", "a.txt", "--runs", "31"])
        help_text = parser.format_help()
        self.assertIn("usage: woa_abc.benchmark", help_text)
        self.assertIn("exactly one", help_text)
        self.assertNotIn("--runs", help_text)
        self.assertNotIn("31", help_text)
        stdout = StringIO()
        with patch("sys.stdout", stdout):
            code = main(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("exactly one", stdout.getvalue())

    def test_default_output_root_follows_the_package_not_the_cwd(self) -> None:
        expected = Path(bench.__file__).resolve().parent / "resultados"
        self.assertEqual(bench.DEFAULT_OUTPUT_ROOT, expected)
        self.assertEqual(expected.name, "resultados")
        self.assertEqual(expected.parent.name, "woa_abc")
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            os.chdir(tmp)
            try:
                request = ExportRequest(instance=Path("unused.txt"))
                self.assertEqual(request.output_root, expected)
                self.assertEqual(bench.DEFAULT_OUTPUT_ROOT, expected)
                self.assertFalse(expected.is_relative_to(Path(tmp)))
            finally:
                os.chdir(previous)

    def test_import_does_not_pull_scipy_or_legacy_batch(self) -> None:
        self.assertEqual(matplotlib.get_backend().lower(), "agg")
        self.assertFalse(any(name == "scipy" or name.startswith("scipy.") for name in sys.modules))
        self.assertNotIn("hybrid_mkp.batch_benchmark", sys.modules)
        self.assertNotIn("hybrid_mkp.orchestrator", sys.modules)
        self.assertNotIn("mh.woa", sys.modules)


class JoinTests(unittest.TestCase):
    def test_join_uses_cycle_and_keeps_partial_blank(self) -> None:
        result = _synthetic_result()
        rows = join_history(result.history, result.diagnostics)
        self.assertEqual([row.cycle for row in rows], [0, 1, 2])
        self.assertEqual(rows[0].diagnostic["delta"], 4.0)
        self.assertIsNone(rows[1].diagnostic)
        self.assertIsNone(rows[1].recovery_applied)
        self.assertEqual(rows[2].diagnostic["delta"], 9.0)
        self.assertFalse(rows[2].completeness)
        self.assertTrue(rows[0].recovery_applied)
        self.assertFalse(rows[2].recovery_applied)
        stray = list(result.diagnostics) + [{"cycle": 9, "delta": 99.0, "fire": True, "recovery_applied": True}]
        self.assertEqual(len(join_history(result.history, stray)), 3)
        with self.assertRaises(bench.ExportError):
            join_history(result.history, [{"delta": 1.0}])
        with self.assertRaises(bench.ExportError):
            join_history(result.history, [result.diagnostics[0], dict(result.diagnostics[0])])

    def test_instantaneous_fitness_is_never_the_archived_best(self) -> None:
        rows = join_history(_synthetic_result().history, _synthetic_result().diagnostics)
        for row in rows:
            cell = bench.history_csv_row(row)
            self.assertEqual(cell["fitness_instantaneo"], "")
            self.assertEqual(cell["epoch"], "")
            self.assertNotEqual(cell["fitness_instantaneo"], cell["fitness_best"])
            self.assertEqual(cell["mh"], "WOA-ABC")
            self.assertEqual(cell["tipo"], "poblacional")
        self.assertEqual(bench.history_csv_row(rows[1])["dtw_ready"], "")
        self.assertEqual(bench.history_csv_row(rows[1])["dtw_delta"], "")
        self.assertEqual(bench.history_csv_row(rows[1])["estado_dtw"], bench.ESTADO_MISSING)
        self.assertEqual(bench.history_csv_row(rows[2])["estado_dtw"], bench.ESTADO_WARMUP)
        partial = join_history([_synthetic_result().history[2]], [])
        self.assertEqual(bench.history_csv_row(partial[0])["estado_dtw"], bench.ESTADO_PARTIAL)
        self.assertEqual(bench.history_csv_row(partial[0])["fitness_instantaneo"], "")
        self.assertEqual(bench.history_csv_row(rows[0])["dtw_delta"], "4")
        self.assertEqual(bench.history_csv_row(rows[2])["dtw_delta"], "9")


class ExporterTests(unittest.TestCase):
    def test_path_safety_is_local_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "tiny.txt"
            _write_instance(instance, 0.0)
            calls = []

            def solver(*_args, **_kwargs):
                calls.append(1)
                raise AssertionError("solver must not run")

            remote = ExportRequest(
                instance="https://example.test/mknapcb1.txt",
                output_root=root,
                timestamp="remote-url",
                population_size=2,
                iterations=2,
            )
            with self.assertRaises(bench.ExportError):
                export_single_run(remote, solver=solver)
            collapsed = _request(Path("https://example.test/mknapcb1.txt"), root, timestamp="collapsed")
            with self.assertRaises(bench.ExportError):
                export_single_run(collapsed, solver=solver)
            with self.assertRaises(bench.ExportError):
                export_single_run(
                    _request(instance, Path("http://example.test/out"), timestamp="remote"),
                    solver=solver,
                )
            with self.assertRaises(ValueError):
                export_single_run(_request(root / "missing.txt", root, timestamp="missing"), solver=solver)
            with self.assertRaises(ValueError):
                export_single_run(_request(instance, root, instance_index=3, timestamp="badidx"), solver=solver)
            file_root = root / "not-dir"
            file_root.write_text("x", encoding="utf-8")
            with self.assertRaises(bench.ExportError):
                export_single_run(_request(instance, file_root, timestamp="file"), solver=solver)
            stderr = StringIO()
            with patch("sys.stderr", stderr):
                code = main(
                    [
                        "--instance",
                        "https://example.test/mknapcb1.txt",
                        "--iterations",
                        "2",
                        "--population-size",
                        "2",
                        "--output-root",
                        str(root),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("local path", stderr.getvalue())
            self.assertEqual(calls, [])
            self.assertFalse((root / "run_ddtw_tiny").exists())
            self.assertFalse((root / "run_ddtw_badidx").exists())
            self.assertFalse(any(path.name.startswith("run_ddtw_") for path in root.iterdir()))

    def test_collision_refuses_existing_directory_without_solve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "tiny.txt"
            _write_instance(instance)
            parent = root / "run_ddtw_taken"
            parent.mkdir()
            (parent / "sentinel.txt").write_text("keep", encoding="utf-8")
            calls = []

            def solver(*_args, **_kwargs):
                calls.append(1)
                raise AssertionError("solver must not run")

            with self.assertRaises(bench.ExportError) as caught:
                export_single_run(_request(instance, root, timestamp="taken"), solver=solver)
            self.assertIn("refusing to overwrite", str(caught.exception))
            self.assertEqual(calls, [])
            self.assertEqual((parent / "sentinel.txt").read_text(encoding="utf-8"), "keep")
            self.assertFalse((parent / "tiny").exists())

    def test_tiny_run_keeps_reference_and_matches_raw_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "mknapcb1.txt"
            _write_instance(instance, 0.0)
            seen = {}

            def solver(inst, config, seed, metadata=None):
                seen["config"] = config
                seen["seed"] = seed
                seen["reference"] = inst.valor_optimo
                return hybrid.solve(inst, config, seed=seed, metadata=metadata)

            outcome = export_single_run(_request(instance, root, seed=43), solver=solver)
            self.assertEqual(outcome.code, 0, outcome.message)
            output = outcome.output_dir
            self.assertEqual(output, root / "run_ddtw_tiny" / "mknapcb1" / "inst_00")
            self.assertEqual(seen["seed"], 43)
            self.assertEqual(seen["reference"], 0.0)
            self.assertEqual(seen["config"].stag_cfg.band, 4)
            self.assertTrue(seen["config"].stag_cfg.use_ddtw)
            self.assertEqual(seen["config"].stag_cfg.min_slope, 0.1)
            self.assertEqual(seen["config"].stag_cfg.improvement_tol, 0.0)

            raw = json.loads((output / "runs" / "run_01.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["effective_config"], raw["metadata"]["config"])
            self.assertEqual(manifest["effective_config"]["stagnation"]["band"], 4)
            self.assertTrue(manifest["effective_config"]["stagnation"]["use_ddtw"])
            self.assertEqual(manifest["effective_config"]["stagnation"]["min_slope"], 0.1)
            self.assertEqual(manifest["detector_request"]["band"], 0)
            self.assertEqual(manifest["detector_request"]["band_mode"], "auto")
            self.assertEqual(manifest["input"]["reference"], 0.0)
            self.assertEqual(manifest["input"]["index"], 0)
            self.assertEqual(manifest["input"]["sha256"], hashlib.sha256(instance.read_bytes()).hexdigest())
            self.assertEqual(Path(manifest["input"]["path"]), instance.resolve())
            self.assertIsNone(raw["gap"])
            self.assertNotIn("24381", json.dumps(manifest))
            self.assertEqual(manifest["counts"]["evaluations"], raw["evaluations"])
            self.assertEqual(manifest["counts"]["evaluation_ceiling"], raw["metadata"]["config"]["max_evaluations"])
            self.assertNotEqual(manifest["counts"]["evaluations"], manifest["counts"]["evaluation_ceiling"])
            self.assertEqual(manifest["counts"]["cycles_completed"], raw["cycles_completed"])
            self.assertEqual(manifest["counts"]["n_switches"], 0)
            self.assertFalse(manifest["solver_rerun"])
            self.assertEqual(manifest["runs"], 1)
            self.assertTrue(manifest["output_completeness"]["complete"])
            self.assertEqual(set(manifest["output_completeness"]["expected"]), set(EXPECTED_ARTIFACTS))
            for relative in EXPECTED_ARTIFACTS:
                self.assertTrue((output / relative).is_file(), relative)
            self.assertFalse((output / "switches_gantt.png").exists())
            self.assertFalse((output / "analisis_estadistico_pvalues.csv").exists())

            run_row = _read_csv(output / "runs_resultados.csv")
            self.assertEqual(list(run_row[0]), list(RUN_CSV_FIELDS))
            self.assertEqual(len(run_row), 1)
            self.assertEqual(run_row[0]["run"], "1")
            self.assertEqual(run_row[0]["gap_pct"], "")
            self.assertEqual(run_row[0]["n_switches"], "0")
            self.assertEqual(run_row[0]["seed"], "43")
            self.assertEqual(int(run_row[0]["evaluations"]), raw["evaluations"])
            self.assertEqual(int(run_row[0]["cycles_completed"]), raw["cycles_completed"])
            self.assertEqual(int(run_row[0]["n_fires"]), manifest["counts"]["n_fires"])
            self.assertEqual(int(run_row[0]["n_recoveries"]), manifest["counts"]["n_recoveries"])
            self.assertEqual(run_row[0]["termination_reason"], raw["termination_reason"])
            self.assertEqual(float(run_row[0]["mejor_valor"]), raw["best_value"])
            self.assertLess(abs(float(run_row[0]["tiempo_s"]) - manifest["timing"]["solve_seconds"]), 0.01)

            history = _read_csv(output / "best_run_01" / "historial_dtw.csv")
            self.assertEqual(len(history), len(raw["history"]))
            self.assertEqual([int(row["cycle"]) for row in history], [item["cycle"] for item in raw["history"]])
            for row in history:
                self.assertEqual(row["fitness_instantaneo"], "")
                self.assertEqual(row["epoch"], "")
                self.assertNotEqual(row["fitness_instantaneo"], row["fitness_best"])
                self.assertEqual(row["mh"], "WOA-ABC")
            detail = (output / "best_run_01" / "historial_dtw_detalle.txt").read_text(encoding="utf-8")
            self.assertIn("no se copia el mejor archivado", detail)
            self.assertIn("se unen por cycle", detail)
            summary = (output / "resumen_pipeline.txt").read_text(encoding="utf-8")
            best = (output / "best_run_01" / "resumen_pipeline_best_run.txt").read_text(encoding="utf-8")
            for text in (summary, best):
                self.assertIn(bench.SUMMARY_BANNER, text)
                self.assertRegex(text, r"n_switches\s*:\s*0\b")
                self.assertIn("n_recoveries", text)
                self.assertIn("no es el consumo", text)
                self.assertIn("desconocido", text)
                self.assertIn("band=4", text)
                self.assertIn("pedido 0", text)
                self.assertIn("DDTW", text)
                self.assertNotIn("24381", text)
                self.assertNotIn("0.0s-", text)
            self.assertIn("Runs ejecutados", summary)
            self.assertRegex(summary, r"Runs ejecutados\s*:\s*1\b")
            self.assertNotIn("31 Runs", summary)

    def test_positive_reference_gap_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "known.txt"
            _write_instance(instance, 20.0)
            outcome = export_single_run(_request(instance, root, timestamp="known"), solver=hybrid.solve)
            self.assertEqual(outcome.code, 0, outcome.message)
            raw = json.loads((outcome.output_dir / "runs" / "run_01.json").read_text(encoding="utf-8"))
            row = _read_csv(outcome.output_dir / "runs_resultados.csv")[0]
            self.assertIsNotNone(raw["gap"])
            self.assertEqual(row["gap_pct"], f"{raw['gap']:.2f}")
            self.assertNotEqual(row["gap_pct"], "")
            manifest = json.loads((outcome.output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["input"]["reference"], 20.0)

    def test_budget_partial_cycle_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "partial.txt"
            _write_instance(instance, 0.0)
            outcome = export_single_run(
                _request(instance, root, timestamp="partial", max_evaluations=2),
                solver=hybrid.solve,
            )
            self.assertEqual(outcome.code, 0, outcome.message)
            raw = json.loads((outcome.output_dir / "runs" / "run_01.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["cycles_completed"], 0)
            self.assertEqual(raw["diagnostics"], [])
            self.assertFalse(raw["history"][-1]["completeness"])
            history = _read_csv(outcome.output_dir / "best_run_01" / "historial_dtw.csv")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["completeness"], "False")
            self.assertEqual(history[0]["dtw_ready"], "")
            self.assertEqual(history[0]["fitness_instantaneo"], "")
            self.assertEqual(history[0]["estado_dtw"], bench.ESTADO_PARTIAL)
            self.assertIn("Ciclo parcial sin diagnostico", history[0]["estado_dtw"])

    def test_plot_failure_keeps_raw_without_a_second_solve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "tiny.txt"
            _write_instance(instance, 0.0)
            calls = []

            def solver(inst, config, seed, metadata=None):
                calls.append(seed)
                return hybrid.solve(inst, config, seed=seed, metadata=metadata)

            with patch.object(bench, "write_plots", side_effect=RuntimeError("plot boom")):
                outcome = export_single_run(_request(instance, root, timestamp="fail"), solver=solver)
            self.assertEqual(outcome.code, 1)
            self.assertIn("solver was not rerun", outcome.message)
            self.assertEqual(calls, [7])
            output = outcome.output_dir
            raw_path = output / "runs" / "run_01.json"
            self.assertTrue(raw_path.is_file())
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertIn("best_solution", raw)
            self.assertTrue((output / "runs_resultados.csv").is_file())
            self.assertFalse((output / "boxplot_runs.png").exists())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "plot_failed")
            self.assertIn("plot boom", manifest["plot_error"])
            self.assertFalse(manifest["solver_rerun"])
            self.assertFalse(manifest["output_completeness"]["complete"])
            self.assertIn("boxplot_runs.png", manifest["output_completeness"]["missing"])
            self.assertEqual(raw_path.read_text(encoding="utf-8").count("best_solution"), 1)

    def test_synthetic_recoveries_do_not_become_switches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "synth.txt"
            _write_instance(instance, 0.0)
            result = _synthetic_result()

            def solver(*_args, **_kwargs):
                return result

            outcome = export_single_run(_request(instance, root, timestamp="synth"), solver=solver)
            self.assertEqual(outcome.code, 0, outcome.message)
            row = _read_csv(outcome.output_dir / "runs_resultados.csv")[0]
            self.assertEqual(row["n_switches"], "0")
            self.assertEqual(row["n_fires"], "2")
            self.assertEqual(row["n_recoveries"], "1")
            self.assertEqual(row["gap_pct"], "")
            history = _read_csv(outcome.output_dir / "best_run_01" / "historial_dtw.csv")
            self.assertEqual(history[0]["dtw_delta"], "4")
            self.assertEqual(history[1]["dtw_delta"], "")
            self.assertEqual(history[1]["fitness_instantaneo"], "")
            self.assertEqual(history[2]["dtw_delta"], "9")
            self.assertEqual(history[2]["completeness"], "False")
            manifest = json.loads((outcome.output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["effective_config"], result.metadata["config"])
            self.assertEqual(manifest["counts"]["n_switches"], 0)
            self.assertEqual(manifest["counts"]["n_recoveries"], 1)
            text = (outcome.output_dir / "resumen_pipeline.txt").read_text(encoding="utf-8")
            self.assertRegex(text, r"n_switches\s*:\s*0\b")
            self.assertRegex(text, r"n_recoveries\s*:\s*1\b")
            self.assertIn("no switches", text)


class PlotTests(unittest.TestCase):
    def _rows(self) -> list[HistoryRow]:
        return join_history(_synthetic_result().history, _synthetic_result().diagnostics)

    def test_fonts_recovery_markers_and_empty_delta_label(self) -> None:
        rows = self._rows()
        fig, ax = matplotlib.pyplot.subplots()
        draw_convergence(ax, rows, 0.0)
        self.assertEqual(ax.title.get_fontsize(), 20)
        self.assertEqual(ax.xaxis.label.get_fontsize(), 18)
        self.assertEqual(ax.yaxis.label.get_fontsize(), 18)
        self.assertEqual(ax.xaxis.label.get_text(), "Hybrid cycles")
        self.assertEqual(ax.legend_.get_texts()[0].get_fontsize(), 15)
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertIn("WOA\u2013ABC", labels)
        self.assertNotIn("parameter recovery", labels)
        self.assertNotIn("Optimum (0)", labels)
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots()
        draw_delta(ax, [HistoryRow(0, 1, 1, 2.0, 1.0, 0, 1.0, True, None, None)])
        drawn = " ".join(text.get_text() for text in ax.texts)
        self.assertIn("Empty diagnostic", drawn)
        self.assertIn("Not evidence of improvement", drawn)
        self.assertIn("empty diagnostic", ax.get_title())
        self.assertEqual(len(ax.lines), 0)
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots()
        draw_parameters(ax, rows)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.assertIn("a", by_label)
        self.assertIn("rho", by_label)
        self.assertIn("parameter recovery", by_label)
        self.assertIn("fire without recovery", by_label)
        recovery_path = by_label["parameter recovery"].get_paths()[0]
        fire_path = by_label["fire without recovery"].get_paths()[0]
        self.assertNotEqual(recovery_path.vertices.tolist(), fire_path.vertices.tolist())
        self.assertNotEqual(recovery_path.codes.tolist(), fire_path.codes.tolist())
        matplotlib.pyplot.close(fig)

        fig, ax = matplotlib.pyplot.subplots(figsize=(6, 5))
        draw_boxplot(fig, ax, 8.0, "tiny/inst_00", 0.0)
        self.assertIn("n = 1", ax.get_title())
        self.assertIn("not inferential", ax.get_title())
        self.assertNotIn("31", ax.get_title())
        self.assertEqual(fig.texts[0].get_text(), BOX_NOTE)
        self.assertEqual(ax.title.get_fontsize(), 11)
        self.assertEqual(ax.xaxis.label.get_fontsize(), 10)
        matplotlib.pyplot.close(fig)
        self.assertIn("n = 1", BOX_TITLE)
        self.assertIn("Empty diagnostic", EMPTY_DELTA_LABEL)

    def test_mkp_style_uses_one_segment_and_keeps_recovery_on_parameters(self) -> None:
        rows = self._rows()
        expected = [row.cycle for row in rows if row.recovery_applied]
        self.assertTrue(expected)

        def boundaries(ax):
            return [line for line in ax.lines
                    if line.get_linestyle() == "--"
                    and list(line.get_xdata())[:1] == list(line.get_xdata())[1:]]

        for draw in (lambda ax: draw_convergence(ax, rows, 0.0),
                     lambda ax: draw_delta(ax, rows)):
            with self.subTest(draw=draw):
                fig, ax = matplotlib.pyplot.subplots()
                try:
                    draw(ax)
                    starts = boundaries(ax)
                    self.assertEqual([line.get_xdata()[0] for line in starts], [0])
                    for line in starts:
                        self.assertEqual(line.get_linewidth(), 1.5)
                        self.assertEqual(line.get_alpha(), 0.5)
                        self.assertNotEqual(line.get_color(), "#FF5722")
                    labels = [text.get_text() for text in ax.legend_.get_texts()]
                    self.assertIn("WOA\u2013ABC", labels)
                    self.assertNotIn("parameter recovery", labels)
                    handle = ax.legend_.legend_handles[labels.index("WOA\u2013ABC")]
                    self.assertIsInstance(handle, matplotlib.patches.Patch)
                    self.assertEqual(ax.get_xlabel(), "Hybrid cycles")
                    self.assertFalse(any("switch" in label.lower() for label in labels))
                finally:
                    matplotlib.pyplot.close(fig)
        fig, ax = matplotlib.pyplot.subplots()
        try:
            draw_parameters(ax, rows)
            self.assertEqual([line.get_xdata()[0] for line in boundaries(ax)], expected)
            self.assertIn("parameter recovery", ax.get_legend_handles_labels()[1])
            self.assertNotIn("switch", " ".join(ax.get_legend_handles_labels()[1]).lower())
        finally:
            matplotlib.pyplot.close(fig)
        fig, ax = matplotlib.pyplot.subplots()
        try:
            draw_convergence(ax, rows * 50, 0.0)
            self.assertEqual(len(boundaries(ax)), 1)
            self.assertNotIn("parameter recovery", ax.get_legend_handles_labels()[1])
        finally:
            matplotlib.pyplot.close(fig)

    def test_saved_png_signature_and_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "plot.txt"
            _write_instance(instance, 0.0)
            outcome = export_single_run(_request(instance, root, timestamp="png"), solver=hybrid.solve)
            self.assertEqual(outcome.code, 0, outcome.message)
            expected = {
                "best_run_01/fitness_convergence.png": CONVERGENCE_SIZE,
                "best_run_01/dtw_delta.png": DELTA_SIZE,
                "best_run_01/parameter_recovery.png": PARAMETER_SIZE,
                "boxplot_runs.png": (6, 5),
            }
            for relative, size in expected.items():
                path = outcome.output_dir / relative
                width, height, dpi = _png_size(path)
                self.assertGreater(path.stat().st_size, 500, relative)
                self.assertEqual(width, int(size[0] * DPI), relative)
                self.assertEqual(height, int(size[1] * DPI), relative)
                self.assertIsNotNone(dpi)
                self.assertLess(abs(dpi - DPI), 1.0, relative)
            detail = (outcome.output_dir / "best_run_01" / "historial_dtw.csv").read_text(encoding="utf-8")
            self.assertNotIn("ESTANCAMIENTO", detail)
            delta = outcome.output_dir / "best_run_01" / "dtw_delta.png"
            self.assertTrue(delta.is_file())


class DocumentTests(unittest.TestCase):
    def test_doc_has_command_and_layout(self) -> None:
        text = (ROOT / "woa_abc" / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "python -m woa_abc.benchmark --instance instancias/mknapcb1.txt "
            "--instance-index 0 --seed 43 --population-size 30 --iterations 1000",
            text,
        )
        self.assertIn("woa_abc/resultados/run_ddtw_", text)
        self.assertIn("relocated", text)
        self.assertFalse((ROOT / "binary_woa_abc.py").exists())
        self.assertFalse((ROOT / "binary_woa_abc_benchmark.py").exists())
        self.assertFalse((ROOT / "binary_woa_abc.md").exists())
        for name in (
            "manifest.json",
            "runs/run_01.json",
            "runs_resultados.csv",
            "resumen_pipeline.txt",
            "boxplot_runs.png",
            "best_run_01/",
            "parameter_recovery.png",
        ):
            self.assertIn(name, text)
        self.assertIn("run_ddtw_", text)


class SerializationTests(unittest.TestCase):
    def test_raw_json_matches_solver_dataclass(self) -> None:
        result = _synthetic_result()
        self.assertEqual(json.loads(json.dumps(asdict(result))), asdict(result))


if __name__ == "__main__":
    unittest.main()
