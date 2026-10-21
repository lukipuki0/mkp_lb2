"""Checks for the 135-run additive campaign. Exporter plots are mocked except one smoke."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from woa_abc.benchmark import (
    DEFAULT_OUTPUT_ROOT as BENCHMARK_OUTPUT_ROOT,
    EXPECTED_ARTIFACTS,
    ExportError,
    ExportOutcome,
    ExportRequest,
    export_single_run,
)
from woa_abc.campaign import (
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_SEED,
    FILES,
    GROUP_RUNS,
    INDICES,
    PLANNED_RUNS,
    RESULT_FIELDS,
    VARIANTS,
    CampaignError,
    build_parser,
    main,
    run_campaign,
)


GROUP_N = {
    1: 100,
    4: 100,
    7: 100,
    2: 250,
    5: 250,
    8: 250,
    3: 500,
    6: 500,
    9: 500,
}


def _write_file(path: Path, n: int, optima: tuple[float, float, float], m: int = 1) -> None:
    lines = [str(len(optima))]
    for optimum in optima:
        lines.append(f"{n} {m} {optimum}")
        lines.append(" ".join(["1"] * n))
        for _ in range(m):
            lines.append(" ".join(["1"] * n))
        lines.append(" ".join(["10"] * m))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _populate(directory: Path, optima: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    for number, n in GROUP_N.items():
        _write_file(directory / f"mknapcb{number}.txt", n, optima)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_stub(request: ExportRequest, *, status: str = "completed") -> None:
    destination = Path(request.destination)
    destination.mkdir(parents=True, exist_ok=True)
    cycles = 1 if request.variant == "v1" else 3
    fitness = {"v0": 1.0, "v1": 2.0, "v2": 3.0, "v3": 4.0, "v4": 5.0}[request.variant]
    raw = {
        "best_value": fitness,
        "gap": None if request.instance_index != 1 else 1.25,
        "evaluations": 10 + request.instance_index,
        "cycles_completed": cycles,
        "scout_limit": 100 if request.variant == "v1" else 30,
        "evaluations_by_phase": {"scouts": 4},
        "metadata": {"seed": request.seed, "config": {} if request.variant == "v0" else {"variant": request.variant}},
    }
    manifest = {
        "status": status,
        "counts": {
            "evaluations": raw["evaluations"],
            "cycles_completed": cycles,
            "n_recoveries": 2 if request.variant == "v2" else 0,
        },
        "timing": {"solve_seconds": 0.5},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    runs = destination / "runs"
    runs.mkdir(exist_ok=True)
    (runs / "run_01.json").write_text(json.dumps(raw), encoding="utf-8")


def _expected_destination(campaign: Path, file_name: str, index: int, variant: str) -> Path:
    number = int(file_name.replace("mknapcb", "").replace(".txt", ""))
    return campaign / str(GROUP_N[number]) / f"mknapcb{number}" / f"inst_{index:02d}" / variant


class CampaignPlanTests(unittest.TestCase):
    def _run(self, instances: Path, output: Path, calls: list, **kwargs):
        def exporter(request: ExportRequest):
            calls.append(request)
            _write_stub(request)
            return ExportOutcome(0, Path(request.destination), "ok")

        with patch("woa_abc.campaign.export_single_run", exporter):
            return run_campaign(instances, output, timestamp="plan", **kwargs)

    def test_selects_27_instances_and_five_variants_in_three_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances = root / "instances"
            output = root / "out"
            instances.mkdir()
            _populate(instances, (0.0, 99.0, 0.0))
            (instances / "counter.txt").write_text("0", encoding="utf-8")
            calls: list[ExportRequest] = []
            stdout = StringIO()
            with patch("sys.stdout", stdout):
                outcome = self._run(instances, output, calls)
            self.assertEqual(outcome.code, 0, outcome.message)
            log = stdout.getvalue()
            self.assertTrue(log.startswith("start variant=v0 file=mknapcb1.txt index=0 n=100\n"))
            self.assertEqual(log.count("start variant="), 135)
            self.assertEqual(log.count("completed variant="), 135)
            self.assertIn("completed variant=v4 file=mknapcb9.txt index=2 n=500", log)
            self.assertEqual(len(calls), PLANNED_RUNS)
            self.assertEqual(PLANNED_RUNS, 135)
            self.assertEqual(len(FILES), 9)
            self.assertEqual(INDICES, (0, 1, 2))
            self.assertEqual(VARIANTS, ("v0", "v1", "v2", "v3", "v4"))
            keys = [(call.instance.name, call.instance_index, call.variant) for call in calls]
            expected = [
                (name, index, variant)
                for name in FILES
                for index in INDICES
                for variant in VARIANTS
            ]
            self.assertEqual(keys, expected)
            self.assertEqual(len({(name, index) for name, index, _ in keys}), 27)
            self.assertEqual(calls[0].seed, DEFAULT_SEED)
            self.assertEqual(calls[0].population_size, DEFAULT_POPULATION_SIZE)
            self.assertEqual(calls[0].iterations, DEFAULT_ITERATIONS)
            self.assertEqual(calls[0].max_evaluations, DEFAULT_MAX_EVALUATIONS)
            self.assertIsNone(calls[0].scout_limit)
            self.assertEqual(calls[0].max_evaluations, 180030)
            campaign = output / "campaign_plan"
            self.assertEqual(outcome.output_dir, campaign)
            experiment = json.loads((campaign / "experiment.json").read_text(encoding="utf-8"))
            summary = json.loads((campaign / "campaign_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(experiment["runs_planned"], 135)
            self.assertEqual(experiment["status"], "completed")
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(experiment["superiority_claim"])
            self.assertFalse(summary["superiority_claim"])
            self.assertEqual(summary["runs_completed"], 135)
            self.assertEqual(summary["runs_failed"], 0)
            self.assertIsNone(summary["failed_selection"])
            self.assertIn("no superiority", summary["note"])
            self.assertEqual(summary["groups"], {"100": 45, "250": 45, "500": 45})
            self.assertEqual(GROUP_RUNS, 45)
            selections = experiment["selections"]
            self.assertEqual(len(selections), 135)
            by_index = {(item["file"], item["index"]): item["reference"] for item in selections}
            self.assertEqual(by_index[("mknapcb1.txt", 0)], 0.0)
            self.assertEqual(by_index[("mknapcb1.txt", 1)], 99.0)
            self.assertEqual(by_index[("mknapcb1.txt", 2)], 0.0)
            self.assertNotIn(24381, {item["reference"] for item in selections})
            rows = _read_csv(campaign / "campaign_results.csv")
            self.assertEqual(list(rows[0]), list(RESULT_FIELDS))
            self.assertEqual(len(rows), 135)
            index1 = [row for row in rows if row["file"] == "mknapcb2.txt" and row["index"] == "1"]
            index2 = [row for row in rows if row["file"] == "mknapcb2.txt" and row["index"] == "2"]
            self.assertEqual({row["reference"] for row in index1}, {"99"})
            self.assertEqual({row["gap"] for row in index1}, {"1.25"})
            self.assertEqual({row["reference"] for row in index2}, {"0"})
            self.assertEqual({row["gap"] for row in index2}, {""})
            v0 = rows[0]
            self.assertEqual(v0["variant"], "v0")
            self.assertEqual(v0["fitness"], "1")
            self.assertEqual(v0["evaluations"], "10")
            self.assertEqual(v0["cycles_completed"], "3")
            self.assertEqual(v0["scouts"], "4")
            self.assertEqual(v0["recoveries"], "0")
            self.assertEqual(v0["elapsed"], "0.5")
            self.assertEqual(v0["seed"], "43")
            self.assertEqual(v0["n"], "100")
            self.assertEqual(v0["m"], "1")
            v1 = [row for row in rows if row["variant"] == "v1"][0]
            self.assertEqual(v1["cycles_completed"], "1")
            self.assertEqual(v1["scout_limit"], "100")
            self.assertEqual(v1["status"], "completed")
            self.assertEqual((instances / "counter.txt").read_text(encoding="utf-8"), "0")

    def test_groups_use_loaded_n_and_exact_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances = root / "instances"
            output = root / "out"
            instances.mkdir()
            _populate(instances)
            (output / "run_ddtw_old").mkdir(parents=True)
            (output / "run_ddtw_old" / "sentinel.txt").write_text("keep", encoding="utf-8")
            calls: list[ExportRequest] = []
            outcome = self._run(instances, output, calls)
            self.assertEqual(outcome.code, 0, outcome.message)
            campaign = outcome.output_dir
            self.assertEqual(
                sorted(path.name for path in campaign.iterdir()),
                ["100", "250", "500", "campaign_results.csv", "campaign_summary.json", "experiment.json"],
            )
            for label, numbers in (("100", (1, 4, 7)), ("250", (2, 5, 8)), ("500", (3, 6, 9))):
                group = campaign / label
                self.assertEqual(
                    sorted(path.name for path in group.iterdir() if path.is_dir()),
                    [f"mknapcb{number}" for number in numbers],
                )
                self.assertEqual(len(list(group.glob("mknapcb*/inst_*/v*"))), 45)
                self.assertEqual(len(_read_csv(group / "group_summary.csv")), 45)
            for call in calls:
                self.assertEqual(
                    call.destination,
                    _expected_destination(campaign, call.instance.name, call.instance_index, call.variant),
                )
                self.assertNotIn("run_ddtw", call.destination.parts)
                self.assertFalse((call.destination / "run_ddtw_plan").exists())
            self.assertEqual((output / "run_ddtw_old" / "sentinel.txt").read_text(encoding="utf-8"), "keep")
            self.assertFalse(any(path.name.startswith("run_ddtw_") and path.name != "run_ddtw_old" for path in output.iterdir()))

    def test_wrong_loaded_n_is_not_regrouped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances = root / "instances"
            output = root / "out"
            instances.mkdir()
            _populate(instances)
            _write_file(instances / "mknapcb1.txt", 250, (0.0, 0.0, 0.0))
            calls = []

            def exporter(request):
                calls.append(request)
                raise AssertionError("solver must not run")

            with patch("woa_abc.campaign.export_single_run", exporter):
                with self.assertRaises(CampaignError) as caught:
                    run_campaign(instances, output, timestamp="wrongn")
            self.assertEqual(
                str(caught.exception),
                "mknapcb1.txt index 0 loaded n=250; expected 100",
            )
            self.assertEqual(calls, [])
            self.assertFalse(output.exists())

    def test_fresh_directory_refuses_existing_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances = root / "instances"
            output = root / "out"
            instances.mkdir()
            _populate(instances)
            existing = output / "campaign_taken"
            existing.mkdir(parents=True)
            (existing / "sentinel.txt").write_text("keep", encoding="utf-8")
            calls = []

            def exporter(request):
                calls.append(request)
                raise AssertionError("solver must not run")

            with patch("woa_abc.campaign.export_single_run", exporter):
                with self.assertRaises(CampaignError) as caught:
                    run_campaign(instances, output, timestamp="taken")
            self.assertIn("refusing to overwrite existing directory", str(caught.exception))
            self.assertEqual(calls, [])
            self.assertEqual((existing / "sentinel.txt").read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(existing.iterdir()), [existing / "sentinel.txt"])


class FailureTests(unittest.TestCase):
    def test_progress_is_persisted_and_failure_does_not_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances = root / "instances"
            output = root / "out"
            instances.mkdir()
            _populate(instances)
            calls = []

            def exporter(request: ExportRequest):
                calls.append(request)
                campaign = output / "campaign_fail"
                if len(calls) == 3:
                    rows = _read_csv(campaign / "campaign_results.csv")
                    exporter.rows_before = len(rows)
                    summary = json.loads((campaign / "campaign_summary.json").read_text(encoding="utf-8"))
                    exporter.status_before = summary["status"]
                    _write_stub(request, status="plot_failed")
                    (Path(request.destination) / "runs" / "run_01.json").write_text(
                        '{"best_value": 8, "kept": true}\n',
                        encoding="utf-8",
                    )
                    return ExportOutcome(1, Path(request.destination), "plot_failed: raw JSON kept; solver was not rerun")
                _write_stub(request)
                return ExportOutcome(0, Path(request.destination), "ok")

            with patch("woa_abc.campaign.export_single_run", exporter):
                outcome = run_campaign(instances, output, timestamp="fail")
            self.assertEqual(outcome.code, 1)
            self.assertEqual(
                outcome.message,
                "stopped on mknapcb1.txt index 0 variant v2; solver was not rerun",
            )
            self.assertEqual(len(calls), 3)
            self.assertEqual([call.variant for call in calls], ["v0", "v1", "v2"])
            self.assertEqual(exporter.rows_before, 2)
            self.assertEqual(exporter.status_before, "running")
            campaign = outcome.output_dir
            rows = _read_csv(campaign / "campaign_results.csv")
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["status"] for row in rows], ["completed", "completed", "failed"])
            summary = json.loads((campaign / "campaign_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertNotEqual(summary["status"], "completed")
            self.assertEqual(summary["runs_completed"], 2)
            self.assertEqual(summary["runs_failed"], 1)
            self.assertEqual(summary["failed_selection"]["variant"], "v2")
            self.assertEqual(summary["failed_selection"]["file"], "mknapcb1.txt")
            self.assertEqual(summary["failed_selection"]["index"], 0)
            raw = calls[2].destination / "runs" / "run_01.json"
            self.assertEqual(json.loads(raw.read_text(encoding="utf-8"))["kept"], True)
            self.assertFalse(summary["superiority_claim"])


class PreflightTests(unittest.TestCase):
    def test_invalid_or_missing_data_does_not_solve_or_allocate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances = root / "instances"
            output = root / "missing-file-out"
            instances.mkdir()
            _populate(instances)
            (instances / "mknapcb9.txt").unlink()
            (instances / "counter.txt").write_text("0", encoding="utf-8")
            calls = []

            def exporter(request):
                calls.append(1)
                raise AssertionError("solver must not run")

            with patch("woa_abc.campaign.export_single_run", exporter):
                with self.assertRaises(CampaignError) as missing:
                    run_campaign(instances, output, timestamp="missing")
                self.assertEqual(str(missing.exception), "missing instance file: mknapcb9.txt")
                self.assertFalse(output.exists())

                _write_file(instances / "mknapcb9.txt", 500, (0.0,))
                short_out = root / "short-out"
                with self.assertRaises(ValueError) as short:
                    run_campaign(instances, short_out, timestamp="short")
                self.assertIn("instance index must be in 0..0", str(short.exception))
                self.assertFalse(short_out.exists())

                broken = root / "broken"
                broken.mkdir()
                (broken / "counter.txt").write_text("0", encoding="utf-8")
                url_out = root / "url-out"
                with self.assertRaises(CampaignError) as url_error:
                    run_campaign("https://example.test/instances", url_out, timestamp="url")
                self.assertEqual(str(url_error.exception), "instances dir must be a local path, not a URL")
                self.assertFalse(url_out.exists())
                self.assertEqual((broken / "counter.txt").read_text(encoding="utf-8"), "0")

                with self.assertRaises(ValueError) as seed_error:
                    run_campaign(instances, root / "seed-out", seed=-1, timestamp="seed")
                self.assertEqual(str(seed_error.exception), "seed must be a nonnegative integer")
                self.assertFalse((root / "seed-out").exists())
                self.assertEqual(calls, [])
                self.assertEqual((instances / "counter.txt").read_text(encoding="utf-8"), "0")


class CliTests(unittest.TestCase):
    def test_cli_defaults_and_invalid_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--instances-dir", "instancias"])
        self.assertEqual(args.seed, 43)
        self.assertEqual(args.population_size, 30)
        self.assertEqual(args.iterations, 1000)
        self.assertEqual(args.max_evaluations, 180030)
        self.assertEqual(args.output_root, DEFAULT_OUTPUT_ROOT)
        self.assertEqual(DEFAULT_OUTPUT_ROOT, BENCHMARK_OUTPUT_ROOT)
        self.assertFalse(hasattr(args, "runs"))
        help_text = " ".join(parser.format_help().split())
        self.assertIn("135", help_text)
        self.assertIn("45", help_text)
        self.assertIn("v0", help_text)
        self.assertIn("v4", help_text)
        self.assertIn("180030", help_text)
        self.assertIn("(default: 43)", help_text)
        self.assertIn("(default: 30)", help_text)
        self.assertIn("does not download", help_text)
        self.assertIn("no superiority", help_text)
        self.assertNotIn("--runs", help_text)
        stdout = StringIO()
        stderr = StringIO()
        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            code = main(["--help"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("usage: woa_abc.campaign", stdout.getvalue())
        self.assertIn("135", stdout.getvalue())

        stdout = StringIO()
        stderr = StringIO()
        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            code = main([])
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("--instances-dir", stderr.getvalue())
        self.assertIn("required", stderr.getvalue())

        stdout = StringIO()
        stderr = StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                code = main(["--instances-dir", tmp, "--runs", "31"])
            self.assertEqual(code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("unrecognized arguments: --runs", stderr.getvalue())
            self.assertEqual(list(Path(tmp).iterdir()), [])

        stdout = StringIO()
        stderr = StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr), patch(
                "woa_abc.campaign.export_single_run",
                side_effect=AssertionError("solver must not run"),
            ):
                code = main(["--instances-dir", tmp, "--output-root", str(output), "--seed", "-1"])
            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "error: seed must be a nonnegative integer\n")
            self.assertFalse(output.exists())

        text = (Path(__file__).resolve().parents[1] / "woa_abc" / "README.md").read_text(encoding="utf-8")
        self.assertIn("python -m woa_abc.campaign --instances-dir instancias", text)
        self.assertIn("135", text)
        self.assertIn("180030", text)
        self.assertIn("mknapcb1", text)
        self.assertIn("100", text)
        self.assertIn("250", text)
        self.assertIn("500", text)
        self.assertIn("one run", text.lower())
        self.assertIn("no superiority", text.lower())


class ExporterCompatibilityTests(unittest.TestCase):
    def test_omitted_destination_keeps_default_exporter_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "tiny.txt"
            instance.write_text("1\n3 1 0\n5 4 3\n2 2 2\n4\n", encoding="utf-8")
            request = ExportRequest(
                instance=instance,
                seed=43,
                population_size=2,
                iterations=2,
                output_root=root / "out",
                timestamp="kept",
            )
            self.assertIsNone(request.destination)
            outcome = export_single_run(request)
            self.assertEqual(outcome.code, 0, outcome.message)
            self.assertEqual(outcome.output_dir, root / "out" / "run_ddtw_kept" / "tiny" / "inst_00")
            raw = json.loads((outcome.output_dir / "runs" / "run_01.json").read_text(encoding="utf-8"))
            self.assertIsNone(raw["gap"])
            self.assertNotIn("variant", raw["metadata"]["config"])

    def test_explicit_destination_smoke_writes_plots_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "mknapcb1.txt"
            instance.write_text("1\n3 1 0\n5 4 3\n2 2 2\n4\n", encoding="utf-8")
            destination = root / "100" / "mknapcb1" / "inst_00" / "v0"
            outcome = export_single_run(
                ExportRequest(
                    instance=instance,
                    seed=43,
                    population_size=2,
                    iterations=2,
                    max_evaluations=20,
                    output_root=root / "unused",
                    destination=destination,
                    variant="v0",
                )
            )
            self.assertEqual(outcome.code, 0, outcome.message)
            self.assertEqual(outcome.output_dir, destination)
            self.assertNotIn("run_ddtw", destination.parts)
            for relative in EXPECTED_ARTIFACTS:
                self.assertTrue((destination / relative).is_file(), relative)
            raw = json.loads((destination / "runs" / "run_01.json").read_text(encoding="utf-8"))
            self.assertIsNone(raw["gap"])
            self.assertNotIn("variant", raw["metadata"]["config"])
            self.assertEqual(raw["metadata"]["config"]["max_evaluations"], 20)
            calls = []

            def solver(*_args, **_kwargs):
                calls.append(1)
                raise AssertionError("solver must not run")

            with self.assertRaises(ExportError) as caught:
                export_single_run(
                    ExportRequest(
                        instance=instance,
                        seed=43,
                        population_size=2,
                        iterations=2,
                        output_root=root / "unused",
                        destination=destination,
                    ),
                    solver=solver,
                )
            self.assertEqual(str(caught.exception), f"refusing to overwrite existing path: {destination}")
            self.assertEqual(calls, [])
            self.assertTrue((destination / "runs" / "run_01.json").is_file())

            remote = root / "remote-out"
            with self.assertRaises(ExportError) as remote_error:
                export_single_run(
                    ExportRequest(
                        instance=instance,
                        population_size=2,
                        iterations=2,
                        output_root=root / "unused",
                        destination="https://example.test/out",
                    ),
                    solver=solver,
                )
            self.assertIn("local path", str(remote_error.exception))
            self.assertEqual(calls, [])
            self.assertFalse(remote.exists())


if __name__ == "__main__":
    unittest.main()
