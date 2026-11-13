"""Report-only recovery contracts: synthetic saved records, never optimization."""
import builtins
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

ENTRY = Path(__file__).resolve().parents[1] / "HRES2-H2/report_asoc_ddtw_noabc.py"
POPULATION = ["PSO", "GWO", "WOA", "EHO", "ACO"]


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.source.mkdir()
        self.output = Path(self.temp.name) / "report"
        self.manifest = {"effective_configuration": {
            "n_runs": 31, "seed_base": 42, "nominal_iterations": 1000,
            "modes": ["ddtw"], "hybrid_population": POPULATION,
            "baselines": ["PSO"]}}
        self.records = [{
            "run": run, "seed": 42 + run, "method": "Hybrid DDTW",
            "mode": "ddtw", "role": "hybrid", "feasible": run != 31,
            "objective": 100 + run, "physical_lcoe": run / 100,
            "physical_lcoh": 17.0, "physical_agsr": 0.2,
            "solution": [174.0, 14.0, 50.0, 2.0],
            "decoded": dict(zip(("wind_mw", "pv_mw", "n_el_units", "electrolyzer_mw",
                "battery_mw", "battery_duration_h", "battery_mwh", "total_h2_kg",
                "electrolyzer_cf"), (174.0, 26.0, 14, 70.0, 50.0, 4.0, 200.0, 8862879.0, 0.759))),
            "nominal_iterations": 1000, "actual_iterations": 1040,
            "overshoot_iterations": 40, "elapsed_seconds": 3.0,
            "result_elapsed_seconds": None,
        } for run in range(1, 32)]
        baseline = dict(self.records[0], method="PSO", role="baseline", mode="baseline", seed=1043)
        self.records.append(baseline)

    def invoke(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if any(word in name.lower() for word in ("benchmark", "orchestrator", "wpeb", "numpy")):
                raise AssertionError("report imported optimization stack: " + name)
            return original_import(name, *args, **kwargs)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("builtins.__import__", guarded), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            spec = importlib.util.spec_from_file_location("report_under_test", ENTRY)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            code = module.main(["--source", str(self.source), "--output", str(self.output)])
        return code, stdout.getvalue(), stderr.getvalue()

    def save(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.source / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.records), encoding="utf-8")
        return {p.name: p.read_bytes() for p in self.source.iterdir()}

    def test_export_statistics_provenance_preservation_and_existing_destination(self):
        before = self.save()
        self.assertEqual(self.invoke(), (0, "Recovered 31 Hybrid DDTW records; excluded 1 baseline records.\n", ""))
        summary = json.loads((self.output / "summary.json").read_text())
        values = [r["physical_lcoe"] for r in self.records[:31]]
        self.assertEqual(summary["physical_lcoe"], {"mean": statistics.mean(values),
            "sample_std": statistics.stdev(values), "min": min(values), "max": max(values), "units": "CNY/kWh"})
        self.assertEqual(summary["feasible_count"], 30)
        self.assertEqual(summary["best_feasible_record"], self.records[0])
        self.assertEqual(summary["excluded_baseline_count"], 1)
        self.assertEqual(summary["unavailable_telemetry"], ["convergence", "DDTW", "Gantt", "switches"])
        for name, raw in before.items():
            self.assertEqual(summary["sources"][name], {"path": str((self.source / name).resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
        exported = [json.loads(line) for line in (self.output / "runs.jsonl").read_text().splitlines()]
        self.assertEqual(exported, self.records[:31])
        with (self.output / "runs.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 31)
        self.assertEqual(float(rows[0]["physical_lcoe"]), 0.01)
        self.assertIn("not completed", (self.output / "summary.md").read_text())
        output_before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.assertEqual(self.invoke(), (2, "", "error: refusing to overwrite existing output destination\n"))
        self.assertEqual(output_before, {p.name: p.read_bytes() for p in self.output.iterdir()})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})

    def test_reject_invalid_records_before_any_output(self):
        mutations = {
            "missing": lambda: self.records.pop(0),
            "duplicate": lambda: self.records.append(self.records[0]),
            "seed": lambda: self.records[0].update(seed=44),
            "method": lambda: self.records[0].update(method="Hybrid DTW"),
            "mode": lambda: self.records[0].update(mode="dtw"),
            "id": lambda: self.records[0].update(run=True),
            "nonfinite": lambda: self.records[0].update(physical_lcoe=float("nan")),
            "null": lambda: self.records[0].update(physical_lcoh=None),
            "decoded": lambda: self.records[0]["decoded"].update(wind_mw=float("inf")),
            "baseline": lambda: self.records[-1].update(seed=43),
            "population": lambda: self.manifest["effective_configuration"].update(hybrid_population=POPULATION + ["ABC"]),
        }
        original = json.dumps(self.records)
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                self.records = json.loads(original)
                mutate()
                before = self.save()
                self.assertEqual(self.invoke(), (2, "", "error: invalid saved campaign\n"))
                self.assertFalse(self.output.exists())
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
