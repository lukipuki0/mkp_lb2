"""Campaign contracts without expensive optimization or optional test runners."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sensitivity import design, run


def protocol():
    return {"schema_version": 1, "name": "test_pilot", "seeds": [43, 44],
            "modes": ["dtw", "ddtw"], "max_epoch_iters": None,
            "domains": [{"domain": "cec", "budget": 1000,
                         "baseline": {"window": 40, "band_ratio": 0.1},
                         "cases": [{"id": "f1", "function": 1, "dimension": 10}],
                         "factors": {"window": [40, 60], "band_ratio": [0.1001, 0.2]}}]}


def fake_adapter(domain, case, monitor, seed, budget, max_epoch_iters=None):
    return {"metric": float(seed), "metric_name": "optimum_error", "objective": float(seed + 300),
            "feasible": True, "actual_iters": budget, "n_epochs": 1, "n_switches": 0,
            "elapsed_seconds": 0.001, "effective_monitor": {"SA": monitor}, "solution": [0] * 10}


class DesignTests(unittest.TestCase):
    def test_provenance_includes_mkp_binarization_dependency(self):
        source = run.provenance()
        self.assertIn("lb2/binarization.py", source["source_files"])

    def test_oat_deduplicates_baselines_and_integer_bands(self):
        tasks = design.expand_tasks(protocol())
        self.assertEqual(len(tasks), 12)  # baseline + window + band; two modes, two seeds
        self.assertEqual(len({task["task_id"] for task in tasks}), len(tasks))
        self.assertEqual(sum(task["is_baseline"] for task in tasks), 4)
        by_configuration = {}
        for task in tasks:
            by_configuration.setdefault(task["config_id"], []).append(task["seed"])
            self.assertIn("band_ratio", task["requested_monitor"])
            self.assertNotIn("band_ratio", task["monitor"])
            self.assertEqual(task["monitor"]["use_ddtw"], task["mode"] == "ddtw")
        self.assertTrue(all(seeds == [43, 44] for seeds in by_configuration.values()))

    def test_ids_stable_under_factor_order_and_numeric_equivalence(self):
        first = protocol()
        second = copy.deepcopy(first)
        second["domains"][0]["factors"] = dict(reversed(list(first["domains"][0]["factors"].items())))
        second["domains"][0]["baseline"]["p_low"] = 30
        second["domains"][0]["baseline"]["p_high"] = 70.0
        self.assertEqual([t["task_id"] for t in design.expand_tasks(first)],
                         [t["task_id"] for t in design.expand_tasks(second)])

    def test_rejects_invalid_fields_and_values(self):
        for key, value in [("seeds", [43, 43]), ("seeds", [True]), ("seeds", [2**32]),
                           ("modes", ["dtw", "dtw"]), ("max_epoch_iters", 0),
                           ("schema_version", True), ("name", "../escape"), ("name", "CON")]:
            with self.subTest(key=key, value=value):
                sample = protocol()
                sample[key] = value
                with self.assertRaises(ValueError):
                    design.validate_protocol(sample)
        for monitor in [{"window": True}, {"window": 1}, {"band_ratio": 0},
                        {"band_ratio": 1.1}, {"min_slope": float("nan")},
                        {"p_low": 80, "p_high": 20}, {"use_ddtw": False},
                        {"adapt_thresholds": "yes"}]:
            with self.subTest(monitor=monitor):
                sample = protocol()
                sample["domains"][0]["baseline"].update(monitor)
                with self.assertRaises(ValueError):
                    design.validate_protocol(sample)
        for unknown in ["extra", "configurations"]:
            sample = protocol()
            sample[unknown] = []
            with self.assertRaises(ValueError):
                design.validate_protocol(sample)

    def test_rejects_duplicate_and_invalid_cases(self):
        sample = protocol()
        sample["domains"][0]["cases"] *= 2
        with self.assertRaises(ValueError):
            design.validate_protocol(sample)
        sample = protocol()
        sample["domains"][0]["cases"][0]["dimension"] = 10.0
        with self.assertRaises(ValueError):
            design.validate_protocol(sample)
        sample = protocol()
        sample["domains"] *= 2
        with self.assertRaises(ValueError):
            design.validate_protocol(sample)

    def test_mkp_requires_local_bounded_case_and_reference_source(self):
        sample = protocol()
        spec = sample["domains"][0]
        spec.update(domain="mkp", cases=[{"id": "mkp", "file": "instancias/mknapcb1.txt", "index": 0}])
        design.validate_protocol(sample)
        for changes in [{"file": "../outside.txt"}, {"index": 30}, {"index": True},
                        {"reference_value": 24381}, {"file": "https://example.org/data"}]:
            altered = copy.deepcopy(sample)
            altered["domains"][0]["cases"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                design.validate_protocol(altered)

    def test_checked_in_configs_are_valid_and_tiny_smoke_is_explicit(self):
        smoke = design.load_protocol(design.REPO_ROOT / "sensitivity/configs/smoke.json")
        pilot = design.load_protocol(design.REPO_ROOT / "sensitivity/configs/rotational_pilot_1k.json")
        self.assertEqual(len(design.expand_tasks(smoke)), 6)
        self.assertEqual(smoke["max_epoch_iters"], 4)
        self.assertIsNone(pilot["max_epoch_iters"])
        self.assertEqual(pilot["seeds"], list(range(43, 74)))
        self.assertEqual({domain["domain"] for domain in pilot["domains"]}, {"mkp", "cec", "hres"})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "resultados/sensitivity/test_pilot"
        self.provenance_patch = patch.object(run, "provenance", return_value={"source_fingerprint": "test"})
        self.provenance_patch.start()
        self.addCleanup(self.provenance_patch.stop)
        root_patch = patch.object(run, "REPO_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_dry_run_imports_no_adapter_and_creates_no_output(self):
        config = self.root / "protocol.json"
        config.write_text(json.dumps(protocol()))
        with patch.dict("sys.modules", {"sensitivity.adapters": None}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["--config", str(config), "--dry-run"]), 0)
        self.assertFalse(self.output.exists())

    def test_resume_completes_only_pending_tasks_and_exports_exact_json(self):
        records = run.run_campaign(protocol(), self.output, limit=1, adapter=fake_adapter)
        self.assertEqual(len(records), 1)
        with self.assertRaisesRegex(ValueError, "already contains"):
            run.run_campaign(protocol(), self.output, adapter=fake_adapter)
        calls = []
        def tracked(*args, **kwargs):
            calls.append(args)
            return fake_adapter(*args, **kwargs)
        records = run.run_campaign(protocol(), self.output, resume=True, adapter=tracked)
        self.assertEqual(len(calls), 11)
        self.assertEqual(len(records), 12)
        with patch.dict("sys.modules", {"sensitivity.adapters": None}):
            self.assertEqual(len(run.run_campaign(protocol(), self.output, resume=True)), 12)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["successful_tasks"], 12)
        self.assertEqual(manifest["pending_tasks"], 0)
        self.assertEqual(manifest["protocol"], protocol())
        self.assertEqual(len((self.output / "runs.jsonl").read_text().splitlines()), 12)
        self.assertEqual(len(run.load_records(self.output)), 12)

    def test_failed_tasks_recorded_then_retried(self):
        def fail(*args, **kwargs):
            raise RuntimeError("planned failure")
        records = run.run_campaign(protocol(), self.output, limit=1, adapter=fail)
        self.assertEqual(records[0]["status"], "failed")
        self.assertIn("planned failure", records[0]["error"])
        records = run.run_campaign(protocol(), self.output, resume=True, limit=1, adapter=fake_adapter)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "success")

    def test_rejects_changed_protocol_source_data_and_corrupt_metadata(self):
        run.run_campaign(protocol(), self.output, limit=1, adapter=fake_adapter)
        altered = protocol()
        altered["seeds"] = [45]
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            run.run_campaign(altered, self.output, resume=True, adapter=fake_adapter)
        with patch.object(run, "provenance", return_value={"source_fingerprint": "different"}):
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                run.run_campaign(protocol(), self.output, resume=True, adapter=fake_adapter)
        with patch.object(run, "input_fingerprints", return_value={"input.txt": "different"}):
            with self.assertRaisesRegex(ValueError, "input data changed"):
                run.run_campaign(protocol(), self.output, resume=True, adapter=fake_adapter)
        record_path = next((self.output / "records").glob("*.json"))
        record = json.loads(record_path.read_text())
        record["seed"] = 1
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "metadata"):
            run.run_campaign(protocol(), self.output, resume=True, adapter=fake_adapter)

    def test_incomplete_temporary_record_is_not_success(self):
        run.run_campaign(protocol(), self.output, limit=1, adapter=fake_adapter)
        task = design.expand_tasks(protocol())[1]
        temporary = self.output / "records" / f"{task['task_id']}.json.tmp"
        temporary.write_text('{"partial":')
        with self.assertWarnsRegex(RuntimeWarning, "incomplete temporary"):
            records = run.run_campaign(protocol(), self.output, resume=True, limit=1, adapter=fake_adapter)
        self.assertEqual(len(records), 2)
        self.assertFalse(temporary.exists())

    def test_invalid_result_recorded_as_failure_and_null_metric_allowed(self):
        def invalid(*args, **kwargs):
            result = fake_adapter(*args, **kwargs)
            result["objective"] = float("nan")
            return result
        records = run.run_campaign(protocol(), self.output, limit=1, adapter=invalid)
        self.assertEqual(records[0]["status"], "failed")
        def unavailable(*args, **kwargs):
            result = fake_adapter(*args, **kwargs)
            result["metric"] = None
            return result
        records = run.run_campaign(protocol(), self.output, resume=True, limit=1, adapter=unavailable)
        self.assertEqual(records[0]["status"], "success")
        self.assertIsNone(records[0]["metric"])

    def test_rejects_unsafe_output_nonpositive_limit_and_lock(self):
        with self.assertRaisesRegex(ValueError, "subdirectory"):
            run.run_campaign(protocol(), self.root / "unrelated", adapter=fake_adapter)
        with self.assertRaisesRegex(ValueError, "limit"):
            run.run_campaign(protocol(), self.output, limit=0, adapter=fake_adapter)
        run.run_campaign(protocol(), self.output, limit=1, adapter=fake_adapter)
        (self.output / ".campaign.lock").write_text("active worker")
        with self.assertRaisesRegex(ValueError, "locked"):
            run.run_campaign(protocol(), self.output, resume=True, adapter=fake_adapter)


if __name__ == "__main__":
    unittest.main()
