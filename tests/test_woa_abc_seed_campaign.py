"""woa_abc.seed_campaign: one instance, many seeds, best run of each policy."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from woa_abc import benchmark, seed_campaign
from woa_abc.profiles import POLICIES

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "instancias"
RUN_FIELDS = ["file", "index", "policy", "seed", "status", "best_fitness", "known_optimum",
              "gap_pct", "evaluations", "generations", "changes", "elapsed_s"]
STAT_FIELDS = ["file", "index", "policy", "runs", "best_fitness", "worst_fitness",
               "mean_fitness", "median_fitness", "std_fitness", "best_seed",
               "known_optimum", "best_gap_pct", "mean_gap_pct", "mean_elapsed_s"]
TINY = ["--population-size", "4", "--iterations", "6", "--max-evaluations", "200"]


def main(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = seed_campaign.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def stub_export_factory(fitness, calls):
    """Export that writes only manifest and raw JSON; fitness(policy, seed) decides the value."""
    def export(request):
        calls.append(request)
        destination = benchmark.allocate_explicit_destination(request.destination)
        value = fitness(request.policy, request.seed)
        benchmark._write_json(destination / "manifest.json", {
            "status": "completed", "counts": {"n_profile_changes": request.seed % 3},
            "timing": {"solve_seconds": 0.25}})
        benchmark._write_json(destination / "runs/run_01.json", {
            "best_value": value, "gap": (24381 - value) / 24381 * 100, "evaluations": 200,
            "cycles_completed": 6, "metadata": {"seed": request.seed,
                                                "config": {"policy": request.policy}}})
        return benchmark.ExportOutcome(0, destination, "ok")
    return export


class SeedCampaignLayoutTest(unittest.TestCase):

    def test_real_tiny_run_writes_gwo_de_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "exp"
            code, stdout, stderr = main("--instances-dir", str(INSTANCES), "--files", "mknapcb1.txt",
                                        "--indices", "0", "--seeds", "1", "2", "--out", str(out), *TINY)
            self.assertEqual(code, 0, stderr)
            folder = out / "mknapcb1" / "inst_00"
            fields, rows = read_csv(folder / "runs.csv")
            self.assertEqual(fields, RUN_FIELDS)
            self.assertEqual(len(rows), 8)
            self.assertEqual({(r["policy"], r["seed"]) for r in rows},
                             {(p, s) for p in POLICIES for s in ("1", "2")})
            self.assertTrue(all(r["status"] == "completed" for r in rows))
            self.assertEqual({r["known_optimum"] for r in rows}, {"24381.0"})
            stat_fields, stats = read_csv(folder / "statistics.csv")
            self.assertEqual(stat_fields, STAT_FIELDS)
            self.assertEqual([s["policy"] for s in stats], list(POLICIES))
            self.assertTrue(all(s["runs"] == "2" for s in stats))
            for policy in POLICIES:
                best = folder / policy / "best_run"
                self.assertTrue((best / "manifest.json").is_file())
                self.assertTrue((best / "runs" / "run_01.json").is_file())
                self.assertTrue((best / "best_run_01" / "fitness_convergence.png").is_file())
            self.assertTrue((folder / "convergence_comparison.png").is_file())
            self.assertTrue((folder / "convergence_comparison.pdf").is_file())
            self.assertEqual(read_csv(out / "summary.csv")[1], rows)
            self.assertEqual(read_csv(out / "statistics.csv")[1], stats)
            config = json.loads((out / "campaign_config.json").read_text(encoding="utf-8"))
            self.assertEqual((config["status"], config["total_runs"], config["completed_runs"]),
                             ("completed", 8, 8))
            self.assertEqual(config["seeds"], [1, 2])
            leftovers = sorted(p.relative_to(out).as_posix() for p in out.iterdir())
            self.assertEqual(leftovers, ["campaign_config.json", "mknapcb1", "statistics.csv", "summary.csv"])
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             sorted(["runs.csv", "statistics.csv", "convergence_comparison.png",
                                     "convergence_comparison.pdf", *POLICIES]))


class SeedCampaignSelectionTest(unittest.TestCase):

    def _run(self, fitness, *extra, seeds=("3", "1", "2")):
        calls = []
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "exp"
        with patch.object(seed_campaign, "export_single_run", stub_export_factory(fitness, calls)), \
                patch.object(seed_campaign, "plot_comparison", lambda *args: None):
            code, stdout, stderr = main("--instances-dir", str(INSTANCES), "--files", "mknapcb1.txt",
                                        "--indices", "0", "--seeds", *seeds, "--out", str(out), *extra)
        return code, stderr, out, calls

    def test_best_run_is_highest_fitness_then_lower_seed(self):
        # binary_simple: seed 2 best; others tie at 500 -> seed 1 must win whatever the order.
        fitness = lambda policy, seed: 900.0 if (policy == "binary_simple" and seed == 2) else 500.0
        code, stderr, out, calls = self._run(fitness)
        self.assertEqual(code, 0, stderr)
        folder = out / "mknapcb1" / "inst_00"
        _, stats = read_csv(folder / "statistics.csv")
        winners = {s["policy"]: s["best_seed"] for s in stats}
        self.assertEqual(winners, {"binary_simple": "2", "binary_complex": "1",
                                   "continuous_simple": "1", "continuous_complex": "1"})
        for policy, seed in winners.items():
            raw = json.loads((folder / policy / "best_run" / "runs" / "run_01.json").read_text())
            self.assertEqual(raw["metadata"]["seed"], int(seed))
        self.assertEqual(len(calls), 12)
        self.assertEqual({c.reaction for c in calls}, {"policy"})
        self.assertEqual((calls[0].population_size, calls[0].iterations, calls[0].max_evaluations),
                         (30, 1000, 180030))
        self.assertFalse(any(p.name.startswith(".") for p in out.rglob("*")), "temp folders left")

    def test_failed_export_stops_with_failed_row(self):
        calls = []
        def export(request):
            if request.seed == 2 and request.policy == "binary_complex":
                calls.append(request)
                return benchmark.ExportOutcome(1, request.destination, "solver failed")
            return stub_export_factory(lambda p, s: 100.0, calls)(request)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "exp"
        with patch.object(seed_campaign, "export_single_run", export), \
                patch.object(seed_campaign, "plot_comparison", lambda *args: None):
            code, stdout, stderr = main("--instances-dir", str(INSTANCES), "--files", "mknapcb1.txt",
                                        "--indices", "0", "--seeds", "1", "2", "--out", str(out))
        self.assertEqual(code, 1)
        self.assertIn("error:", stderr)
        self.assertIn("seed=2", stderr)
        _, rows = read_csv(out / "mknapcb1" / "inst_00" / "runs.csv")
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[-1]["status"], "failed")
        self.assertEqual(rows[-1]["best_fitness"], "")
        config = json.loads((out / "campaign_config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["status"], config["completed_runs"]), ("failed", 5))
        self.assertEqual(len(calls), 6)


class SeedCampaignValidationTest(unittest.TestCase):

    def test_invalid_input_exits_2_and_creates_nothing(self):
        cases = {
            "bad file": ["--files", "otro.txt"],
            "path file": ["--files", "../mknapcb1.txt"],
            "duplicate seed": ["--seeds", "1", "1"],
            "negative seed": ["--seeds", "-1"],
            "duplicate index": ["--indices", "0", "0"],
            "out of range index": ["--indices", "30"],
            "bad population": ["--population-size", "0"],
        }
        for label, extra in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "exp"
                base = {"--files": ["mknapcb1.txt"], "--indices": ["0"], "--seeds": ["1"]}
                args = ["--instances-dir", str(INSTANCES), "--out", str(out)]
                for flag, values in base.items():
                    if flag not in extra:
                        args += [flag, *values]
                with patch.object(seed_campaign, "export_single_run",
                                  side_effect=AssertionError("must not run")):
                    code, stdout, stderr = main(*args, *extra)
                self.assertEqual(code, 2, label)
                self.assertIn("error:", stderr)
                self.assertFalse(out.exists(), label)

    def test_non_empty_out_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "exp"
            out.mkdir()
            (out / "keep.txt").write_text("x")
            with patch.object(seed_campaign, "export_single_run", side_effect=AssertionError("run")):
                code, _, stderr = main("--instances-dir", str(INSTANCES), "--files", "mknapcb1.txt",
                                       "--indices", "0", "--seeds", "1", "--out", str(out))
            self.assertEqual(code, 2)
            self.assertIn("not empty", stderr)
            self.assertEqual([p.name for p in out.iterdir()], ["keep.txt"])

    def test_dry_run_counts_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "exp"
            with patch.object(seed_campaign, "export_single_run", side_effect=AssertionError("run")):
                code, stdout, stderr = main("--instances-dir", str(INSTANCES), "--files", "mknapcb1.txt",
                                            "--indices", "0", "--out", str(out), "--dry-run")
            self.assertEqual(code, 0, stderr)
            self.assertIn("runs=124", stdout)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
