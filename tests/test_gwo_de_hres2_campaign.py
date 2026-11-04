"""Public tests for the GWO-DE HRES2 multi-seed campaign (gwo_de_hres2.campaign).

Campaigns run in-process through campaign.main with campaign.load_function
patched to a cheap bounded function; the real model is only used by the
smoke run outside the suite.

Spec: odd/tasks/gwo-de-hres2.md (S6-S8).
"""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")
from scipy.stats import mannwhitneyu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gwo_de_hres2 import campaign  # noqa: E402
from tests.test_gwo_de_hres2 import FakeFunction  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n"
SMALL = ["--policies", "binary_complex", "vanilla_exploration", "--seeds", "1", "2", "3",
         "--pop-size", "4", "--iterations", "3"]


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(campaign, "load_function", FakeFunction), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = campaign.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class HolmTests(unittest.TestCase):
    def test_holm_step_down_is_monotone_and_capped(self):
        self.assertEqual(campaign.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(campaign.holm([0.6, 0.5]), [1.0, 1.0])
        self.assertEqual(campaign.holm([]), [])


class CampaignTests(unittest.TestCase):
    def test_small_campaign_writes_runs_statistics_tests_and_best_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "c")
            code, stdout, stderr = run_main(SMALL + ["--out", str(out)])
            self.assertEqual(code, 0, stderr)
            self.assertIn("12/12", stdout)

            cfg = json.loads((out / "campaign_config.json").read_text(encoding="utf-8"))
            self.assertEqual((cfg["status"], cfg["total_runs"], cfg["completed_runs"]),
                             ("completed", 12, 12))
            self.assertEqual(cfg["signals"], ["dtw", "ddtw"])

            runs = read_csv(out / "runs.csv")
            self.assertEqual(len(runs), 12)
            self.assertEqual(len((out / "runs.jsonl").read_text(encoding="utf-8").splitlines()), 12)
            self.assertEqual({(r["signal"], r["policy"]) for r in runs},
                             {(s, p) for s in ("dtw", "ddtw")
                              for p in ("binary_complex", "vanilla_exploration")})

            stats = read_csv(out / "statistics.csv")
            self.assertEqual(len(stats), 4)
            for row in stats:
                values = [float(r["objective"]) for r in runs
                          if (r["signal"], r["policy"]) == (row["signal"], row["policy"])]
                self.assertEqual(int(row["runs"]), 3)
                self.assertAlmostEqual(float(row["best"]), min(values))
                self.assertAlmostEqual(float(row["worst"]), max(values))
                self.assertAlmostEqual(float(row["mean"]), statistics.mean(values))
                self.assertAlmostEqual(float(row["median"]), statistics.median(values))
                self.assertAlmostEqual(float(row["sd"]), statistics.stdev(values))
                seeds = [int(r["seed"]) for r in runs
                         if (r["signal"], r["policy"]) == (row["signal"], row["policy"])
                         and float(r["objective"]) == min(values)]
                self.assertEqual(int(row["best_seed"]), min(seeds))
                best = json.loads((out / row["signal"] / row["policy"] / "best_run.json")
                                  .read_text(encoding="utf-8"))
                self.assertEqual(best["seed"], min(seeds))
                self.assertEqual(len(best["best_series"]), 3)

            tests = read_csv(out / "tests.csv")
            families = [r["family"] for r in tests]
            self.assertEqual(families.count("dtw"), 1)
            self.assertEqual(families.count("ddtw"), 1)
            self.assertEqual(families.count("dtw_vs_ddtw"), 2)
            for row in tests:
                if row["family"] == "dtw_vs_ddtw":
                    a = [float(r["objective"]) for r in runs
                         if r["signal"] == "dtw" and r["policy"] == row["a"]]
                    b = [float(r["objective"]) for r in runs
                         if r["signal"] == "ddtw" and r["policy"] == row["b"]]
                else:
                    a = [float(r["objective"]) for r in runs
                         if r["signal"] == row["family"] and r["policy"] == row["a"]]
                    b = [float(r["objective"]) for r in runs
                         if r["signal"] == row["family"] and r["policy"] == row["b"]]
                self.assertAlmostEqual(float(row["p_value"]),
                                       mannwhitneyu(a, b, alternative="two-sided").pvalue)
                self.assertGreaterEqual(float(row["p_holm"]), float(row["p_value"]))

            for signal in ("dtw", "ddtw"):
                self.assertTrue((out / signal / "convergence_comparison.png").read_bytes()
                                .startswith(PNG))
            resumen = (out / "resumen.txt").read_text(encoding="utf-8")
            self.assertIn("binary_complex", resumen)
            self.assertIn("Holm", resumen)

    def test_single_signal_campaign_has_no_cross_signal_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "c")
            code, _, stderr = run_main(SMALL + ["--signals", "ddtw", "--out", str(out)])
            self.assertEqual(code, 0, stderr)
            self.assertEqual(len(read_csv(out / "runs.csv")), 6)
            self.assertEqual({r["family"] for r in read_csv(out / "tests.csv")}, {"ddtw"})

    def test_dry_run_reports_plan_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "c")
            code, stdout, stderr = run_main(["--dry-run", "--out", str(out)])
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn("372 runs", stdout)
            self.assertFalse(out.exists())

    def test_non_empty_output_is_rejected_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "keep.txt").write_text("x", encoding="utf-8")
            code, stdout, stderr = run_main(SMALL + ["--out", tmp])
            self.assertEqual((code, stdout), (2, ""))
            self.assertIn("output directory is not empty", stderr)
            self.assertEqual(os.listdir(tmp), ["keep.txt"])

    def test_invalid_selection_is_rejected_before_writing(self):
        cases = [(["--seeds", "1", "1"], "duplicate seed"),
                 (["--pop-size", "3"], "--pop-size must be >= 4"),
                 (["--iterations", "0"], "--iterations must be >= 1"),
                 (["--policies", "bogus"], "invalid choice: 'bogus'")]
        for extra, message in cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp, "c")
                code, _, stderr = run_main(extra + ["--out", str(out)])
                self.assertEqual(code, 2)
                self.assertIn(message, stderr)
                self.assertFalse(out.exists())

    def test_failing_run_marks_campaign_failed(self):
        calls = {"n": 0}
        real = campaign.run_search

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return real(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "c")
            with mock.patch.object(campaign, "run_search", flaky):
                code, _, stderr = run_main(SMALL + ["--out", str(out)])
            self.assertEqual(code, 1)
            self.assertIn("boom", stderr)
            cfg = json.loads((out / "campaign_config.json").read_text(encoding="utf-8"))
            self.assertEqual((cfg["status"], cfg["completed_runs"]), ("failed", 1))
            self.assertEqual(len(read_csv(out / "runs.csv")), 1)


if __name__ == "__main__":
    unittest.main()
