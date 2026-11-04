"""Public tests for the GWO-DE DTW/DDTW hybrid on HRES2 (gwo_de_hres2).

Algorithm tests use a cheap bounded function with the HRES2Function
interface (lb_vector, ub_vector, _evaluate, get_info). The CLI tests run the
real HRES2 model with a tiny population and two generations.

Spec: odd/tasks/gwo-de-hres2.md (S1-S5).
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtw_stagnation import StagnationConfig  # noqa: E402
from gwo_de_dtw.controller import ProfileController  # noqa: E402
from gwo_de_dtw.policies import CAMPAIGN_POLICIES  # noqa: E402
from gwo_de_hres2 import config  # noqa: E402
from gwo_de_hres2.control import run_search  # noqa: E402
from gwo_de_hres2.search import GWODEContinuous  # noqa: E402


class FakeFunction:
    """Shifted sphere on HRES2-like bounds; minimum 0.25 at `target`."""

    def __init__(self):
        self.lb_vector = np.array([0.0, 10.0, 0.0, 0.0])
        self.ub_vector = np.array([200.0, 20.0, 50.0, 2.0])
        self.target = np.array([120.0, 14.0, 20.0, 1.0])
        self.calls = 0

    def _evaluate(self, x):
        self.calls += 1
        scaled = (np.asarray(x) - self.target) / (self.ub_vector - self.lb_vector)
        return 0.25 + float(np.sum(scaled ** 2))

    def get_info(self, x):
        value = self._evaluate(x)
        return {"feasible": True, "lcoe_cny_per_kwh": value, "lcoh_cny_per_kg": 2 * value,
                "agsr": 0.1, "wind_mw": float(x[0]), "pv_mw": 200.0 - float(x[0]),
                "n_el_units": int(round(x[1])), "electrolyzer_mw": 5.0 * round(x[1]),
                "battery_mw": float(x[2]), "battery_duration_h": 1.0,
                "battery_mwh": float(x[2]), "total_h2_kg": 1.0, "electrolyzer_cf": 0.5}


def monitor_cfg(use_ddtw=True, window=5):
    return StagnationConfig(window=window, band=1, min_slope=0.1, plateau_max=3,
                            patience=2, use_ddtw=use_ddtw, adapt_thresholds=True,
                            p_low=30.0, p_high=70.0)


class SearchTests(unittest.TestCase):
    def test_to_physical_maps_normalized_corners_to_bounds(self):
        func = FakeFunction()
        search = GWODEContinuous(func, np.random.default_rng(0), pop_size=4)
        np.testing.assert_allclose(search.to_physical(-np.ones(4)), func.lb_vector)
        np.testing.assert_allclose(search.to_physical(np.ones(4)), func.ub_vector)
        np.testing.assert_allclose(search.to_physical(np.zeros(4)),
                                   (func.lb_vector + func.ub_vector) / 2)

    def test_evaluations_are_n_at_init_plus_2n_per_generation(self):
        func = FakeFunction()
        search = GWODEContinuous(func, np.random.default_rng(0), pop_size=5)
        search.initialize()
        self.assertEqual(search.evaluations, 5)
        for _ in range(3):
            search.step()
        self.assertEqual(search.evaluations, 5 + 3 * 10)
        self.assertEqual(func.calls, search.evaluations)
        self.assertEqual(sum(search.last_step_stats[k] for k in ("gwo_wins", "de_wins")), 5)

    def test_rejects_population_below_four(self):
        with self.assertRaises(ValueError):
            GWODEContinuous(FakeFunction(), np.random.default_rng(0), pop_size=3)


class RunSearchTests(unittest.TestCase):
    def test_every_campaign_policy_runs_and_records_a_bounded_minimum(self):
        func = FakeFunction()
        for policy in CAMPAIGN_POLICIES:
            with self.subTest(policy=policy):
                rec = run_search(func, monitor_cfg(), policy=policy, pop_size=6,
                                 iterations=12, seed=3)
                self.assertEqual(rec["policy"], policy)
                self.assertEqual(rec["generations"], 12)
                self.assertEqual(len(rec["best_series"]), 12)
                self.assertEqual(len(rec["profile_series"]), 12)
                self.assertEqual(rec["evaluations"], 6 + 12 * 12)
                series = rec["best_series"]
                self.assertTrue(all(b <= a for a, b in zip(series, series[1:])))
                self.assertEqual(rec["objective"], series[-1])
                sol = np.array(rec["solution"])
                self.assertTrue(np.all(sol >= func.lb_vector) and np.all(sol <= func.ub_vector))
                self.assertAlmostEqual(func._evaluate(sol), rec["objective"])
                self.assertIs(rec["feasible"], True)
                self.assertEqual(rec["physical_lcoe"], rec["objective"])
                self.assertEqual(rec["decoded"]["n_el_units"], int(round(sol[1])))

    def test_vanilla_policies_keep_one_profile(self):
        func = FakeFunction()
        for policy, name in (("vanilla_exploitation", "exploit"), ("vanilla_exploration", "explore")):
            rec = run_search(func, monitor_cfg(), policy=policy, pop_size=5, iterations=8, seed=1)
            self.assertEqual(set(rec["profile_series"]), {name})
            self.assertEqual(rec["changes"], [])

    def test_monitor_receives_negated_best_so_far(self):
        seen = []
        original = ProfileController.observe

        def spy(self, best):
            seen.append(best)
            return original(self, best)

        with mock.patch.object(ProfileController, "observe", spy):
            rec = run_search(FakeFunction(), monitor_cfg(), policy="binary_complex",
                             pop_size=5, iterations=6, seed=2)
        self.assertEqual(seen, [-b for b in rec["best_series"]])

    def test_same_seed_reproduces_the_run(self):
        a = run_search(FakeFunction(), monitor_cfg(), policy="continuous_complex",
                       pop_size=5, iterations=6, seed=7)
        b = run_search(FakeFunction(), monitor_cfg(), policy="continuous_complex",
                       pop_size=5, iterations=6, seed=7)
        self.assertEqual(a["best_series"], b["best_series"])
        self.assertEqual(a["solution"], b["solution"])

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            run_search(FakeFunction(), monitor_cfg(), policy="nope", pop_size=5, iterations=2)

    def test_defaults_follow_hres2_paper_monitor(self):
        self.assertEqual((config.POP_SIZE, config.ITERATIONS, config.SEED), (30, 1000, 1))
        cfg = config.monitor_config(use_ddtw=True)
        self.assertEqual((cfg.window, cfg.band, cfg.min_slope, cfg.plateau_max, cfg.patience,
                          cfg.adapt_thresholds, cfg.p_low, cfg.p_high),
                         (40, 4, 0.1, 15, 8, True, 30.0, 70.0))
        self.assertTrue(cfg.use_ddtw)
        self.assertFalse(config.monitor_config(use_ddtw=False).use_ddtw)


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "gwo_de_hres2.run", *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)


class CliTests(unittest.TestCase):
    def test_cli_writes_one_record_per_policy_and_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "smoke")
            proc = run_cli("--policies", "binary_complex,vanilla_exploration",
                           "--pop-size", "4", "--iterations", "2", "--seed", "5",
                           "--signal", "dtw", "--output-dir", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            lines = Path(out, "runs.jsonl").read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]
            self.assertEqual([r["policy"] for r in records],
                             ["binary_complex", "vanilla_exploration"])
            for rec in records:
                self.assertEqual((rec["seed"], rec["signal"], rec["evaluations"]), (5, "dtw", 4 + 2 * 8))
                self.assertIn(rec["feasible"], (True, False))
                self.assertEqual(len(rec["solution"]), 4)
                self.assertIn("battery_duration_h", rec["decoded"])
                self.assertIn(f"{rec['policy']}", proc.stdout)
            manifest = json.loads(Path(out, "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["configuration"]["policies"],
                             ["binary_complex", "vanilla_exploration"])
            self.assertFalse(manifest["monitor"]["use_ddtw"])

    def test_cli_rejects_non_empty_output_dir_and_leaves_it_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "keep.txt").write_text("x", encoding="utf-8")
            proc = run_cli("--iterations", "1", "--pop-size", "4", "--output-dir", tmp)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, "")
            self.assertIn("output directory is not empty", proc.stderr)
            self.assertEqual(os.listdir(tmp), ["keep.txt"])

    def test_cli_rejects_unknown_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "o")
            proc = run_cli("--policies", "bogus", "--output-dir", out)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("unknown policy: bogus", proc.stderr)
            self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
