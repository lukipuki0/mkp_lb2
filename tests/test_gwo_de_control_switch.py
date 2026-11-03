"""Tests for the dtw_kirito switch of the binary policies (Kirito rolling control).

dtw_kirito=False (default) keeps the fresh-window control; dtw_kirito=True keeps one
monitor and, on every ready observation, uses explore iff the policy trigger
holds and exploit otherwise. Continuous policies ignore the switch.

Spec: odd/tasks/gwo-de-dtw-hybrid.md
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtw_stagnation import StagnationConfig  # noqa: E402
from gwo_de_dtw import EXPLOIT, EXPLORE, ProfileController, config, run_hybrid  # noqa: E402
from gwo_de_dtw.policies import (  # noqa: E402
    BINARY_POLICIES, CONTINUOUS_POLICIES, POLICIES, enters_explore,
)
from gwo_de_dtw.run import build_parser  # noqa: E402

W = 10


def fixed_cfg(use_ddtw=False):
    # Fixed thresholds: theta_c = 0.1 W = 1, theta_r = 0.5 W, theta_delta = 0.3 W.
    return StagnationConfig(window=W, band=0, min_slope=2.0, plateau_max=4,
                            patience=2, use_ddtw=use_ddtw, adapt_thresholds=False)


def feed(ctrl, values):
    return [ctrl.observe(v) for v in values]


def make_instance(n=30, m=5, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.integers(10, 100, n).astype(float)
    r = rng.integers(5, 50, (m, n)).astype(float)
    b = 0.4 * r.sum(axis=1)
    densidad = p / ((r / b[:, None]).mean(axis=0) + 1e-12)
    return {"n": n, "m": m, "p": p, "r": r, "b": b, "optimo": 0.0, "densidad": densidad}


# Flat window (explore at observation W), one small improvement, then no
# improvement. With DTW, D2_vs_const = 0.25 * k: <= theta_c = 1 up to k = 4,
# 1.25 at observation W + 5 where no_improve_len = 4 = plateau_max.
SIMPLE_SERIES = [100.0] * W + [100.25] * 6
# Mixed series: flat, rising, flat again.
MIXED_SERIES = [100.0] * 15 + [100.0 + 3 * k for k in range(1, 12)] + [133.0] * 20


class ConfigTests(unittest.TestCase):
    def test_dtw_kirito_defaults_to_false(self):
        self.assertIs(config.DTW_KIRITO, False)
        self.assertFalse(ProfileController(fixed_cfg()).dtw_kirito)


class FreshWindowDefaultTests(unittest.TestCase):
    def test_false_equals_default_and_resets_after_every_change(self):
        for policy in POLICIES:
            default = feed(ProfileController(fixed_cfg(), policy=policy), MIXED_SERIES)
            fresh = feed(ProfileController(fixed_cfg(), policy=policy, dtw_kirito=False),
                         MIXED_SERIES)
            self.assertEqual(default, fresh, policy)
        outs = feed(ProfileController(fixed_cfg(), policy="binary_simple", dtw_kirito=False),
                    SIMPLE_SERIES)
        self.assertTrue(outs[W - 1]["changed"])
        self.assertEqual([o["n"] for o in outs[W:]], [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(o["profile"] == "explore" for o in outs[W - 1:]))

    def test_false_returns_on_improvement_after_fresh_window(self):
        ctrl = ProfileController(fixed_cfg(), policy="binary_simple", dtw_kirito=False)
        feed(ctrl, [100.0] * (W - 1) + [100.5])
        self.assertIs(ctrl.profile, EXPLORE)
        outs = feed(ctrl, [101.0 + k for k in range(W)])
        self.assertTrue(all(not o["changed"] for o in outs[:-1]))
        self.assertTrue(outs[-1]["changed"])
        self.assertIs(ctrl.profile, EXPLOIT)


class DtwKiritoBinaryTests(unittest.TestCase):
    def test_simple_keeps_monitor_and_returns_without_improvement(self):
        ctrl = ProfileController(fixed_cfg(), policy="binary_simple", dtw_kirito=True)
        outs = feed(ctrl, SIMPLE_SERIES)
        self.assertEqual([o["n"] for o in outs], list(range(1, len(SIMPLE_SERIES) + 1)))
        self.assertEqual([o["D2_vs_const"] for o in outs[W - 1:]],
                         [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
        self.assertEqual([o["profile"] for o in outs[W - 1:]],
                         ["explore"] * 5 + ["exploit"] * 2)
        self.assertEqual([o["changed"] for o in outs[W - 1:]],
                         [True, False, False, False, False, True, False])
        # Improvement at observation W + 1 does not return to exploit ...
        self.assertEqual(outs[W]["no_improve_len"], 0)
        self.assertEqual(outs[W]["profile"], "explore")
        # ... and the return at W + 5 happens on a plateau (fresh rule would stay).
        self.assertGreaterEqual(outs[W + 4]["no_improve_len"], ctrl.cfg.plateau_max)
        self.assertIs(ctrl.profile, EXPLOIT)
        self.assertTrue(all(o["intensity"] is None for o in outs))

    def test_complex_follows_fire_on_every_ready_observation(self):
        ctrl = ProfileController(fixed_cfg(), policy="binary_complex", dtw_kirito=True)
        outs = feed(ctrl, [100.0] * (W + 3) + [100.25])
        self.assertEqual([o["fire"] for o in outs[W - 1:]], [False, True, True, True, False])
        self.assertEqual([o["profile"] for o in outs[W - 1:]],
                         ["exploit", "explore", "explore", "explore", "exploit"])
        self.assertEqual(outs[-1]["n"], W + 4)

    def test_profile_is_explore_iff_trigger_on_ready_observations(self):
        for policy in BINARY_POLICIES:
            for use_ddtw in (False, True):
                ctrl = ProfileController(fixed_cfg(use_ddtw), policy=policy, dtw_kirito=True)
                outs = feed(ctrl, MIXED_SERIES)
                self.assertEqual([o["n"] for o in outs],
                                 list(range(1, len(MIXED_SERIES) + 1)))
                names = ["exploit"]
                for out in outs:
                    if out["ready"]:
                        want = "explore" if enters_explore(policy, out) else "exploit"
                    else:
                        want = names[-1]
                    self.assertEqual(out["profile"], want)
                    self.assertEqual(out["changed"], want != names[-1])
                    names.append(want)
                self.assertIn("explore", names, (policy, use_ddtw))

    def test_continuous_policies_ignore_the_switch(self):
        for policy in CONTINUOUS_POLICIES:
            off = feed(ProfileController(fixed_cfg(), policy=policy), MIXED_SERIES)
            on = feed(ProfileController(fixed_cfg(), policy=policy, dtw_kirito=True),
                      MIXED_SERIES)
            self.assertEqual(off, on, policy)


class RunHybridSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = make_instance()
        cls.cfg = StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                                   patience=1, use_ddtw=True)

    def run_policy(self, **kwargs):
        return run_hybrid(self.inst, self.cfg, pop_size=8, iterations=60, seed=5, **kwargs)

    def test_record_stores_flag_and_false_equals_default(self):
        for policy in POLICIES:
            default = self.run_policy(policy=policy)
            fresh = self.run_policy(policy=policy, dtw_kirito=False)
            self.assertIs(default["dtw_kirito"], False)
            self.assertNotIn("dtw_main", default)
            self.assertEqual(default.keys(), fresh.keys())
            for key in default:
                if key != "best_solution":
                    self.assertEqual(default[key], fresh[key], (policy, key))
            np.testing.assert_array_equal(default["best_solution"], fresh["best_solution"])

    def test_continuous_run_is_unchanged_except_flag(self):
        for policy in CONTINUOUS_POLICIES:
            off = self.run_policy(policy=policy)
            on = self.run_policy(policy=policy, dtw_kirito=True)
            self.assertIs(on["dtw_kirito"], True)
            for key in off:
                if key not in ("dtw_kirito", "best_solution"):
                    self.assertEqual(off[key], on[key], (policy, key))
            np.testing.assert_array_equal(off["best_solution"], on["best_solution"])

    def test_binary_run_never_resets_and_follows_trigger(self):
        for policy in BINARY_POLICIES:
            res = self.run_policy(policy=policy, dtw_kirito=True)
            self.assertIs(res["dtw_kirito"], True)
            monitor = res["monitor"]
            self.assertEqual([o["n"] for o in monitor], list(range(1, 61)))
            for g in range(59):
                out = monitor[g]
                if out["ready"]:
                    want = "explore" if enters_explore(policy, out) else "exploit"
                else:
                    want = res["profile_series"][g]
                self.assertEqual(res["profile_series"][g + 1], want, (policy, g))
                self.assertEqual(out["changed"], g in res["changes"])
            self.assertEqual(res["intensity_series"], [None] * 60)


def record(policy, dtw_kirito, values=MIXED_SERIES):
    cfg = fixed_cfg()
    ctrl = ProfileController(cfg, policy=policy, dtw_kirito=dtw_kirito)
    profiles, outputs = [], []
    params = {k: [] for k in ("a", "F", "CR", "G3")}
    for value in values:
        profiles.append(ctrl.profile.name)
        for key in params:
            params[key].append(getattr(ctrl.profile, key))
        outputs.append(ctrl.observe(value))
    return dict(best_series=values, profile_series=profiles, monitor=outputs,
                parameter_series=params, monitor_config=asdict(cfg), policy=policy,
                intensity_series=[None] * len(values), dtw_kirito=dtw_kirito)


class TemporalReplayTests(unittest.TestCase):
    def test_dtw_kirito_record_replays_without_resets(self):
        from gwo_de_dtw.plots.temporal import build_series
        for policy in BINARY_POLICIES:
            res = record(policy, True)
            self.assertGreater(sum(o["changed"] for o in res["monitor"]), 1, policy)
            data = build_series(res)
            self.assertTrue(data["ready"][W - 1:].astype(bool).all(), policy)

    def test_record_without_flag_still_replays_with_resets(self):
        from gwo_de_dtw.plots.temporal import build_series
        res = record("binary_simple", False)
        del res["dtw_kirito"]
        self.assertTrue(any(o["changed"] for o in res["monitor"]))
        data = build_series(res)
        first = next(i for i, o in enumerate(res["monitor"]) if o["changed"])
        self.assertFalse(data["ready"][first + 1])
        res["dtw_kirito"] = True
        with self.assertRaisesRegex(ValueError, "Monitor mismatch"):
            build_series(res)

    def test_historical_dtw_main_key_replays_as_dtw_kirito(self):
        from gwo_de_dtw.plots.temporal import build_series
        for policy in BINARY_POLICIES:
            res = record(policy, True)
            res["dtw_main"] = res.pop("dtw_kirito")
            data = build_series(res)
            self.assertTrue(data["ready"][W - 1:].astype(bool).all(), policy)
            # The new key wins when both are present.
            res["dtw_kirito"] = False
            with self.assertRaisesRegex(ValueError, "Monitor mismatch"):
                build_series(res)


class CliSwitchTests(unittest.TestCase):
    ARGS = ("--instance", str(ROOT / "instancias" / "mknapcb1.txt"), "--pop", "8",
            "--iterations", "25", "--window", "10", "--seed", "3")

    def run_cli(self, *args, cwd):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        return subprocess.run([sys.executable, "-B", "-m", "gwo_de_dtw.run", *args],
                              cwd=cwd, env=env, capture_output=True, text=True, timeout=300)

    def test_parser_switch(self):
        parser = build_parser()
        self.assertIs(parser.parse_args([]).dtw_kirito, False)
        self.assertIs(parser.parse_args(["--dtw-kirito"]).dtw_kirito, True)
        self.assertIs(parser.parse_args(["--dtw-kirito", "--no-dtw-kirito"]).dtw_kirito, False)
        action = next(a for a in parser._actions if a.dest == "dtw_kirito")
        self.assertIsInstance(action, argparse.BooleanOptionalAction)
        help_text = parser.format_help()
        self.assertIn("--dtw-kirito, --no-dtw-kirito", help_text)
        self.assertIn("Kirito rolling control", action.help)
        self.assertNotIn("--dtw-main", help_text)
        self.assertNotIn("DTW_optimization-main", help_text)

    def test_dtw_kirito_run_prints_flag_and_records_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self.run_cli(*self.ARGS, "--dtw-kirito", "--out", str(Path(tmp) / "x.json"),
                               cwd=tmp)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stderr, "")
            lines = res.stdout.splitlines()
            self.assertEqual(lines[0], "instance=mknapcb1.txt[0] binarization=lb2 "
                                       "signal=ddtw W=10 pop=8 iterations=25 seed=3 "
                                       "dtw_kirito=True")
            self.assertEqual(len(lines), 1 + 4 * 5 + 2)
            for i, policy in enumerate(POLICIES):
                block = lines[1 + 5 * i:6 + 5 * i]
                self.assertTrue(block[0].startswith(f"policy={policy} best="))
                self.assertTrue(all(line.startswith("  [plot] ") for line in block[1:]))
                folder = Path(tmp) / "x" / policy
                data = json.loads((folder / "result.json").read_text())
                self.assertIs(data["dtw_kirito"], True)
                self.assertNotIn("dtw_main", data)
                for name in ("temporal_diagnostics.png", "temporal_series.csv",
                             "parameters.png", "fitness_convergence.png", "dtw_delta.png"):
                    self.assertTrue((folder / name).is_file(), (policy, name))

    def test_default_run_header_unchanged_and_records_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self.run_cli(*self.ARGS, "--out", str(Path(tmp) / "x.json"), cwd=tmp)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stderr, "")
            self.assertEqual(res.stdout.splitlines()[0],
                             "instance=mknapcb1.txt[0] binarization=lb2 "
                             "signal=ddtw W=10 pop=8 iterations=25 seed=3")
            for policy in POLICIES:
                data = json.loads((Path(tmp) / "x" / policy / "result.json").read_text())
                self.assertIs(data["dtw_kirito"], False)
                self.assertEqual(data["hybrid_mode"], "competitive")
                self.assertTrue(all(set(s) == {"gwo_wins", "de_wins", "accepted"}
                                    for s in data["step_stats"]))


if __name__ == "__main__":
    unittest.main()
