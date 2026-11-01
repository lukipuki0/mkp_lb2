"""Tests for the four GWO-DE profile policies (S20) and the four-policy CLI (S21).

Spec: odd/tasks/gwo-de-dtw-hybrid.md
"""
from dataclasses import asdict
import json
import math
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
from gwo_de_dtw import EXPLOIT, EXPLORE, ProfileController, run_hybrid  # noqa: E402
from gwo_de_dtw.policies import (  # noqa: E402
    BINARY_POLICIES, CONTINUOUS_POLICIES, POLICIES, blend, intensity,
)

W = 10


def fixed_cfg(window=W, use_ddtw=False):
    # Fixed thresholds: theta_c = 0.1 W, theta_r = 0.5 W, theta_delta = 0.3 W.
    return StagnationConfig(window=window, band=0, min_slope=2.0, plateau_max=4,
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


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


class PolicyNamesTests(unittest.TestCase):
    def test_four_policies_in_order(self):
        self.assertEqual(POLICIES, ("binary_simple", "binary_complex",
                                    "continuous_simple", "continuous_complex"))
        self.assertEqual(BINARY_POLICIES, ("binary_simple", "binary_complex"))
        self.assertEqual(CONTINUOUS_POLICIES, ("continuous_simple", "continuous_complex"))

    def test_config_defaults(self):
        from gwo_de_dtw import config
        self.assertEqual(config.POLICY, "binary_complex")
        self.assertEqual((config.SCALE, config.K, config.CENTER), (2.0, 5.0, 0.5))

    def test_unknown_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "policy"):
            ProfileController(fixed_cfg(), policy="binary_medium")
        with self.assertRaisesRegex(ValueError, "policy"):
            run_hybrid(make_instance(), fixed_cfg(), pop_size=8, iterations=1,
                       policy="binary_medium")


class BlendTests(unittest.TestCase):
    def test_endpoints_are_the_two_profiles(self):
        for value, ref in ((0.0, EXPLOIT), (1.0, EXPLORE)):
            p = blend(EXPLOIT, EXPLORE, value)
            self.assertEqual((p.name, p.a, p.F, p.CR, p.G3),
                             (ref.name, ref.a, ref.F, ref.CR, ref.G3))

    def test_linear_values_and_name_switch_above_half(self):
        p = blend(EXPLOIT, EXPLORE, 0.5)
        self.assertEqual(p.name, "exploit")
        self.assertAlmostEqual(p.a, 1.25)
        self.assertAlmostEqual(p.F, 0.7)
        self.assertAlmostEqual(p.CR, 0.6)
        self.assertAlmostEqual(p.G3, 0.1)
        self.assertEqual(blend(EXPLOIT, EXPLORE, 0.5000001).name, "explore")
        p = blend(EXPLOIT, EXPLORE, 0.8)
        self.assertEqual(p.name, "explore")
        self.assertAlmostEqual(p.a, 0.2 * 0.5 + 0.8 * 2.0)
        self.assertAlmostEqual(p.CR, 0.2 * 0.9 + 0.8 * 0.3)


class BinaryPolicyTests(unittest.TestCase):
    # Flat window whose last generation improved: D2 small, no plateau.
    SERIES = [100.0] * (W - 1) + [100.5]

    def test_binary_simple_enters_on_flat_window_with_recent_improvement(self):
        simple = ProfileController(fixed_cfg(), policy="binary_simple")
        complex_ = ProfileController(fixed_cfg(), policy="binary_complex")
        outs = feed(simple, self.SERIES)
        couts = feed(complex_, self.SERIES)
        last = outs[-1]
        self.assertTrue(last["ready"])
        self.assertEqual(last["no_improve_len"], 0)
        self.assertLessEqual(last["D2_vs_const"], last["theta_c"])
        self.assertTrue(last["changed"])
        self.assertEqual(last["profile"], "explore")
        self.assertIs(simple.profile, EXPLORE)
        self.assertTrue(all(not o["changed"] for o in outs[:-1]))
        self.assertTrue(all(not o["changed"] for o in couts))
        self.assertIs(complex_.profile, EXPLOIT)

    def test_binary_simple_stays_in_exploit_when_window_is_not_flat(self):
        ctrl = ProfileController(fixed_cfg(), policy="binary_simple")
        outs = feed(ctrl, [100.0 + 2 * k for k in range(W + 5)])
        self.assertTrue(all(o["D2_vs_const"] > o["theta_c"] for o in outs if o["ready"]))
        self.assertTrue(all(not o["changed"] for o in outs))
        self.assertIs(ctrl.profile, EXPLOIT)

    def test_binary_simple_resets_monitor_and_returns_with_s8_rule(self):
        ctrl = ProfileController(fixed_cfg(), policy="binary_simple")
        feed(ctrl, self.SERIES)
        improving = [101.0 + k for k in range(W)]
        outs = feed(ctrl, improving)
        self.assertEqual([o["n"] for o in outs], list(range(1, W + 1)))
        self.assertTrue(all(not o["ready"] and not o["changed"] for o in outs[:-1]))
        self.assertTrue(outs[-1]["ready"])
        self.assertLess(outs[-1]["no_improve_len"], 4)
        self.assertTrue(outs[-1]["changed"])
        self.assertIs(ctrl.profile, EXPLOIT)

    def test_binary_simple_stays_in_explore_while_stagnant(self):
        ctrl = ProfileController(fixed_cfg(), policy="binary_simple")
        feed(ctrl, self.SERIES)
        outs = feed(ctrl, [100.5] * (3 * W))
        self.assertTrue(all(not o["changed"] for o in outs))
        self.assertIs(ctrl.profile, EXPLORE)

    def test_binary_outputs_have_no_intensity(self):
        for policy in BINARY_POLICIES:
            ctrl = ProfileController(fixed_cfg(), policy=policy)
            outs = feed(ctrl, [100.0] * (2 * W))
            self.assertTrue(all("intensity" in o and o["intensity"] is None for o in outs))


class ContinuousPolicyTests(unittest.TestCase):
    def test_intensity_is_zero_before_ready(self):
        for policy in CONTINUOUS_POLICIES:
            ctrl = ProfileController(fixed_cfg(), policy=policy)
            outs = feed(ctrl, [100.0] * (W - 1))
            self.assertTrue(all(o["intensity"] == 0.0 and not o["changed"] for o in outs))
            self.assertTrue(all(o["profile"] == "exploit" for o in outs))
            self.assertEqual((ctrl.profile.a, ctrl.profile.F, ctrl.profile.CR),
                             (EXPLOIT.a, EXPLOIT.F, EXPLOIT.CR))

    def test_continuous_simple_intensity_from_d2_and_theta_c(self):
        # W = 100 -> fixed theta_c = 10; one final rise of 4 -> D2 = 4.
        ctrl = ProfileController(fixed_cfg(window=100), policy="continuous_simple")
        out = feed(ctrl, [100.0] * 99 + [104.0])[-1]
        self.assertEqual((out["D2_vs_const"], out["theta_c"]), (4.0, 10.0))
        self.assertAlmostEqual(out["intensity"], 0.8)
        self.assertEqual(out["intensity"], 1.0 - min(max(4.0 / (10.0 * 2.0 + 1e-10), 0.0), 1.0))
        expected = blend(EXPLOIT, EXPLORE, out["intensity"])
        self.assertEqual(ctrl.profile, expected)
        self.assertEqual(out["profile"], "explore")
        self.assertTrue(out["changed"])
        wider = ProfileController(fixed_cfg(window=100), policy="continuous_simple", scale=4.0)
        self.assertAlmostEqual(feed(wider, [100.0] * 99 + [104.0])[-1]["intensity"], 0.9)

    def test_continuous_complex_intensity_is_sigmoid_of_delta_ratio(self):
        ctrl = ProfileController(fixed_cfg(), policy="continuous_complex")
        outs = feed(ctrl, [100.0] * (W + 3))
        for out in outs[W - 1:]:
            ratio = out["delta"] / (out["theta_delta"] + 1e-10)
            self.assertAlmostEqual(out["intensity"], sigmoid(5.0 * (ratio - 0.5)), places=12)
        self.assertGreater(outs[-1]["intensity"], 0.5)
        self.assertEqual(ctrl.profile.name, "explore")
        steep = ProfileController(fixed_cfg(), policy="continuous_complex", k=1.0, center=0.0)
        out = feed(steep, [100.0] * W)[-1]
        self.assertAlmostEqual(out["intensity"],
                               sigmoid(out["delta"] / (out["theta_delta"] + 1e-10)), places=12)

    def test_continuous_complex_is_near_zero_for_negative_delta(self):
        ctrl = ProfileController(fixed_cfg(), policy="continuous_complex")
        out = feed(ctrl, [100.0 + 2 * k for k in range(W)])[-1]
        self.assertLess(out["delta"], 0)
        self.assertGreater(out["theta_delta"], 0)
        self.assertLess(out["intensity"], 0.01)
        self.assertEqual(out["profile"], "exploit")

    def test_negative_theta_delta_is_used_as_is(self):
        out = {"delta": -6.0, "theta_delta": -3.0}
        self.assertAlmostEqual(intensity("continuous_complex", out, 2.0, 5.0, 0.5),
                               sigmoid(5.0 * (-6.0 / (-3.0 + 1e-10) - 0.5)))

    def test_continuous_policies_never_reset_the_monitor(self):
        series = [100.0] * (2 * W) + [100.0 + 3 * k for k in range(1, 3 * W)]
        for policy in CONTINUOUS_POLICIES:
            ctrl = ProfileController(fixed_cfg(), policy=policy)
            outs = feed(ctrl, series)
            self.assertTrue(any(o["changed"] for o in outs), policy)
            self.assertEqual([o["n"] for o in outs], list(range(1, len(series) + 1)))
            self.assertTrue(all(o["ready"] for o in outs[W - 1:]))
            names = ["exploit"] + [o["profile"] for o in outs]
            for i, out in enumerate(outs):
                self.assertEqual(out["changed"], names[i + 1] != names[i])
                self.assertEqual(out["profile"], "explore" if out["intensity"] > 0.5 else "exploit")


class RunHybridPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = make_instance()
        cls.cfg = StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                                   patience=1, use_ddtw=True)

    def run_policy(self, **kwargs):
        return run_hybrid(self.inst, self.cfg, pop_size=8, iterations=60, seed=5, **kwargs)

    def test_default_policy_is_binary_complex(self):
        default = self.run_policy()
        explicit = self.run_policy(policy="binary_complex")
        self.assertEqual(default["policy"], "binary_complex")
        self.assertEqual(explicit["policy"], "binary_complex")
        for key in ("best_series", "profile_series", "parameter_series", "changes",
                    "evaluations", "monitor", "best_fitness", "intensity_series"):
            self.assertEqual(default[key], explicit[key], key)
        self.assertEqual(default["intensity_series"], [None] * 60)

    def test_continuous_run_records_intensity_and_blended_parameters(self):
        for policy in CONTINUOUS_POLICIES:
            res = self.run_policy(policy=policy)
            self.assertEqual(res["policy"], policy)
            series = res["intensity_series"]
            self.assertEqual(len(series), 60)
            self.assertEqual(series[0], 0.0)
            for g in range(59):
                self.assertEqual(series[g + 1], res["monitor"][g]["intensity"])
                self.assertEqual(res["profile_series"][g + 1] != res["profile_series"][g],
                                 g in res["changes"])
            for g, value in enumerate(series):
                p = blend(EXPLOIT, EXPLORE, value)
                self.assertEqual(res["profile_series"][g], p.name)
                for key in ("a", "F", "CR", "G3"):
                    self.assertEqual(res["parameter_series"][key][g], getattr(p, key))

    def test_binary_simple_run_records_policy(self):
        res = self.run_policy(policy="binary_simple")
        self.assertEqual(res["policy"], "binary_simple")
        self.assertEqual(res["intensity_series"], [None] * 60)


def record(values, policy):
    cfg = fixed_cfg(window=4)
    ctrl = ProfileController(cfg, policy=policy)
    profiles, outputs, intensities = [], [], []
    params = {k: [] for k in ("a", "F", "CR", "G3")}
    for value in values:
        profiles.append(ctrl.profile.name)
        intensities.append(ctrl.intensity)
        for key in params:
            params[key].append(getattr(ctrl.profile, key))
        outputs.append(ctrl.observe(value))
    return dict(best_series=values, profile_series=profiles, monitor=outputs,
                parameter_series=params, monitor_config=asdict(cfg), policy=policy,
                intensity_series=intensities, use_ddtw=False, seed=1, window=4)


OLD_SCALARS = ('t', 'f_star', 'D_R', 'D_C', 'delta', 'theta_R', 'theta_C',
               'theta_delta', 's', 'mode_used', 'a', 'F', 'CR', 'ready',
               'adaptive_threshold', 'fire', 'no_improve_len', 'trigger_streak')


class TemporalPolicyTests(unittest.TestCase):
    SERIES = [100.0] * 12 + [100.0 + 3 * k for k in range(1, 20)]

    def test_replay_is_verified_for_every_policy(self):
        from gwo_de_dtw.plots.temporal import build_series
        for policy in POLICIES:
            res = record(self.SERIES, policy)
            self.assertTrue(any(o["changed"] for o in res["monitor"]), policy)
            data = build_series(res)
            ready = data["ready"].astype(bool)
            if policy in CONTINUOUS_POLICIES:
                self.assertTrue(ready[3:].all(), policy)
                np.testing.assert_array_equal(data["intensity"],
                                              np.asarray(res["intensity_series"], dtype=float))
            else:
                self.assertTrue(np.isnan(data["intensity"]).all())

    def test_continuous_export_includes_intensity(self):
        from gwo_de_dtw.plots import save_all_plots
        res = record(self.SERIES, "continuous_simple")
        res.update(binarization="lb2", known_optimum=0.0, generations=len(self.SERIES))
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_all_plots(res, tmp)
            self.assertEqual(len(paths), 4)
            header = (Path(tmp) / "temporal_series.csv").read_text().splitlines()[0].split(",")
            self.assertEqual(header, list(OLD_SCALARS) + ["intensity"])
            with np.load(Path(tmp) / "temporal_series.npz") as data:
                np.testing.assert_array_equal(data["intensity"], res["intensity_series"])
            meta = json.loads((Path(tmp) / "temporal_metadata.json").read_text())
            self.assertEqual(meta["policy"], "continuous_simple")
            self.assertIn("intensity", meta["notes"])

    def test_old_record_without_policy_replays_as_binary(self):
        from gwo_de_dtw.plots.temporal import build_series
        res = record(self.SERIES, "binary_complex")
        del res["policy"], res["intensity_series"]
        data = build_series(res)
        self.assertTrue(np.isnan(data["intensity"]).all())


LB2_EVALS = 8 + 4 * 8 * 25


class RunCliPolicyTests(unittest.TestCase):
    ARGS = ("--instance", str(ROOT / "instancias" / "mknapcb1.txt"), "--pop", "8",
            "--iterations", "25", "--window", "10", "--seed", "3")

    def run_cli(self, *args, cwd):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        return subprocess.run([sys.executable, "-B", "-m", "gwo_de_dtw.run", *args],
                              cwd=cwd, env=env, capture_output=True, text=True, timeout=300)

    def assert_policy_line(self, line, policy):
        self.assertRegex(line, rf"^policy={policy} best=\d+ known_optimum=24381 "
                               rf"gap=\d+\.\d{{2}}% evaluations={LB2_EVALS} "
                               r"generations=25 changes=\d+$")

    def test_runs_four_policies_and_writes_default_group_without_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self.run_cli(*self.ARGS, cwd=tmp)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stderr, "")
            lines = res.stdout.splitlines()
            self.assertEqual(lines[0], "instance=mknapcb1.txt[0] binarization=lb2 "
                                       "signal=ddtw W=10 pop=8 iterations=25 seed=3")
            self.assertEqual(len(lines), 1 + 4 * 5 + 2)
            policy_lines = [line for line in lines if line.startswith("policy=")]
            self.assertEqual(policy_lines, lines[1:21:5])
            for line, policy in zip(policy_lines, POLICIES):
                self.assert_policy_line(line, policy)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["gwo_de_dtw"])
            groups = list((Path(tmp) / "gwo_de_dtw" / "results").iterdir())
            self.assertEqual(len(groups), 1)
            self.assertEqual(sorted(p.name for p in groups[0].iterdir()),
                             sorted(list(POLICIES) + ["convergence_comparison.pdf",
                                                      "convergence_comparison.png"]))

    def test_out_writes_json_and_plots_per_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self.run_cli(*self.ARGS, "--out", str(Path(tmp) / "x.json"), cwd=tmp)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stderr, "")
            lines = res.stdout.splitlines()
            policy_lines = [i for i, line in enumerate(lines) if line.startswith("policy=")]
            self.assertEqual(len(policy_lines), 4)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["x"])
            for i, policy in zip(policy_lines, POLICIES):
                self.assert_policy_line(lines[i], policy)
                folder = Path(tmp) / "x" / policy
                plots = lines[i + 1:i + 5]
                self.assertEqual(len(plots), 4)
                self.assertTrue(all(line.startswith("  [plot] ") and str(folder) in line
                                    for line in plots))
                data = json.loads((folder / "result.json").read_text())
                self.assertEqual(data["policy"], policy)
                self.assertEqual(len(data["intensity_series"]), 25)
                self.assertTrue((folder / "temporal_series.csv").is_file())
                self.assertTrue((folder / "parameters.png").is_file())

    def test_cli_has_no_policy_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self.run_cli("--policy", "binary_simple", cwd=tmp)
        self.assertEqual(res.returncode, 2)
        self.assertEqual(res.stdout, "")
        self.assertIn("unrecognized arguments: --policy binary_simple", res.stderr)


if __name__ == "__main__":
    unittest.main()
