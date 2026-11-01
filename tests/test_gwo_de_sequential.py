"""Tests for the opt-in sequential hybrid mode (hybrid_mode="sequential").

competitive (default): every individual gets one GWO and one DE candidate, the
better of the two is compared with the individual (unchanged behavior).
sequential: one full GWO sweep with greedy >= acceptance, then one full DE
sweep over the population that results from it; DE works in place, so later
DE candidates also see earlier accepted DE rows.

Spec: odd/tasks/gwo-de-dtw-hybrid.md
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtw_stagnation import StagnationConfig  # noqa: E402
from gwo_de_dtw import EXPLOIT, EXPLORE, GWODEHybrid, Profile, config, run_hybrid  # noqa: E402
import gwo_de_dtw.controller as controller_mod  # noqa: E402
import gwo_de_dtw.hybrid as hybrid_mod  # noqa: E402
from gwo_de_dtw.binarization import lb2_binarize, sigmoid_binarize  # noqa: E402
from gwo_de_dtw.de import de_candidate  # noqa: E402
from gwo_de_dtw.gwo import gwo_candidate  # noqa: E402
from gwo_de_dtw.policies import POLICIES  # noqa: E402
from gwo_de_dtw.run import build_parser  # noqa: E402

N = 8
SEQUENTIAL_KEYS = {"gwo_accepted", "de_accepted", "accepted_total"}
COMPETITIVE_KEYS = {"gwo_wins", "de_wins", "accepted"}


def make_instance(n=30, m=5, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.integers(10, 100, n).astype(float)
    r = rng.integers(5, 50, (m, n)).astype(float)
    b = 0.4 * r.sum(axis=1)
    densidad = p / ((r / b[:, None]).mean(axis=0) + 1e-12)
    return {"n": n, "m": m, "p": p, "r": r, "b": b, "optimo": 0.0, "densidad": densidad}


def monitor_cfg():
    return StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                            patience=1, use_ddtw=True)


def reference_competitive_step(h):
    """Copy of the competitive generation before hybrid_mode existed (pre-edit code)."""
    order = np.argsort(h.fit)[::-1]
    leaders = h.x_real[order[:3]].copy()
    gwo_wins = de_wins = accepted = 0
    for i in range(h.pop_size):
        u_gwo = gwo_candidate(h.x_real[i], leaders, h.profile.a, h.rng, h.v_max)
        u_de = de_candidate(i, h.x_real, h.profile.F, h.profile.CR, h.rng, h.v_max)
        if h.binarization == "sigmoid":
            b_gwo, f_gwo = sigmoid_binarize(u_gwo, h.inst, h.rng)
            b_de, f_de = sigmoid_binarize(u_de, h.inst, h.rng)
        else:
            b_gwo, f_gwo = lb2_binarize(u_gwo, h.x_bin[i], h.inst, h.rng,
                                        h.G1, h.G2, h.profile.G3, h.v_max)
            b_de, f_de = lb2_binarize(u_de, h.x_bin[i], h.inst, h.rng,
                                      h.G1, h.G2, h.profile.G3, h.v_max)
        if f_gwo >= f_de:
            u, b, f = u_gwo, b_gwo, f_gwo
            gwo_wins += 1
        else:
            u, b, f = u_de, b_de, f_de
            de_wins += 1
        if f >= h.fit[i]:
            h.x_real[i] = u
            h.x_bin[i] = b
            h.fit[i] = f
            accepted += 1
    idx = int(np.argmax(h.fit))
    if h.fit[idx] > h.best_fitness:
        h.best_fitness = float(h.fit[idx])
        h.best_solution = h.x_bin[idx].copy()
    return {"gwo_wins": gwo_wins, "de_wins": de_wins, "accepted": accepted}


class Recorder:
    """Spies on the operators and LB2 inside hybrid.py, in call order."""

    def __init__(self, h, lb2=None):
        self.h = h
        self.events = []
        self.lb2 = lb2  # optional replacement (v, current, ...) -> (bits, fitness)

    def gwo(self, x, leaders, a, rng, v_max):
        u = gwo_candidate(x, leaders, a, rng, v_max)
        self.events.append({"op": "gwo", "x": x.copy(), "a": a, "u": u.copy()})
        return u

    def de(self, i, population, F, CR, rng, v_max):
        u = de_candidate(i, population, F, CR, rng, v_max)
        self.events.append({"op": "de", "i": i, "pop": population.copy(), "F": F, "CR": CR,
                            "u": u.copy(), "best": self.h.best_fitness,
                            "fit": self.h.fit.copy()})
        return u

    def binarize(self, v, current, inst, rng, G1, G2, G3, v_max):
        if self.lb2 is None:
            bits, fit = lb2_binarize(v, current, inst, rng, G1, G2, G3, v_max)
        else:
            bits, fit = self.lb2(v, current, inst)
        self.events.append({"op": "bin", "current": np.array(current).copy(), "G3": G3,
                            "bits": np.array(bits).copy(), "fit": fit})
        return bits, fit

    def patches(self):
        return (patch.object(hybrid_mod, "gwo_candidate", self.gwo),
                patch.object(hybrid_mod, "de_candidate", self.de),
                patch.object(hybrid_mod, "lb2_binarize", self.binarize))


def run_step(h, recorder):
    p1, p2, p3 = recorder.patches()
    with p1, p2, p3:
        return h.step()


class ModeOptionTests(unittest.TestCase):
    def setUp(self):
        self.inst = make_instance()

    def test_config_default_is_competitive(self):
        self.assertEqual(config.HYBRID_MODE, "competitive")
        self.assertEqual(hybrid_mod.HYBRID_MODES, ("competitive", "sequential"))
        h = GWODEHybrid(self.inst, np.random.default_rng(0), pop_size=N)
        self.assertEqual(h.hybrid_mode, "competitive")

    def test_unknown_mode_is_rejected(self):
        for mode in ("", "Sequential", "gwo_de", None):
            with self.assertRaises(ValueError):
                GWODEHybrid(self.inst, np.random.default_rng(0), pop_size=N, hybrid_mode=mode)
            with self.assertRaises(ValueError):
                run_hybrid(self.inst, monitor_cfg(), pop_size=N, iterations=1, hybrid_mode=mode)


class CompetitiveUnchangedTests(unittest.TestCase):
    """Default and explicit competitive reproduce the pre-edit generation exactly."""

    def test_matches_pre_edit_reference(self):
        inst = make_instance()
        for binarization in ("lb2", "sigmoid"):
            for kwargs in ({}, {"hybrid_mode": "competitive"}):
                h = GWODEHybrid(inst, np.random.default_rng(11), pop_size=N,
                                binarization=binarization, **kwargs)
                ref = GWODEHybrid(inst, np.random.default_rng(11), pop_size=N,
                                  binarization=binarization)
                h.initialize()
                ref.initialize()
                for g in range(6):
                    profile = EXPLORE if g % 2 else EXPLOIT
                    h.set_profile(profile)
                    ref.set_profile(profile)
                    best = h.step()
                    stats = reference_competitive_step(ref)
                    self.assertEqual(h.last_step_stats, stats, (binarization, g))
                    self.assertEqual(best, ref.best_fitness)
                    np.testing.assert_array_equal(h.x_real, ref.x_real)
                    np.testing.assert_array_equal(h.x_bin, ref.x_bin)
                    np.testing.assert_array_equal(h.fit, ref.fit)
                    np.testing.assert_array_equal(h.best_solution, ref.best_solution)
                    self.assertEqual(h.rng.random(), ref.rng.random())

    def test_run_hybrid_default_equals_competitive_and_records_mode(self):
        inst = make_instance()
        for policy in POLICIES:
            default = run_hybrid(inst, monitor_cfg(), pop_size=N, iterations=30, seed=5,
                                 policy=policy)
            explicit = run_hybrid(inst, monitor_cfg(), pop_size=N, iterations=30, seed=5,
                                  policy=policy, hybrid_mode="competitive")
            self.assertEqual(default["hybrid_mode"], "competitive")
            self.assertEqual(default.keys(), explicit.keys())
            for key in default:
                if key != "best_solution":
                    self.assertEqual(default[key], explicit[key], (policy, key))
            np.testing.assert_array_equal(default["best_solution"], explicit["best_solution"])
            self.assertTrue(all(set(s) == COMPETITIVE_KEYS for s in default["step_stats"]))


class SequentialStepTests(unittest.TestCase):
    def setUp(self):
        self.inst = make_instance()

    def make(self, seed=1, **kwargs):
        h = GWODEHybrid(self.inst, np.random.default_rng(seed), pop_size=N,
                        hybrid_mode="sequential", **kwargs)
        h.initialize()
        return h

    def test_all_gwo_calls_precede_all_de_calls(self):
        h = self.make()
        for _ in range(3):
            rec = Recorder(h)
            run_step(h, rec)
            ops = [e["op"] for e in rec.events]
            self.assertEqual(ops, ["gwo", "bin"] * N + ["de", "bin"] * N)
            self.assertEqual([e["i"] for e in rec.events if e["op"] == "de"], list(range(N)))

    def test_de_reads_population_after_gwo_acceptances_and_works_in_place(self):
        h = self.make(seed=4)
        for g in range(4):
            x_real, x_bin, fit = h.x_real.copy(), h.x_bin.copy(), h.fit.copy()
            rec = Recorder(h)
            run_step(h, rec)
            gwo = rec.events[:2 * N]
            de = rec.events[2 * N:]
            gwo_accepted = de_accepted = 0
            # GWO phase: every GWO call sees the input row; accepted rows are written.
            for i in range(N):
                call, binz = gwo[2 * i], gwo[2 * i + 1]
                np.testing.assert_array_equal(call["x"], x_real[i])
                np.testing.assert_array_equal(binz["current"], x_bin[i])
                np.testing.assert_array_equal(
                    call["u"], np.clip(call["u"], -h.v_max, h.v_max))
                if binz["fit"] >= fit[i]:
                    x_real[i], x_bin[i], fit[i] = call["u"], binz["bits"], binz["fit"]
                    gwo_accepted += 1
            # DE phase: population and current bits after GWO and earlier DE rows.
            for i in range(N):
                call, binz = de[2 * i], de[2 * i + 1]
                np.testing.assert_array_equal(call["pop"], x_real, (g, i))
                np.testing.assert_array_equal(binz["current"], x_bin[i], (g, i))
                np.testing.assert_array_equal(call["fit"], fit, (g, i))
                if binz["fit"] >= fit[i]:
                    x_real[i], x_bin[i], fit[i] = call["u"], binz["bits"], binz["fit"]
                    de_accepted += 1
            np.testing.assert_array_equal(h.x_real, x_real)
            np.testing.assert_array_equal(h.x_bin, x_bin)
            np.testing.assert_array_equal(h.fit, fit)
            self.assertEqual(h.last_step_stats, {"gwo_accepted": gwo_accepted,
                                                 "de_accepted": de_accepted,
                                                 "accepted_total": gwo_accepted + de_accepted})

    def test_gwo_leaders_are_frozen_at_generation_start(self):
        h = self.make(seed=2)
        leaders = h.x_real[np.argsort(h.fit)[::-1][:3]].copy()
        seen = []

        def gwo(x, lead, a, rng, v_max):
            seen.append(lead.copy())
            return gwo_candidate(x, lead, a, rng, v_max)

        # Every GWO candidate accepted (equal fitness) would move a leader row
        # if the leaders were not copied.
        rec = Recorder(h, lb2=lambda v, cur, inst: (np.array(cur), float(inst["p"] @ cur)))
        p1, p2, p3 = rec.patches()
        with p2, p3, patch.object(hybrid_mod, "gwo_candidate", gwo):
            h.step()
        self.assertEqual(len(seen), N)
        for lead in seen:
            np.testing.assert_array_equal(lead, leaders)

    def test_rejected_proposals_leave_input_for_de(self):
        h = self.make(seed=3)
        x_real, x_bin, fit = h.x_real.copy(), h.x_bin.copy(), h.fit.copy()
        best = h.best_fitness
        rec = Recorder(h, lb2=lambda v, cur, inst: (np.array(cur), -1.0))
        run_step(h, rec)
        for e in rec.events:
            if e["op"] == "de":
                np.testing.assert_array_equal(e["pop"], x_real)
                np.testing.assert_array_equal(e["fit"], fit)
        de_bins = [e for e in rec.events[2 * N:] if e["op"] == "bin"]
        for i, e in enumerate(de_bins):
            np.testing.assert_array_equal(e["current"], x_bin[i])
        np.testing.assert_array_equal(h.x_real, x_real)
        np.testing.assert_array_equal(h.x_bin, x_bin)
        np.testing.assert_array_equal(h.fit, fit)
        self.assertEqual(h.best_fitness, best)
        self.assertEqual(h.last_step_stats,
                         {"gwo_accepted": 0, "de_accepted": 0, "accepted_total": 0})

    def test_equal_fitness_is_accepted_in_both_phases(self):
        h = self.make(seed=6)
        fit = h.fit.copy()
        # Same bits, same fitness: a tie in both phases, so both write the real vector.
        rec = Recorder(h, lb2=lambda v, cur, inst: (np.array(cur), float(inst["p"] @ cur)))
        run_step(h, rec)
        gwo_u = np.array([e["u"] for e in rec.events if e["op"] == "gwo"])
        de_calls = [e for e in rec.events if e["op"] == "de"]
        np.testing.assert_array_equal(de_calls[0]["pop"], gwo_u)
        np.testing.assert_array_equal(h.x_real, np.array([e["u"] for e in de_calls]))
        np.testing.assert_array_equal(h.fit, fit)
        self.assertEqual(h.last_step_stats,
                         {"gwo_accepted": N, "de_accepted": N, "accepted_total": 2 * N})

    def test_evaluation_cost_is_unchanged(self):
        for binarization, per_generation in (("lb2", 4 * N), ("sigmoid", 2 * N)):
            h = self.make(binarization=binarization)
            self.assertEqual(h.evaluations, N)
            for _ in range(4):
                before = h.evaluations
                h.step()
                self.assertEqual(h.evaluations - before, per_generation, binarization)
            limit = config.V_MAX if binarization == "lb2" else config.SIGMOID_V_MAX
            self.assertTrue(np.all(np.abs(h.x_real) <= limit))

    def test_best_is_recomputed_after_each_phase_and_never_worsens(self):
        h = self.make(seed=8)
        prev_best, prev_fit = h.best_fitness, h.fit.copy()
        improved_after_gwo = False
        for _ in range(8):
            rec = Recorder(h)
            best = run_step(h, rec)
            first_de = next(e for e in rec.events if e["op"] == "de")
            # Best seen by the DE phase = best of the population after GWO.
            self.assertEqual(first_de["best"], max(prev_best, float(first_de["fit"].max())))
            improved_after_gwo |= first_de["best"] > prev_best
            self.assertGreaterEqual(first_de["best"], prev_best)
            self.assertGreaterEqual(best, first_de["best"])
            self.assertTrue(np.all(first_de["fit"] >= prev_fit))
            self.assertTrue(np.all(h.fit >= first_de["fit"]))
            self.assertEqual(best, float(h.fit.max()))
            self.assertEqual(h.best_fitness, float(self.inst["p"] @ h.best_solution))
            prev_best, prev_fit = best, h.fit.copy()
        self.assertTrue(improved_after_gwo)


class SequentialRunTests(unittest.TestCase):
    def setUp(self):
        self.inst = make_instance()

    def test_record_mode_cost_and_stats(self):
        for binarization, per_generation in (("lb2", 4 * N), ("sigmoid", 2 * N)):
            res = run_hybrid(self.inst, monitor_cfg(), pop_size=N, iterations=20, seed=5,
                             binarization=binarization, hybrid_mode="sequential")
            self.assertEqual(res["hybrid_mode"], "sequential")
            self.assertEqual(res["evaluations"], N + per_generation * 20)
            self.assertEqual(res["generations"], 20)
            series = res["best_series"]
            self.assertTrue(all(b >= a for a, b in zip(series, series[1:])))
            self.assertEqual(len(res["step_stats"]), 20)
            for stats in res["step_stats"]:
                self.assertEqual(set(stats), SEQUENTIAL_KEYS)
                self.assertEqual(stats["accepted_total"],
                                 stats["gwo_accepted"] + stats["de_accepted"])
                self.assertLessEqual(stats["gwo_accepted"], N)
                self.assertLessEqual(stats["de_accepted"], N)

    def test_monitor_observes_once_after_full_generation_with_one_profile(self):
        exploit = Profile("exploit", EXPLOIT.a, EXPLOIT.F, EXPLOIT.CR, 0.15)
        explore = Profile("explore", EXPLORE.a, EXPLORE.F, EXPLORE.CR, 0.35)
        observed = []
        real_observe = controller_mod.ProfileController.observe
        h_ref = {}
        rec = Recorder(None)
        real_init = GWODEHybrid.__init__

        def init(self_h, *args, **kwargs):
            real_init(self_h, *args, **kwargs)
            h_ref["h"] = self_h
            rec.h = self_h

        def observe(self_c, best_fitness):
            observed.append((len(rec.events), best_fitness, h_ref["h"].best_fitness))
            out = dict(real_observe(self_c, best_fitness))
            # Alternate profiles every generation (applies from the next one).
            self_c.profile = (self_c.explore if self_c.profile is self_c.exploit
                              else self_c.exploit)
            out["changed"] = True
            out["profile"] = self_c.profile.name
            return out

        p1, p2, p3 = rec.patches()
        with p1, p2, p3, patch.object(GWODEHybrid, "__init__", init), \
                patch.object(controller_mod.ProfileController, "observe", observe):
            res = run_hybrid(self.inst, monitor_cfg(), pop_size=N, iterations=4, seed=5,
                             g3_exploit=0.15, g3_explore=0.35, hybrid_mode="sequential")
        # Recorded events per generation: N x (gwo, bin) + N x (de, bin).
        per_generation = 4 * N
        self.assertEqual([o[0] for o in observed],
                         [per_generation * (g + 1) for g in range(4)])
        self.assertEqual([o[1] for o in observed], res["best_series"])
        self.assertTrue(all(o[1] == o[2] for o in observed))
        self.assertEqual(res["profile_series"], ["exploit", "explore", "exploit", "explore"])
        for g, name in enumerate(res["profile_series"]):
            profile = exploit if name == "exploit" else explore
            events = rec.events[g * per_generation:(g + 1) * per_generation]
            self.assertEqual([e["a"] for e in events if e["op"] == "gwo"], [profile.a] * N)
            self.assertEqual([(e["F"], e["CR"]) for e in events if e["op"] == "de"],
                             [(profile.F, profile.CR)] * N)
            self.assertEqual([e["G3"] for e in events if e["op"] == "bin"],
                             [profile.G3] * (2 * N))

    def test_sequential_changes_only_the_generation_scheme(self):
        a = run_hybrid(self.inst, monitor_cfg(), pop_size=N, iterations=30, seed=5)
        b = run_hybrid(self.inst, monitor_cfg(), pop_size=N, iterations=30, seed=5,
                       hybrid_mode="sequential")
        self.assertEqual(a.keys(), b.keys())
        same = ("policy", "dtw_kirito", "monitor_config", "use_ddtw", "window", "seed",
                "binarization", "G", "pop_size", "iterations", "evaluations", "generations",
                "known_optimum")
        for key in same:
            self.assertEqual(a[key], b[key], key)
        self.assertNotEqual(a["step_stats"], b["step_stats"])


class CliModeTests(unittest.TestCase):
    ARGS = ("--instance", str(ROOT / "instancias" / "mknapcb1.txt"), "--pop", "8",
            "--iterations", "25", "--window", "10", "--seed", "3")

    def run_cli(self, *args, cwd):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        return subprocess.run([sys.executable, "-B", "-m", "gwo_de_dtw.run", *args],
                              cwd=cwd, env=env, capture_output=True, text=True, timeout=300)

    def test_parser_option(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).hybrid_mode, "competitive")
        self.assertEqual(parser.parse_args(["--hybrid-mode", "sequential"]).hybrid_mode,
                         "sequential")
        self.assertIn("--hybrid-mode {competitive,sequential}", parser.format_help())

    def test_invalid_mode_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self.run_cli(*self.ARGS, "--hybrid-mode", "gwo_de",
                               "--out", str(Path(tmp) / "x.json"), cwd=tmp)
            self.assertEqual(res.returncode, 2)
            self.assertEqual(res.stdout, "")
            self.assertIn("argument --hybrid-mode: invalid choice: 'gwo_de'", res.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_sequential_run_writes_four_records_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self.run_cli(*self.ARGS, "--hybrid-mode", "sequential",
                               "--out", str(Path(tmp) / "x.json"), cwd=tmp)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stderr, "")
            lines = res.stdout.splitlines()
            self.assertEqual(lines[0], "instance=mknapcb1.txt[0] binarization=lb2 "
                                       "signal=ddtw W=10 pop=8 iterations=25 seed=3 "
                                       "hybrid_mode=sequential")
            group = Path(tmp) / "x"
            for policy in POLICIES:
                data = json.loads((group / policy / "result.json").read_text())
                self.assertEqual(data["hybrid_mode"], "sequential")
                self.assertEqual(data["policy"], policy)
                self.assertEqual(data["evaluations"], 8 + 4 * 8 * 25)
                self.assertTrue(all(set(s) == SEQUENTIAL_KEYS for s in data["step_stats"]))
                for name in ("fitness_convergence.png", "dtw_delta.png", "parameters.png",
                             "temporal_diagnostics.png", "temporal_series.csv"):
                    self.assertTrue((group / policy / name).is_file(), (policy, name))
            self.assertTrue((group / "convergence_comparison.png").is_file())
            self.assertEqual(lines[-1], f"output={group}")


if __name__ == "__main__":
    unittest.main()
