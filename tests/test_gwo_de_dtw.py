"""Tests for the GWO-DE hybrid with DTW/DDTW fresh-window control.

Spec: odd/tasks/gwo-de-dtw-hybrid.md
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from gwo_de_dtw import (  # noqa: E402
    EXPLOIT, EXPLORE, GWODEHybrid, Profile, ProfileController, run_hybrid,
)
from gwo_de_dtw.binarization import (  # noqa: E402
    SIGMOID_EVALUATIONS, lb2_binarize, lb2_probabilities, sigmoid_binarize,
)
from gwo_de_dtw.de import de_candidate  # noqa: E402
from gwo_de_dtw.gwo import gwo_candidate  # noqa: E402
from dtw_stagnation import StagnationConfig  # noqa: E402
from gwo_de_dtw.problem import reparar  # noqa: E402


def make_instance(n=25, m=4, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.integers(10, 100, n).astype(float)
    r = rng.integers(5, 50, (m, n)).astype(float)
    b = 0.4 * r.sum(axis=1)
    densidad = p / ((r / b[:, None]).mean(axis=0) + 1e-12)
    return {"n": n, "m": m, "p": p, "r": r, "b": b,
            "optimo": 0.0, "densidad": densidad}


def feasible(inst, x):
    return bool(np.all(inst["r"] @ x <= inst["b"]))


class HybridPopulationTests(unittest.TestCase):
    def setUp(self):
        self.inst = make_instance()
        self.N = 10

    def make(self, seed=1):
        return GWODEHybrid(self.inst, np.random.default_rng(seed), pop_size=self.N)

    # S1, S5
    def test_initialization_costs_n_evaluations_and_is_feasible(self):
        h = self.make()
        h.initialize()
        self.assertEqual(h.evaluations, self.N)
        self.assertEqual(h.x_real.shape, (self.N, self.inst["n"]))
        self.assertTrue(np.all(np.abs(h.x_real) <= 100.0))
        for x, f in zip(h.x_bin, h.fit):
            self.assertTrue(feasible(self.inst, x))
            self.assertAlmostEqual(f, float(self.inst["p"] @ x))
        self.assertEqual(h.best_fitness, float(np.max(h.fit)))

    # S2, S5
    def test_each_generation_costs_four_n_and_keeps_population_valid(self):
        h = self.make()
        h.initialize()
        prev_fit = h.fit.copy()
        prev_best = h.best_fitness
        for g in range(5):
            before = h.evaluations
            best = h.step()
            self.assertEqual(h.evaluations - before, 4 * self.N)
            self.assertTrue(np.all(h.fit >= prev_fit))       # greedy >= acceptance
            self.assertGreaterEqual(best, prev_best)         # best never worsens
            self.assertTrue(np.all(np.abs(h.x_real) <= 100.0))
            for x, f in zip(h.x_bin, h.fit):
                self.assertTrue(feasible(self.inst, x))
                self.assertAlmostEqual(f, float(self.inst["p"] @ x))
            stats = h.last_step_stats
            self.assertEqual(stats["gwo_wins"] + stats["de_wins"], self.N)
            self.assertLessEqual(stats["accepted"], self.N)
            prev_fit, prev_best = h.fit.copy(), best
        self.assertEqual(h.best_fitness, float(self.inst["p"] @ h.best_solution))

    # S5: the stop is by iterations, there is no evaluation budget
    def test_max_evaluations_is_not_accepted(self):
        with self.assertRaises(TypeError):
            GWODEHybrid(self.inst, np.random.default_rng(0), pop_size=self.N,
                        max_evaluations=100)
        self.assertFalse(hasattr(self.make(), "can_step"))

    # S12: defaults come from config.py
    def test_defaults_come_from_config(self):
        from gwo_de_dtw import config
        h = GWODEHybrid(self.inst, np.random.default_rng(0))
        self.assertEqual(h.pop_size, config.POP_SIZE)
        self.assertEqual(h.v_max, config.V_MAX)
        self.assertEqual((h.G1, h.G2), (config.G1, config.G2))
        self.assertFalse(hasattr(h, "G3"))
        self.assertEqual(h.profile.G3, config.EXPLOIT_G3)

    # S11: G2 >= v_max flips the sign of the LB2 slope
    def test_g2_not_below_v_max_is_rejected(self):
        for g2 in (100.0, 120.0):
            with self.assertRaises(ValueError):
                GWODEHybrid(self.inst, np.random.default_rng(0), pop_size=self.N, G2=g2)

    # S14: only lb2 and sigmoid exist
    def test_unknown_binarization_is_rejected(self):
        for name in ("foo", "LB2", ""):
            with self.assertRaises(ValueError):
                GWODEHybrid(self.inst, np.random.default_rng(0), pop_size=self.N,
                            binarization=name)

    # S14: default binarization comes from config.py (lb2)
    def test_default_binarization_is_lb2_from_config(self):
        from gwo_de_dtw import config
        h = GWODEHybrid(self.inst, np.random.default_rng(0), pop_size=self.N)
        self.assertEqual(h.binarization, config.BINARIZATION)
        self.assertEqual(h.binarization, "lb2")

    def test_population_smaller_than_four_is_rejected(self):
        with self.assertRaises(ValueError):
            GWODEHybrid(self.inst, np.random.default_rng(0), pop_size=3)

    # S4
    def test_profiles_set_gwo_and_de_parameters_together(self):
        self.assertEqual((EXPLOIT.a, EXPLOIT.F, EXPLOIT.CR), (0.5, 0.5, 0.9))
        self.assertEqual((EXPLORE.a, EXPLORE.F, EXPLORE.CR), (2.0, 0.9, 0.3))
        h = self.make()
        self.assertIs(h.profile, EXPLOIT)
        h.set_profile(EXPLORE)
        self.assertIs(h.profile, EXPLORE)

    # S4/S11: profiles carry LB2 G3 from config; G1 and G2 stay fixed
    def test_profiles_carry_g3_from_config(self):
        from gwo_de_dtw import config
        self.assertEqual((config.G1, config.G2), (0.5, 0.5))
        self.assertEqual(config.EXPLOIT_G3, 0.1)
        self.assertEqual(config.EXPLORE_G3, 0.1)
        self.assertFalse(hasattr(config, "G3"))
        self.assertEqual(EXPLOIT.G3, config.EXPLOIT_G3)
        self.assertEqual(EXPLORE.G3, config.EXPLORE_G3)
        self.assertEqual((EXPLOIT.a, EXPLOIT.F, EXPLOIT.CR), (0.5, 0.5, 0.9))
        self.assertEqual((EXPLORE.a, EXPLORE.F, EXPLORE.CR), (2.0, 0.9, 0.3))

    def test_constructor_has_no_g3_parameter(self):
        with self.assertRaises(TypeError):
            GWODEHybrid(self.inst, np.random.default_rng(0), pop_size=self.N, G3=0.1)

    # S11: a profile change changes the G3 passed to LB2; G1 and G2 do not
    def test_lb2_uses_the_active_profile_g3(self):
        import gwo_de_dtw.hybrid as hybrid_mod
        from unittest.mock import patch
        seen = []
        real = hybrid_mod.lb2_binarize

        def spy(v, current, inst, rng, G1, G2, G3, v_max):
            seen.append((G1, G2, G3))
            return real(v, current, inst, rng, G1, G2, G3, v_max)

        # Distinct values so the profile change is observable (defaults are both 0.1).
        exploit = Profile("exploit", EXPLOIT.a, EXPLOIT.F, EXPLOIT.CR, 0.1)
        explore = Profile("explore", EXPLORE.a, EXPLORE.F, EXPLORE.CR, 0.3)
        h = self.make()
        h.initialize()
        with patch.object(hybrid_mod, "lb2_binarize", spy):
            h.set_profile(exploit)
            h.step()
            exploit_seen = list(seen)
            seen.clear()
            h.set_profile(explore)
            h.step()
            explore_seen = list(seen)
        self.assertEqual(len(exploit_seen), 2 * self.N)
        self.assertEqual(len(explore_seen), 2 * self.N)
        self.assertTrue(all(row == (h.G1, h.G2, 0.1) for row in exploit_seen))
        self.assertTrue(all(row == (h.G1, h.G2, 0.3) for row in explore_seen))
        self.assertNotEqual(exploit_seen[0][2], explore_seen[0][2])

    def test_same_seed_is_deterministic(self):
        runs = []
        for _ in range(2):
            h = self.make(seed=7)
            h.initialize()
            for _ in range(4):
                h.step()
            runs.append((h.best_fitness, h.fit.copy(), h.x_real.copy()))
        self.assertEqual(runs[0][0], runs[1][0])
        np.testing.assert_array_equal(runs[0][1], runs[1][1])
        np.testing.assert_array_equal(runs[0][2], runs[1][2])


class SigmoidHybridTests(unittest.TestCase):
    """S14: sigmoid binarization, continuous space [-6, 6], one evaluation per candidate."""

    def setUp(self):
        self.inst = make_instance()
        self.N = 10

    def make(self, seed=1, **kwargs):
        return GWODEHybrid(self.inst, np.random.default_rng(seed), pop_size=self.N,
                           binarization="sigmoid", **kwargs)

    def assert_population_valid(self, h):
        self.assertTrue(np.all(np.abs(h.x_real) <= 6.0))
        for x, f in zip(h.x_bin, h.fit):
            self.assertTrue(feasible(self.inst, x))
            self.assertAlmostEqual(f, float(self.inst["p"] @ x))

    def test_v_max_defaults_to_sigmoid_range(self):
        from gwo_de_dtw import config
        self.assertEqual(self.make().v_max, config.SIGMOID_V_MAX)
        self.assertEqual(self.make().v_max, 6.0)

    def test_initialization_is_within_range_and_costs_n(self):
        h = self.make()
        h.initialize()
        self.assertEqual(h.evaluations, self.N)
        self.assertEqual(h.x_real.shape, (self.N, self.inst["n"]))
        self.assert_population_valid(h)
        self.assertEqual(h.best_fitness, float(np.max(h.fit)))

    def test_each_generation_costs_two_n_and_stays_in_range(self):
        h = self.make()
        h.initialize()
        prev_fit = h.fit.copy()
        for _ in range(5):
            before = h.evaluations
            h.step()
            self.assertEqual(h.evaluations - before, 2 * self.N)
            self.assertTrue(np.all(h.fit >= prev_fit))
            self.assert_population_valid(h)
            prev_fit = h.fit.copy()
        self.assertEqual(h.best_fitness, float(self.inst["p"] @ h.best_solution))

    # The G2 < v_max check is an LB2 rule; it does not apply to sigmoid.
    def test_g2_check_does_not_apply(self):
        self.assertEqual(self.make(G2=120.0).G2, 120.0)


class SigmoidBinarizationTests(unittest.TestCase):
    def setUp(self):
        self.inst = make_instance()

    # S14: P(bit = 1) = 1 / (1 + exp(-v)), then repair
    def test_returns_feasible_repaired_bits_with_matching_fitness(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            v = rng.uniform(-6.0, 6.0, self.inst["n"])
            bits, fit = sigmoid_binarize(v, self.inst, rng)
            self.assertTrue(feasible(self.inst, bits))
            self.assertAlmostEqual(fit, float(self.inst["p"] @ bits))
            np.testing.assert_array_equal(bits, reparar(bits, self.inst)[0])
        self.assertEqual(SIGMOID_EVALUATIONS, 1)

    def test_extreme_values_give_repaired_all_ones_or_all_zeros(self):
        n = self.inst["n"]
        rng = np.random.default_rng(0)
        bits, fit = sigmoid_binarize(np.full(n, 60.0), self.inst, rng)
        expected, expected_fit = reparar(np.ones(n, dtype=int), self.inst)
        np.testing.assert_array_equal(bits, expected)
        self.assertEqual(fit, expected_fit)
        bits, fit = sigmoid_binarize(np.full(n, -60.0), self.inst, rng)
        expected, expected_fit = reparar(np.zeros(n, dtype=int), self.inst)
        np.testing.assert_array_equal(bits, expected)
        self.assertEqual(fit, expected_fit)

    def test_same_seed_is_deterministic(self):
        v = np.linspace(-6.0, 6.0, self.inst["n"])
        a = sigmoid_binarize(v, self.inst, np.random.default_rng(4))
        b = sigmoid_binarize(v, self.inst, np.random.default_rng(4))
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[1], b[1])


class ConfigTests(unittest.TestCase):
    # S12
    def test_config_defaults(self):
        from gwo_de_dtw import config
        self.assertEqual(config.INSTANCE, "instancias/mknapcb1.txt")
        self.assertEqual(config.INDEX, 0)
        self.assertEqual(config.POP_SIZE, 30)
        self.assertEqual(config.ITERATIONS, 1000)
        self.assertEqual(config.SEED, 1)
        self.assertEqual(config.V_MAX, 100.0)
        self.assertEqual((config.G1, config.G2), (0.5, 0.5))
        self.assertEqual((config.EXPLOIT_G3, config.EXPLORE_G3), (0.1, 0.1))
        self.assertFalse(hasattr(config, "G3"))
        self.assertEqual((EXPLOIT.G3, EXPLORE.G3), (config.EXPLOIT_G3, config.EXPLORE_G3))
        self.assertEqual((EXPLOIT.a, EXPLOIT.F, EXPLOIT.CR),
                         (config.EXPLOIT_A, config.EXPLOIT_F, config.EXPLOIT_CR))
        self.assertEqual((EXPLORE.a, EXPLORE.F, EXPLORE.CR),
                         (config.EXPLORE_A, config.EXPLORE_F, config.EXPLORE_CR))
        self.assertEqual(config.WINDOW, 40)
        self.assertEqual(config.BAND, 0)
        self.assertEqual(StagnationConfig(window=config.WINDOW, band=config.BAND).band, 4)
        self.assertEqual(config.MIN_SLOPE, 2.0)
        self.assertEqual(config.PLATEAU_MAX, 15)
        self.assertEqual(config.PATIENCE, 8)
        self.assertTrue(config.USE_DDTW)
        self.assertTrue(config.ADAPT_THRESHOLDS)
        self.assertEqual((config.P_LOW, config.P_HIGH), (40.0, 60.0))
        self.assertEqual(config.IMPROVEMENT_TOL, 0.0)
        self.assertEqual(config.BINARIZATION, "lb2")
        self.assertEqual(config.SIGMOID_V_MAX, 6.0)


class RepairTests(unittest.TestCase):
    """S1: repair and density ported from hybrid_mkp."""

    P = np.array([10.0, 9.0, 8.0, 1.0, 3.0])
    R = np.array([[2.0, 5.0, 1.0, 1.0, 0.0],
                  [0.0, 1.0, 4.0, 1.0, 0.0]])
    B = np.array([6.0, 5.0])

    def make(self):
        from gwo_de_dtw.problem import densidad_minima
        return {"n": 5, "m": 2, "p": self.P, "r": self.R, "b": self.B,
                "optimo": 0.0, "densidad": densidad_minima(self.P, self.R)}

    # min_i(p_j / r_ij); p/r rows are [5, 1.8, 8, 1, inf] and [inf, 9, 2, 1, inf];
    # inf -> 10 x max finite (90), so item 4 (no weight anywhere) gets 90.
    def test_density_is_min_ratio_with_inf_replaced(self):
        np.testing.assert_allclose(self.make()["densidad"], [5.0, 1.8, 2.0, 1.0, 90.0])

    # Insertion order by density 4, 0, 2, 1, 3: item 1 does not fit (row 0 would be
    # 8 > 6) and is skipped; item 3 still fits afterwards.
    def test_insertion_skips_non_fitting_item_and_inserts_later_one(self):
        bits, fit = reparar(np.zeros(5, dtype=int), self.make())
        np.testing.assert_array_equal(bits, [1, 0, 1, 1, 1])
        self.assertEqual(fit, 22.0)

    # Expulsion from lowest density: item 3, then item 1, until feasible.
    def test_expulsion_removes_lowest_density_until_feasible(self):
        x = np.ones(5, dtype=int)
        bits, fit = reparar(x, self.make())
        np.testing.assert_array_equal(x, np.ones(5, dtype=int))   # input untouched
        np.testing.assert_array_equal(bits, [1, 0, 1, 1, 1])
        self.assertEqual(fit, 22.0)

    def test_loaded_instance_uses_min_ratio_density_and_known_optimum(self):
        from gwo_de_dtw.problem import cargar_instancia
        path = str(ROOT / "instancias" / "mknapcb1.txt")
        inst = cargar_instancia(path, 0)
        self.assertEqual((inst["n"], inst["m"]), (100, 5))
        np.testing.assert_allclose(inst["densidad"], np.min(inst["p"] / inst["r"], axis=0))
        self.assertEqual(inst["optimo"], 24381.0)

    # S13: best known values come from instancias/mkcbres.txt for every index
    def test_best_known_values_come_from_mkcbres_next_to_the_instances(self):
        from gwo_de_dtw.problem import cargar_instancia
        folder = ROOT / "instancias"
        self.assertEqual(cargar_instancia(str(folder / "mknapcb1.txt"), 1)["optimo"], 24274.0)
        self.assertEqual(cargar_instancia(str(folder / "mknapcb9.txt"), 29)["optimo"], 300460.0)

    def test_missing_mkcbres_leaves_optimum_at_zero(self):
        import shutil
        import tempfile
        from gwo_de_dtw.problem import cargar_instancia
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "mknapcb1.txt"
            shutil.copy(ROOT / "instancias" / "mknapcb1.txt", copy)
            self.assertEqual(cargar_instancia(str(copy), 0)["optimo"], 0.0)


class OperatorTests(unittest.TestCase):
    V_MAX = 100.0

    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 15
        self.population = rng.uniform(-150.0, 150.0, (6, self.n))
        self.leaders = self.population[:3].copy()

    # S2: GWO candidate stays inside [-v_max, v_max]
    def test_gwo_candidate_stays_within_bounds(self):
        rng = np.random.default_rng(1)
        for a in (0.5, 2.0):
            u = gwo_candidate(self.population[4], self.leaders, a, rng, self.V_MAX)
            self.assertEqual(u.shape, (self.n,))
            self.assertTrue(np.all(np.abs(u) <= self.V_MAX))

    # S2: a = 0 gives A = 0, so every guided position is its leader
    def test_gwo_candidate_with_zero_a_is_clipped_leader_mean(self):
        u = gwo_candidate(self.population[4], self.leaders, 0.0,
                          np.random.default_rng(1), self.V_MAX)
        expected = np.clip(self.leaders.mean(axis=0), -self.V_MAX, self.V_MAX)
        np.testing.assert_allclose(u, expected)

    # S2: CR = 0 keeps every coordinate of x_i except the forced j_rand one
    def test_de_candidate_with_zero_cr_changes_at_most_one_coordinate(self):
        population = np.clip(self.population, -self.V_MAX, self.V_MAX)
        rng = np.random.default_rng(2)
        for i in range(population.shape[0]):
            u = de_candidate(i, population, 0.5, 0.0, rng, self.V_MAX)
            self.assertLessEqual(int(np.sum(u != population[i])), 1)
            self.assertTrue(np.all(np.abs(u) <= self.V_MAX))


class LB2BinarizationTests(unittest.TestCase):
    def setUp(self):
        self.inst = make_instance()
        n = self.inst["n"]
        current = np.zeros(n, dtype=int)
        current[: n // 3] = 1
        self.current, _ = reparar(current, self.inst)

    # S11 formulas
    def test_probabilities_match_l1_l2_formulas(self):
        v = np.array([-6.0, -1.0, 0.0, 1.0, 6.0])
        L1, L2 = lb2_probabilities(v, 6.0, 0.5, 0.5, 0.5)
        np.testing.assert_allclose(L1, np.clip(-0.5 * v / 5.5 + 0.5, 0, 1))
        np.testing.assert_allclose(L2, np.clip(0.5 * v / 5.5 + 0.5, 0, 1))

    # S11: probability 0 -> no flip, the result is the repaired current solution
    def test_zero_flip_probability_keeps_current_solution(self):
        v = np.zeros(self.inst["n"])
        bits, fit = lb2_binarize(v, self.current, self.inst, np.random.default_rng(0),
                                 G1=0.5, G2=0.5, G3=0.0, v_max=6.0)
        expected, expected_fit = reparar(self.current, self.inst)
        np.testing.assert_array_equal(bits, expected)
        self.assertEqual(fit, expected_fit)

    # S11: probability 1 -> every bit is flipped before repair
    def test_full_flip_probability_flips_every_bit(self):
        v = np.zeros(self.inst["n"])
        bits, fit = lb2_binarize(v, self.current, self.inst, np.random.default_rng(0),
                                 G1=0.5, G2=0.5, G3=1.0, v_max=6.0)
        expected, expected_fit = reparar(1 - self.current, self.inst)
        np.testing.assert_array_equal(bits, expected)
        self.assertEqual(fit, expected_fit)

    # S11: returns the better of the L1 and L2 candidates
    def test_returns_better_of_l1_and_l2(self):
        n = self.inst["n"]
        v = np.full(n, 6.0)   # L1 = 0 (no flip), L2 = 1 (flip all)
        bits, fit = lb2_binarize(v, self.current, self.inst, np.random.default_rng(0),
                                 G1=1.0, G2=0.0, G3=0.0, v_max=6.0)
        _, keep_fit = reparar(self.current, self.inst)
        _, flip_fit = reparar(1 - self.current, self.inst)
        self.assertEqual(fit, max(keep_fit, flip_fit))
        self.assertTrue(feasible(self.inst, bits))


def scripted_cfg(use_ddtw=True):
    # Fixed thresholds (theta_c = 0.1 W, theta_r = 0.5 W) so a flat series fires.
    return StagnationConfig(window=10, band=1, min_slope=2.0, plateau_max=4,
                            patience=2, use_ddtw=use_ddtw, adapt_thresholds=False)


def feed(ctrl, values):
    return [ctrl.observe(v) for v in values]


class ProfileControllerTests(unittest.TestCase):
    W, PATIENCE = 10, 2

    def fire_on_flat(self, ctrl):
        outs = feed(ctrl, [100.0] * (self.W + self.PATIENCE - 1))
        return outs

    # S6, S7: no decision before W observations
    def test_no_change_before_window_is_full(self):
        for use_ddtw in (True, False):
            ctrl = ProfileController(scripted_cfg(use_ddtw))
            outs = feed(ctrl, [100.0] * (self.W - 1))
            self.assertTrue(all(not o["ready"] and not o["changed"] for o in outs))
            self.assertIs(ctrl.profile, EXPLOIT)

    # S8: exploit + fire -> explore (fire needs `patience` ready observations)
    def test_flat_series_fires_and_switches_to_explore(self):
        for use_ddtw in (True, False):
            ctrl = ProfileController(scripted_cfg(use_ddtw))
            outs = self.fire_on_flat(ctrl)
            self.assertEqual([o["changed"] for o in outs].index(True), len(outs) - 1)
            self.assertEqual(outs[-1]["profile"], "explore")
            self.assertIs(ctrl.profile, EXPLORE)

    # S7: the window that fired is discarded; no change before W new
    # observations. S8: the W-th fresh observation with recent improvement
    # returns to exploit.
    def test_fresh_window_waits_w_new_observations_before_returning(self):
        ctrl = ProfileController(scripted_cfg())
        self.fire_on_flat(ctrl)
        improving = [101.0 + k for k in range(self.W)]
        outs = feed(ctrl, improving)
        self.assertTrue(all(not o["changed"] and not o["ready"] for o in outs[:-1]))
        self.assertTrue(outs[-1]["changed"])
        self.assertEqual(outs[-1]["profile"], "exploit")
        self.assertIs(ctrl.profile, EXPLOIT)

    # S8: still stagnant after the fresh window -> stay in explore
    def test_explore_is_kept_while_stagnation_persists(self):
        ctrl = ProfileController(scripted_cfg())
        self.fire_on_flat(ctrl)
        outs = feed(ctrl, [100.0] * (3 * self.W))
        self.assertTrue(all(not o["changed"] for o in outs))
        self.assertIs(ctrl.profile, EXPLORE)

class RunHybridTests(unittest.TestCase):
    # S5: exactly `iterations` generations; evaluations counted as N + 4N per generation
    def test_run_stops_after_iterations_and_counts_evaluations(self):
        inst = make_instance(n=30, m=5, seed=3)
        N = 8
        cfg = StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                               patience=1, use_ddtw=True)
        for T in (0, 1, 7):
            res = run_hybrid(inst, cfg, pop_size=N, iterations=T, seed=5)
            self.assertEqual(res["generations"], T)
            self.assertEqual(res["iterations"], T)
            self.assertEqual(res["evaluations"], N + 4 * N * T)
            self.assertEqual(len(res["best_series"]), T)
            self.assertNotIn("max_evaluations", res)

    # S14: sigmoid costs N + 2N per generation and is reported
    def test_sigmoid_run_counts_two_n_per_generation(self):
        inst = make_instance(n=30, m=5, seed=3)
        N = 8
        cfg = StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                               patience=1, use_ddtw=True)
        for T in (0, 1, 7):
            res = run_hybrid(inst, cfg, pop_size=N, iterations=T, seed=5,
                             binarization="sigmoid")
            self.assertEqual(res["generations"], T)
            self.assertEqual(res["evaluations"], N + 2 * N * T)
            self.assertEqual(res["binarization"], "sigmoid")
        self.assertEqual(run_hybrid(inst, cfg, pop_size=N, iterations=1, seed=5)["binarization"],
                         "lb2")

    def test_run_exports_consistent_series(self):
        inst = make_instance(n=30, m=5, seed=3)
        N = 8
        cfg = StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                               patience=1, use_ddtw=True)
        res = run_hybrid(inst, cfg, pop_size=N, iterations=60, seed=5)
        self.assertEqual(res["generations"], 60)
        self.assertEqual(res["evaluations"], N + 4 * N * 60)
        series = res["best_series"]
        self.assertEqual(len(series), 60)
        self.assertTrue(all(b >= a for a, b in zip(series, series[1:])))
        self.assertEqual(res["best_fitness"], series[-1])
        self.assertEqual(res["best_fitness"], float(inst["p"] @ res["best_solution"]))
        self.assertTrue(feasible(inst, res["best_solution"]))
        profiles = res["profile_series"]
        self.assertEqual(len(profiles), 60)
        self.assertEqual(profiles[0], "exploit")
        # A change decided after generation g applies from generation g+1.
        for g in range(59):
            self.assertEqual(profiles[g + 1] != profiles[g], g in res["changes"])
        self.assertEqual(len(res["monitor"]), 60)

    # S15: parameters actually used in every generation
    def test_parameter_series_follows_profile_series(self):
        inst = make_instance(n=30, m=5, seed=3)
        cfg = StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                               patience=1, use_ddtw=True)
        res = run_hybrid(inst, cfg, pop_size=8, iterations=60, seed=5,
                         g3_exploit=0.1, g3_explore=0.3)
        params = res["parameter_series"]
        self.assertEqual(sorted(params), ["CR", "F", "G3", "a"])
        expected = {"exploit": (0.5, 0.5, 0.9, 0.1), "explore": (2.0, 0.9, 0.3, 0.3)}
        self.assertIn("explore", res["profile_series"])
        for key in ("a", "F", "CR", "G3"):
            self.assertEqual(len(params[key]), res["generations"])
        for g, name in enumerate(res["profile_series"]):
            self.assertEqual(
                (params["a"][g], params["F"][g], params["CR"][g], params["G3"][g]),
                expected[name])

    # S11: both profile values at 0.1 is a fixed G3, recorded every generation
    def test_fixed_g3_records_constant_series(self):
        from gwo_de_dtw import config
        inst = make_instance(n=30, m=5, seed=3)
        cfg = StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                               patience=1, use_ddtw=True)
        res = run_hybrid(inst, cfg, pop_size=8, iterations=60, seed=5,
                         g3_exploit=0.1, g3_explore=0.1)
        self.assertEqual(res["parameter_series"]["G3"], [0.1] * res["generations"])
        self.assertEqual(res["G"], {
            "G1": config.G1, "G2": config.G2, "G3_exploit": 0.1, "G3_explore": 0.1,
        })

    # S4/S11: the profiles built for the run are what LB2 actually flips with
    def test_run_hybrid_lb2_uses_profile_g3(self):
        import gwo_de_dtw.controller as controller_mod
        import gwo_de_dtw.hybrid as hybrid_mod
        from unittest.mock import patch
        inst = make_instance(n=30, m=5, seed=3)
        cfg = StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                               patience=1, use_ddtw=True)
        N, T = 8, 2
        seen = []
        real = hybrid_mod.lb2_binarize
        real_observe = controller_mod.ProfileController.observe

        def spy(v, current, inst_arg, rng, G1, G2, G3, v_max):
            seen.append(G3)
            return real(v, current, inst_arg, rng, G1, G2, G3, v_max)

        def observe(self, best_fitness):
            out = real_observe(self, best_fitness)
            # A change decided after generation 0 applies on generation 1.
            self.profile = self.explore
            out = dict(out)
            out["changed"] = True
            out["profile"] = self.profile.name
            return out

        with patch.object(controller_mod.ProfileController, "observe", observe), \
             patch.object(hybrid_mod, "lb2_binarize", spy):
            res = run_hybrid(inst, cfg, pop_size=N, iterations=T, seed=5,
                             g3_exploit=0.15, g3_explore=0.35)
        self.assertEqual(res["profile_series"], ["exploit", "explore"])
        self.assertEqual(len(seen), 2 * N * T)
        for g, name in enumerate(res["profile_series"]):
            expected = 0.15 if name == "exploit" else 0.35
            self.assertEqual(res["parameter_series"]["G3"][g], expected)
            self.assertEqual(seen[g * 2 * N:(g + 1) * 2 * N], [expected] * (2 * N))
        self.assertEqual(res["G"], {
            "G1": 0.5, "G2": 0.5, "G3_exploit": 0.15, "G3_explore": 0.35,
        })

    def test_run_hybrid_rejects_g3_parameter(self):
        inst = make_instance()
        cfg = StagnationConfig(window=8, band=1)
        with self.assertRaises(TypeError):
            run_hybrid(inst, cfg, pop_size=8, iterations=0, G3=0.1)

    # S14: sigmoid does not receive G, but the active profile G3 is still recorded
    def test_sigmoid_records_g3_without_using_it(self):
        import gwo_de_dtw.hybrid as hybrid_mod
        from unittest.mock import patch
        inst = make_instance(n=30, m=5, seed=3)
        cfg = StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                               patience=1, use_ddtw=True)
        import gwo_de_dtw.controller as controller_mod
        extras = []
        real = hybrid_mod.sigmoid_binarize
        real_observe = controller_mod.ProfileController.observe

        def spy(v, inst_arg, rng, *extra):
            extras.append(extra)
            return real(v, inst_arg, rng)

        def observe(self, best_fitness):
            out = real_observe(self, best_fitness)
            self.profile = self.explore
            out = dict(out)
            out["changed"] = True
            out["profile"] = self.profile.name
            return out

        with patch.object(controller_mod.ProfileController, "observe", observe), \
             patch.object(hybrid_mod, "sigmoid_binarize", spy):
            res = run_hybrid(inst, cfg, pop_size=8, iterations=2, seed=5,
                             binarization="sigmoid", g3_exploit=0.2, g3_explore=0.4)
        self.assertTrue(extras)
        self.assertTrue(all(extra == () for extra in extras))
        self.assertEqual(res["profile_series"], ["exploit", "explore"])
        self.assertEqual(res["parameter_series"]["G3"], [0.2, 0.4])

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PLOT_FILES = ["dtw_delta.png", "fitness_convergence.png", "parameters.png", "temporal_diagnostics.png"]
TEMPORAL_FILES = ["temporal_diagnostics.pdf", "temporal_series.npz", "temporal_series.csv", "temporal_metadata.json"]


def plot_lines(folder):
    """Expected `[plot]` lines in save order; padding copied from hybrid_mkp."""
    return [f"  [plot] fitness_convergence.png   -> '{Path(folder) / 'fitness_convergence.png'}'",
            f"  [plot] dtw_delta.png             -> '{Path(folder) / 'dtw_delta.png'}'",
            f"  [plot] parameters.png -> '{Path(folder) / 'parameters.png'}'",
            f"  [plot] temporal_diagnostics.png -> '{Path(folder) / 'temporal_diagnostics.png'}'"]


def assert_png_files(test, folder, names):
    test.assertEqual(sorted(p.name for p in Path(folder).iterdir()), sorted(names + TEMPORAL_FILES))
    for name in names:
        data = (Path(folder) / name).read_bytes()
        test.assertGreater(len(data), len(PNG_SIGNATURE))
        test.assertEqual(data[:8], PNG_SIGNATURE)


class PlotTests(unittest.TestCase):
    """S15: one PNG per plot."""

    @classmethod
    def setUpClass(cls):
        inst = make_instance(n=30, m=5, seed=3)
        cls.cfg = StagnationConfig(window=8, band=1, min_slope=2.0, plateau_max=3,
                                   patience=1, use_ddtw=True)
        cls.inst = inst
        cls.res = run_hybrid(inst, cls.cfg, pop_size=8, iterations=60, seed=5)

    def test_profile_colors_are_in_config(self):
        from gwo_de_dtw import config
        self.assertEqual(config.PROFILE_COLORS, {"exploit": "#1f77b4", "explore": "#ff7f0e"})

    def test_save_all_plots_writes_four_pngs_and_temporal_series(self):
        import contextlib
        import io
        import tempfile
        from gwo_de_dtw.plots import save_all_plots
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "run_plots"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                paths = save_all_plots(self.res, folder)
            self.assertEqual([Path(p).name for p in paths],
                             ["fitness_convergence.png", "dtw_delta.png", "parameters.png", "temporal_diagnostics.png"])
            assert_png_files(self, folder, PLOT_FILES)
            self.assertEqual(stdout.getvalue().splitlines(), plot_lines(folder))

    # S15: hybrid_mkp's log_switches become the profile turns
    def test_profile_turns_are_consecutive_profile_segments(self):
        from gwo_de_dtw.plots._common import profile_turns
        series = ["exploit"] * 3 + ["explore"] * 2 + ["exploit"]
        self.assertEqual(profile_turns(series),
                         [("exploit", 3), ("explore", 2), ("exploit", 1)])
        self.assertEqual(profile_turns([]), [])

    # S15: hybrid_mkp's dtw_deltas_global: monitor delta when ready, NaN otherwise
    def test_delta_series_is_nan_where_window_is_not_ready(self):
        import math
        from gwo_de_dtw.plots.dtw_delta import delta_series
        deltas = delta_series(self.res)
        self.assertEqual(len(deltas), self.res["generations"])
        self.assertTrue(any(o["ready"] for o in self.res["monitor"]))
        self.assertFalse(all(o["ready"] for o in self.res["monitor"]))
        for value, out in zip(deltas, self.res["monitor"]):
            if out["ready"]:
                self.assertEqual(value, out["delta"])
            else:
                self.assertTrue(math.isnan(value))

    def test_dtw_delta_is_skipped_when_no_window_is_ready(self):
        import contextlib
        import io
        import tempfile
        from gwo_de_dtw.plots.dtw_delta import plot_dtw_delta
        res = run_hybrid(self.inst, self.cfg, pop_size=8, iterations=5, seed=5)
        self.assertFalse(any(o["ready"] for o in res["monitor"]))
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(plot_dtw_delta(res, tmp), "")
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertEqual(stdout.getvalue(),
                             "  [plot] dtw_delta.png  -> sin datos suficientes, omitido.\n")

    # S15 (L35): G3 is not plotted; a, F and CR for every binarization
    def test_parameters_png_has_three_panels_without_g3(self):
        import tempfile
        from unittest.mock import patch
        from gwo_de_dtw.plots import parameters as parameters_mod
        from gwo_de_dtw.plots._common import save as real_save
        from gwo_de_dtw.plots.parameters import plot_parameters

        sigmoid = run_hybrid(self.inst, self.cfg, pop_size=8, iterations=5, seed=5,
                             binarization="sigmoid")
        counts = []
        labels = []
        real_subplots = parameters_mod.plt.subplots

        def spy_subplots(*args, **kwargs):
            counts.append(args[0] if args else kwargs["nrows"])
            return real_subplots(*args, **kwargs)

        def spy_save(fig, output_dir, filename):
            labels.append([ax.get_ylabel() for ax in fig.axes])
            return real_save(fig, output_dir, filename)

        with tempfile.TemporaryDirectory() as tmp:
            lb2_dir = Path(tmp) / "lb2"
            sig_dir = Path(tmp) / "sig"
            lb2_dir.mkdir()
            sig_dir.mkdir()
            with patch.object(parameters_mod.plt, "subplots", spy_subplots), \
                 patch.object(parameters_mod, "save", spy_save):
                lb2_path = plot_parameters(self.res, lb2_dir)
                sig_path = plot_parameters(sigmoid, sig_dir)
            for path in (lb2_path, sig_path):
                data = Path(path).read_bytes()
                self.assertEqual(data[:8], PNG_SIGNATURE)
                self.assertGreater(len(data), len(PNG_SIGNATURE))
        self.assertEqual(counts, [3, 3])
        self.assertEqual(labels[0], ["a (GWO)", "F (DE)", "CR (DE)"])
        self.assertEqual(labels[1], ["a (GWO)", "F (DE)", "CR (DE)"])


class RunCliTests(unittest.TestCase):
    def run_cli(self, *args):
        import subprocess
        return subprocess.run(
            [sys.executable, "-B", "-m", "gwo_de_dtw.run", *args],
            cwd=PKG_ROOT, capture_output=True, text=True, timeout=60,
        )

    def test_cli_runs_and_writes_json(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "run.json"
            res = self.run_cli("--instance", "instancias/mknapcb1.txt", "--index", "0",
                               "--signal", "dtw", "--pop", "8",
                               "--iterations", "5", "--seed", "3",
                               "--out", str(out_path))
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stderr, "")
            lines = res.stdout.splitlines()
            self.assertEqual(lines[0], "instance=mknapcb1.txt[0] binarization=lb2 "
                                       "signal=dtw W=40 pop=8 iterations=5 seed=3")
            self.assertRegex(lines[1], r"^best=\d+ known_optimum=\d+ gap=\d+\.\d{2}% "
                                       r"evaluations=168 generations=5 changes=\d+$")
            # S15: plot lines follow the summary; 5 < W, so the delta plot is skipped.
            self.assertEqual(len(lines), 6)
            self.assertTrue(all(line.startswith("  [plot] ") for line in lines[2:]))
            assert_png_files(self, Path(tmp) / "run_plots",
                             [f for f in PLOT_FILES if f != "dtw_delta.png"])
            data = json.loads(out_path.read_text())
            self.assertEqual(data["generations"], 5)
            self.assertEqual(data["iterations"], 5)
            self.assertEqual(data["evaluations"], 168)
            self.assertEqual(data["known_optimum"], 24381.0)
            self.assertFalse(data["use_ddtw"])
            self.assertEqual(data["binarization"], "lb2")
            self.assertEqual(data["G"], {
                "G1": 0.5, "G2": 0.5, "G3_exploit": 0.1, "G3_explore": 0.1,
            })
            self.assertEqual(len(data["best_solution"]), 100)

    # S15: --out also writes the three plots into <out without suffix>_plots/
    def test_cli_out_writes_plots_next_to_json(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "demo.json"
            res = self.run_cli("--pop", "8", "--iterations", "45", "--seed", "3", "--window", "40",
                               "--out", str(out_path))
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stderr, "")
            lines = res.stdout.splitlines()
            self.assertEqual(lines[0], "instance=mknapcb1.txt[0] binarization=lb2 "
                                       "signal=ddtw W=40 pop=8 iterations=45 seed=3")
            self.assertRegex(lines[1], r"^best=\d+ known_optimum=\d+ gap=\d+\.\d{2}% "
                                       r"evaluations=1448 generations=45 changes=\d+$")
            folder = Path(tmp) / "demo_plots"
            self.assertEqual(lines[2:], plot_lines(folder))
            assert_png_files(self, folder, PLOT_FILES)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["demo.json", "demo_plots"])

    # S14
    def test_cli_runs_with_sigmoid(self):
        res = self.run_cli("--binarization", "sigmoid", "--iterations", "5", "--pop", "8",
                           "--seed", "3", "--signal", "dtw")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stderr, "")
        lines = res.stdout.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "instance=mknapcb1.txt[0] binarization=sigmoid "
                                   "signal=dtw W=40 pop=8 iterations=5 seed=3")
        self.assertRegex(lines[1], r"^best=\d+ known_optimum=\d+ gap=\d+\.\d{2}% "
                                   r"evaluations=88 generations=5 changes=\d+$")

    # S14: the --g2 check is an LB2 rule only
    def test_cli_sigmoid_ignores_g2_check(self):
        res = self.run_cli("--binarization", "sigmoid", "--g2", "120",
                           "--iterations", "1", "--pop", "8")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(res.stdout.startswith("instance=mknapcb1.txt[0] binarization=sigmoid "))

    def test_cli_rejects_reset_counters(self):
        res = self.run_cli("--reset", "counters")
        self.assertEqual(res.returncode, 2)
        self.assertEqual(res.stdout, "")
        self.assertIn("unrecognized arguments: --reset counters", res.stderr)

    def test_cli_rejects_unknown_binarization(self):
        res = self.run_cli("--binarization", "foo")
        self.assertEqual(res.returncode, 2)
        self.assertEqual(res.stdout, "")
        self.assertIn("invalid choice: 'foo'", res.stderr)

    def test_cli_rejects_g2_not_below_v_max(self):
        res = self.run_cli("--g2", "120")
        self.assertEqual(res.returncode, 2)
        self.assertEqual(res.stdout, "")
        self.assertIn("--g2 must be below v_max=100", res.stderr)

    def test_cli_has_no_evals_option(self):
        res = self.run_cli("--evals", "100")
        self.assertEqual(res.returncode, 2)
        self.assertEqual(res.stdout, "")
        self.assertIn("unrecognized arguments: --evals 100", res.stderr)

    def test_cli_has_no_control_option(self):
        res = self.run_cli("--control", "fresh")
        self.assertEqual(res.returncode, 2)
        self.assertEqual(res.stdout, "")
        self.assertIn("unrecognized arguments: --control fresh", res.stderr)

    def test_cli_rejects_g3(self):
        res = self.run_cli("--g3", "0.1")
        self.assertEqual(res.returncode, 2)
        self.assertEqual(res.stdout, "")
        self.assertIn("unrecognized arguments: --g3 0.1", res.stderr)

    def test_cli_g3_exploit_and_explore_are_recorded(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "run.json"
            res = self.run_cli("--g3-exploit", "0.2", "--g3-explore", "0.4",
                               "--iterations", "5", "--pop", "8", "--seed", "3",
                               "--signal", "dtw", "--out", str(out_path))
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stderr, "")
            lines = res.stdout.splitlines()
            self.assertEqual(lines[0], "instance=mknapcb1.txt[0] binarization=lb2 "
                                       "signal=dtw W=40 pop=8 iterations=5 seed=3")
            data = json.loads(out_path.read_text())
            self.assertEqual(data["G"], {
                "G1": 0.5, "G2": 0.5, "G3_exploit": 0.2, "G3_explore": 0.4,
            })
            self.assertEqual(data["parameter_series"]["G3"], [0.2] * 5)
            self.assertTrue((Path(tmp) / "run_plots" / "parameters.png").is_file())


if __name__ == "__main__":
    unittest.main()
