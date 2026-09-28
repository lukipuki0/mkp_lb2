"""Functional adapter and backward-compatible orchestration regression tests."""

import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dtw_stagnation import StagnationConfig
from sensitivity.adapters import run_case
from sensitivity.adapters import cec, common, hres, mkp
from continuous_benchmark import orchestrator as continuous
from hybrid_mkp import orchestrator as discrete
from hybrid_mkp.mh.sa import SAParams, ejecutar_epoch as sa_epoch
from hybrid_mkp.mkp_core.problem import MKPInstance


def result(objective=3.0, solution=None, history=None, epochs=1):
    return SimpleNamespace(
        mejor_valor_global=objective,
        mejor_solucion_global=[0.0] * 10 if solution is None else solution,
        historial_global=[objective] if history is None else history,
        log_switches=[None] * epochs,
        gap_pct=None,
    )


def small_mkp(reference=0.0):
    return MKPInstance(8, 1, reference, np.arange(1, 9, dtype=float),
                       np.ones((1, 8)), np.array([4.0]))


class AdapterContracts(unittest.TestCase):
    def test_input_validation(self):
        valid = dict(domain="cec", case={"id": "f1", "function_id": 1},
                     monitor={}, seed=43, budget=2)
        for changes in ({"domain": "other"}, {"seed": True}, {"seed": -1},
                        {"budget": 0}, {"case": {}}, {"max_epoch_iters": 0},
                        {"monitor": {"patience": 0}}, {"monitor": {"p_low": 90, "p_high": 10}},
                        {"monitor": {"min_slope": float("nan")}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run_case(**{**valid, **changes})

    def test_mkp_local_file_and_reference(self):
        case = {"id": "cb1", "path": "instancias/mknapcb1.txt", "instance_index": 0,
                "reference_value": 24381, "reference_source": "ORLIB_BKS_INST00, existing runner"}
        loaded = mkp.load_case(case)
        self.assertEqual((loaded.n, loaded.m, loaded.valor_optimo), (100, 5, 24381))
        for changes in ({"path": "https://example.invalid/data"}, {"path": "../escape.txt"},
                        {"instance_index": -1}, {"reference_value": -1}, {"reference_source": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                mkp.load_case({**case, **changes})
        with self.assertRaises(FileNotFoundError):
            mkp.load_case({**case, "path": "instancias/missing.txt"})

    def test_unknown_mkp_reference_is_not_objective_fallback(self):
        fake = result(solution=[0] * 8)
        with patch.object(mkp, "load_case", return_value=small_mkp()), \
             patch.object(discrete, "ejecutar_pipeline", return_value=fake):
            record = run_case("mkp", {"id": "unknown"}, {"patience": 8}, 43, 1)
        self.assertEqual(record["metric_name"], "gap_pct")
        self.assertIsNone(record["metric"])
        self.assertEqual(record["effective_monitor"]["GA"]["patience"], 14)
        self.assertEqual(record["effective_monitor"]["PSO"]["patience"], 8)
        self.assertEqual(record["n_switches"], 0)

    def test_seed_is_set_after_case_construction(self):
        func = SimpleNamespace(n_dim=10, lb=-100, ub=100, optimum=300.0)
        observed = []

        def construct(case):
            np.random.seed(2008)
            random.seed(2008)
            return func

        def execute(*args, **kwargs):
            observed.append((random.random(), np.random.random()))
            return result(objective=301.0)

        expected = (random.Random(43).random(), np.random.RandomState(43).random_sample())
        with patch.object(cec, "load_case", side_effect=construct), \
             patch.object(continuous, "ejecutar_pipeline", side_effect=execute):
            record = run_case("cec", {"id": "f1"}, {}, 43, 1)
        self.assertEqual(observed, [expected])
        self.assertEqual(record["metric"], 1.0)

    def test_hres_infeasible_objective_is_not_lcoe(self):
        func = SimpleNamespace(
            config={"agsr_max": 0.2},
            get_info=lambda x: {"feasible": False, "lcoe_cny_per_kwh": 0.5, "agsr": float("nan")},
        )
        orch = SimpleNamespace(
            POOL_POBLACIONAL_HRES2=["PSO"], POOL_TRAYECTORIA_HRES2=["SA"],
            ejecutar_pipeline_hres2=lambda *args, **kwargs: result(objective=110.0, solution=[1] * 4),
        )
        with patch.object(hres, "load_case", return_value=func), \
             patch.object(hres, "load_hres_module", return_value=orch):
            record = run_case("hres", {"id": "hres"}, {}, 43, 1)
        self.assertEqual(record["objective"], 110.0)
        self.assertIsNone(record["metric"])
        self.assertEqual(record["raw_lcoe_cny_per_kwh"], 0.5)
        self.assertIsNone(record["constraints"]["agsr"])
        json.dumps(record, allow_nan=False)

    def test_result_counts_actual_iterations_and_overshoot(self):
        record = common.result_record(
            result(history=[1] * 5, epochs=3), metric_name="test", metric=1,
            feasible=True, budget=2, elapsed_seconds=0.1, effective={},
        )
        self.assertEqual((record["actual_iters"], record["budget_overshoot"],
                          record["n_epochs"], record["n_switches"]), (5, 3, 3, 2))


class OrchestrationRegression(unittest.TestCase):
    def setUp(self):
        self.func = SimpleNamespace(optimum=0.0, name="test", n_dim=1, lb=-1, ub=1)
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        count = kwargs.get("max_epoch_iters", 1)
        return SimpleNamespace(mejor_valor=1.0, mejor_solucion=[0.0],
                               historial=[1.0] * count, dtw_deltas=[], dtw_info_hist=[])

    def test_continuous_defaults_remain_population_only(self):
        continuous.ejecutar_pipeline(self.func, max_iters=3, verbose=False,
                                    pool_poblacional=["P1", "P2"], ejecutar_mh_fn=self.execute)
        self.assertTrue(all(call["mh_nombre"].startswith("P") for call in self.calls))
        self.assertTrue(all("max_epoch_iters" not in call for call in self.calls))
        self.assertNotEqual(self.calls[0]["mh_nombre"], self.calls[1]["mh_nombre"])

    def test_hres_alternates_trajectory_and_population(self):
        orch = common.load_hres_module("orchestrator.py")
        with patch.object(orch, "_ejecutar_mh_hres2", side_effect=self.execute):
            res = orch.ejecutar_pipeline_hres2(
                self.func, max_iters=3, verbose=False,
                pool_poblacional=["P"], pool_trayectoria=["T"],
            )
        self.assertEqual([call["mh_nombre"] for call in self.calls], ["P", "T", "P"])
        self.assertEqual([log.tipo for log in res.log_switches],
                         ["poblacional", "trayectoria", "poblacional"])

    def test_explicit_cap_passes_remaining_budget(self):
        res = continuous.ejecutar_pipeline(
            self.func, max_iters=3, verbose=False, pool_poblacional=["P"],
            ejecutar_mh_fn=self.execute, max_epoch_iters=2,
        )
        self.assertEqual([call["max_epoch_iters"] for call in self.calls], [2, 1])
        self.assertEqual(len(res.historial_global), 3)

    def test_empty_history_fails_instead_of_hanging(self):
        empty = SimpleNamespace(historial=[])
        with self.assertRaisesRegex(RuntimeError, "empty epoch"):
            continuous.ejecutar_pipeline(self.func, max_iters=1, verbose=False,
                                         ejecutar_mh_fn=lambda **kwargs: empty)
        with patch.object(discrete, "_ejecutar_mh", return_value=empty), \
             self.assertRaisesRegex(RuntimeError, "empty epoch"):
            discrete.ejecutar_pipeline(small_mkp(), max_iters=1, verbose=False)

    def test_invalid_pools_fail(self):
        for kwargs in ({"pool_poblacional": []}, {"pool_poblacional": [None]},
                       {"pool_poblacional": "PSO"}, {"pool_trayectoria": "SA"},
                       {"pool_trayectoria": ["SA"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                continuous.ejecutar_pipeline(self.func, max_iters=1, verbose=False, **kwargs)

    def test_population_default_epoch_length_is_preserved(self):
        for cap, expected in ((None, 300), (2, 2)):
            with self.subTest(cap=cap), patch.object(continuous, "_pso_epoch") as execute:
                continuous._ejecutar_mh("PSO", self.func, None, StagnationConfig(),
                                       "random", 0, False, max_epoch_iters=cap)
                self.assertEqual(execute.call_args.args[1].iterations, expected)

    def test_mkp_patience_and_epoch_defaults_are_preserved(self):
        for cap, expected in ((None, 500), (2, 2)):
            with self.subTest(cap=cap), patch.object(discrete, "_ga_epoch") as execute:
                discrete._ejecutar_mh("GA", small_mkp(), None, StagnationConfig(patience=8),
                                     "mixed", 0, False, max_epoch_iters=cap)
                params = execute.call_args.args[1]
                self.assertEqual(params.generations, expected)
                self.assertEqual(params.stag_cfg.patience, 14)

    def test_sa_explicit_level_cap_without_truncation(self):
        params = SAParams(T_inicial=4.0, T_final=1.0, alpha=0.5,
                          iter_por_T=1, use_stagnation=False, max_levels=1)
        capped = sa_epoch(small_mkp(), params, verbose=False)
        self.assertEqual(len(capped.historial), 1)
        params.max_levels = None
        legacy = sa_epoch(small_mkp(), params, verbose=False)
        self.assertEqual(len(legacy.historial), 2)


class TinyRealAdapterSmokes(unittest.TestCase):
    def test_three_domains_and_monitor_modes(self):
        cases = {
            "mkp": {"id": "cb1", "path": "instancias/mknapcb1.txt", "instance_index": 0,
                    "reference_value": 24381, "reference_source": "existing ORLIB_BKS_INST00"},
            "cec": {"id": "f1", "function_id": 1, "dimension": 10},
            "hres": {"id": "hres"},
        }
        for domain, case in cases.items():
            for use_ddtw in (False, True):
                with self.subTest(domain=domain, use_ddtw=use_ddtw):
                    record = run_case(domain, case, {
                        "window": 2, "patience": 1, "plateau_max": 1, "use_ddtw": use_ddtw,
                    }, seed=43, budget=4, max_epoch_iters=2)
                    self.assertEqual(record["actual_iters"], 4)
                    self.assertEqual(record["n_epochs"], 2)
                    self.assertEqual(record["n_switches"], 1)
                    self.assertEqual(record["budget_overshoot"], 0)
                    json.dumps(record, allow_nan=False)
                    self.assertTrue(all(cfg["use_ddtw"] == use_ddtw
                                        for cfg in record["effective_monitor"].values()))


if __name__ == "__main__":
    unittest.main()
