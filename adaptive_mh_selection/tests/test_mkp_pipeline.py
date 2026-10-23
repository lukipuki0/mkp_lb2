"""Native iteration, optional policy and legacy caller regressions."""
import dataclasses
import inspect
import json
import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import hybrid_mkp.orchestrator as native
from hybrid_mkp.mkp_core.problem import MKPInstance


def instance():
    return MKPInstance(8, 2, 0., np.array([9., 5., 8., 3., 7., 11., 6., 4.]),
                       np.array([[4., 2., 3., 2., 4., 5., 3., 2.],
                                 [2., 3., 4., 1., 2., 4., 2., 3.]]), np.array([13., 11.]))


def epoch(value=10., n=3, info=True):
    return SimpleNamespace(mejor_valor=value, mejor_solucion=[1, 0] * 4,
                           historial=[value] * n, historial_inst=[value - 1] * n,
                           dtw_deltas=[.2],
                           dtw_info_hist=([{'ready': False}] * (n - 1) +
                                          [{'ready': True, 'delta': .2}]) if info else [])


def deterministic(result):
    data = dataclasses.asdict(result)
    for block in data['log_switches']:
        block.pop('t_inicio')
        block.pop('t_fin')
    return json.dumps(data, sort_keys=True)  # Native missing deltas remain NaN.


class NativeHookTests(unittest.TestCase):
    def test_default_uses_original_random_choice_and_epoch_overshoot(self):
        chosen = ['ACO', 'TS']
        with patch.object(native.random, 'choice', side_effect=chosen) as choice, \
                patch.object(native, '_ejecutar_mh', side_effect=[epoch(), epoch(12.)]) as dispatch:
            result = native.ejecutar_pipeline(instance(), max_iters=4, verbose=False)
        self.assertEqual([c.args[0] for c in choice.call_args_list],
                         [native.POOL_POBLACIONAL, native.POOL_TRAYECTORIA])
        self.assertEqual(len(result.historial_global), 6)
        self.assertEqual(result.historial_global, [10.] * 3 + [12.] * 3)
        self.assertIsNone(dispatch.call_args_list[0].kwargs['solucion_global'])
        self.assertEqual(dispatch.call_args_list[1].kwargs['solucion_global'], [1, 0] * 4)
        self.assertTrue(all(c.kwargs['max_epoch_iters'] is None for c in dispatch.call_args_list))
        self.assertEqual(len(result.historial_inst_global), 6)
        self.assertEqual(len(result.dtw_deltas_global), 6)
        self.assertEqual([i for i, v in enumerate(result.dtw_deltas_global) if math.isfinite(v)], [2, 5])
        self.assertEqual(len(result.dtw_info_global), 6)

    def test_optional_hook_observes_completed_native_blocks(self):
        policy = Mock()
        policy.choose.side_effect = ['GA', 'SA']
        results = [epoch(), epoch(12.)]
        with patch.object(native, '_ejecutar_mh', side_effect=results):
            result = native.ejecutar_pipeline(instance(), 4, verbose=False, policy=policy)
        self.assertEqual(policy.choose.call_args_list[0].args, (tuple(native.POOL_POBLACIONAL), 'poblacional'))
        self.assertEqual(policy.observe.call_args_list[0].args, ('GA', 'poblacional', -math.inf, results[0], 3, 10.))
        self.assertEqual(policy.observe.call_args_list[1].args, ('SA', 'trayectoria', 10., results[1], 3, 12.))
        self.assertIsInstance(result, native.PipelineResult)
        self.assertEqual(list(inspect.signature(native.ejecutar_pipeline).parameters)[-1], 'policy')

    def test_invalid_choice_and_failed_epoch_are_not_observed(self):
        policy = Mock()
        policy.choose.return_value = 'SA'
        with patch.object(native, '_ejecutar_mh') as dispatch:
            with self.assertRaises(ValueError):
                native.ejecutar_pipeline(instance(), 1, verbose=False, policy=policy)
        dispatch.assert_not_called()
        policy.observe.assert_not_called()
        policy.choose.return_value = 'GA'
        with patch.object(native, '_ejecutar_mh', return_value=epoch(n=0)):
            with self.assertRaisesRegex(RuntimeError, 'empty'):
                native.ejecutar_pipeline(instance(), 1, verbose=False, policy=policy)
        policy.observe.assert_not_called()


class WrapperTests(unittest.TestCase):
    def test_uniform_exact_native_parity_and_rng_restoration(self):
        from adaptive_mh_selection import RunConfig, run_pipeline
        from adaptive_mh_selection.orchestrator import isolated_rng
        state, np_state = random.getstate(), np.random.get_state()
        for seed in (1, 42, 17):
            config = RunConfig(seed=seed, selector='uniform', max_iters=5, max_epoch_iters=2)
            with isolated_rng(seed):
                original = native.ejecutar_pipeline(instance(), 5, stag_cfg=config.monitor,
                                                    verbose=False, max_epoch_iters=2)
            wrapped = run_pipeline(instance(), config)
            self.assertEqual(deterministic(wrapped['native_result']), deterministic(original))
            self.assertEqual(wrapped['summary']['actual_iterations'], 5)
            self.assertEqual(wrapped['summary']['iteration_overshoot'], 0)
            self.assertEqual(wrapped['summary']['best_solution'], original.mejor_solucion_global)
            self.assertEqual(wrapped['summary']['best_fitness'], original.mejor_valor_global)
            self.assertTrue(instance().es_factible(original.mejor_solucion_global))
            self.assertEqual(instance().evaluar(original.mejor_solucion_global), original.mejor_valor_global)
        self.assertEqual(random.getstate(), state)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertEqual(np.random.get_state()[2:], np_state[2:])
        with patch.object(native, 'ejecutar_pipeline', side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                run_pipeline(instance(), RunConfig())
        self.assertEqual(random.getstate(), state)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])

    def test_shared_pipeline_warmup_ewma_offsets_and_missing_info(self):
        from adaptive_mh_selection import RunConfig, run_pipeline
        config = RunConfig(max_iters=7)
        with patch.object(native, '_ejecutar_mh', side_effect=[epoch(100.), epoch(120., info=False), epoch(130.)]), \
                patch.object(native, 'ejecutar_pipeline', wraps=native.ejecutar_pipeline) as pipeline:
            result = run_pipeline(instance(), config)
        pipeline.assert_called_once()
        self.assertIsNone(pipeline.call_args.kwargs['max_epoch_iters'])
        summary, records = result['summary'], result['selection_records']
        self.assertEqual((summary['requested_iterations'], summary['actual_iterations'], summary['iteration_overshoot']), (7, 9, 2))
        self.assertEqual((summary['n_blocks'], summary['transitions']), (3, 2))
        self.assertEqual([r['iteration_start'] for r in records], [0, 3, 6])
        self.assertEqual([r['iteration_end'] for r in records], [3, 6, 9])
        self.assertEqual([r['pool'] for r in records], ['poblacional', 'trayectoria', 'poblacional'])
        self.assertEqual(records[0]['reward'], 0.)
        self.assertTrue(records[0]['warmup'])
        self.assertIsNone(records[0]['fitness_before'])
        self.assertAlmostEqual(records[1]['reward'], 20 / 100 / 3)
        self.assertAlmostEqual(records[2]['reward'], 10 / 100 / 3)
        self.assertEqual(records[2]['scores_before'], dict.fromkeys(native.POOL_POBLACIONAL, 0.))
        self.assertAlmostEqual(records[1]['scores_after'][records[1]['solver']], .3 * 20 / 100 / 3)
        self.assertFalse(summary['dtw_info_complete'])
        self.assertEqual(summary['dtw_info_iterations'], 6)
        self.assertNotIn('evaluations', summary)
        json.dumps(summary, allow_nan=False)
        json.dumps(records, allow_nan=False)

    def test_configured_defaults_shared_with_cli_and_invalid_config(self):
        import contextlib
        import io
        from adaptive_mh_selection import RunConfig
        from adaptive_mh_selection.config import default_monitor
        from adaptive_mh_selection.run import main as run_main
        from hybrid_mkp import rotating_benchmark as rotating
        config = RunConfig()
        source = default_monitor()
        self.assertEqual((config.seed, config.selector, config.max_iters, config.max_epoch_iters),
                         (42, 'adaptive', 1000, None))
        self.assertEqual((config.alpha, config.epsilon, config.pop_injection_mode), (.3, .2, 'mixed'))
        self.assertTrue(config.use_ddtw)
        self.assertEqual((source.window, source.band, source.min_slope, source.plateau_max, source.patience),
                         (70, 7, 2, 25, 15))
        self.assertEqual((source.p_low, source.p_high, source.adapt_thresholds, source.improvement_tol),
                         (30., 70., True, 0.))
        self.assertTrue(source.use_ddtw)
        self.assertEqual(config.monitor.use_ddtw, config.use_ddtw)
        for name in ('window', 'band', 'min_slope', 'plateau_max', 'patience', 'adapt_thresholds',
                     'p_low', 'p_high', 'improvement_tol'):
            self.assertEqual(getattr(config.monitor, name), getattr(source, name), name)
        # Native rotating_benchmark constants stay with the original solver; they are not these defaults.
        self.assertEqual((rotating.STAG_WINDOW, rotating.STAG_MIN_SLOPE, rotating.STAG_USE_DDTW), (40, .5, False))
        self.assertNotEqual(config.monitor.window, rotating.STAG_WINDOW)
        self.assertNotEqual(config.monitor.min_slope, rotating.STAG_MIN_SLOPE)
        self.assertNotEqual(config.use_ddtw, rotating.STAG_USE_DDTW)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = run_main(['--dry-run'])
        self.assertEqual(code, 0)
        cli = json.loads(stdout.getvalue())['runs'][0]
        self.assertEqual(cli['use_ddtw'], config.use_ddtw)
        self.assertEqual(cli['monitor']['band'], source.band)
        self.assertEqual((cli['monitor']['p_low'], cli['monitor']['p_high']), (source.p_low, source.p_high))
        self.assertEqual(cli['monitor']['window'], source.window)
        self.assertEqual(cli['pop_injection_mode'], 'mixed')
        for kwargs in ({'max_iters': 0}, {'max_iters': True}, {'max_epoch_iters': 0},
                       {'seed': -1}, {'selector': 'bad'}, {'alpha': math.nan}):
            with self.assertRaises(ValueError):
                RunConfig(**kwargs)
        with self.assertRaises(TypeError):
            RunConfig(max_evaluations=100)

    def test_all_native_dispatcher_limits_patience_and_repair_paths(self):
        from adaptive_mh_selection import RunConfig
        from hybrid_mkp.mkp_core.repair import reparar_solucion
        from mkp_core.repair import reparar_solucion as root_repair
        config = RunConfig()
        floor_monitor = dataclasses.replace(config.monitor, patience=8)
        inst = instance()
        inst.evaluate_candidate = Mock(side_effect=AssertionError('retired hook'))
        self.assertEqual(reparar_solucion([1, 0] * 4, inst), root_repair([1, 0] * 4, inst))
        for solver in native.POOL_POBLACIONAL + native.POOL_TRAYECTORIA:
            with self.subTest(solver=solver), patch.object(native, '_' + solver.lower() + '_epoch') as call:
                native._ejecutar_mh(solver, inst, None, floor_monitor, 'mixed', 0, False)
                params = call.call_args.args[1]
                self.assertEqual(params.stag_cfg.patience, 14 if solver in ('GA', 'ILS', 'WOA', 'GWO') else 8)
                native._ejecutar_mh(solver, inst, None, config.monitor, 'mixed', 0, False)
                self.assertEqual(call.call_args.args[1].stag_cfg.patience, config.monitor.patience)
                if solver == 'GA': self.assertEqual(params.generations, 500)
                elif solver == 'SA': self.assertIsNone(params.max_levels)
                else: self.assertEqual(params.iterations, 300 if solver in native.POOL_POBLACIONAL else 2000)
                native._ejecutar_mh(solver, inst, None, config.monitor, 'mixed', 0, False, max_epoch_iters=1)
                capped = call.call_args.args[1]
                self.assertEqual(getattr(capped, 'generations', getattr(capped, 'iterations', getattr(capped, 'max_levels', None))), 1)
        # Exercise direct ACO construction and all legacy solvers without evaluation hooks.
        for solver in native.POOL_POBLACIONAL + native.POOL_TRAYECTORIA:
            result = native._ejecutar_mh(solver, inst, None, config.monitor, 'mixed', 0, False, max_epoch_iters=1)
            self.assertEqual(len(result.historial), 1)
            self.assertEqual(len(result.historial_inst), 1)
            self.assertTrue(inst.es_factible(result.mejor_solucion))
        inst.evaluate_candidate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
