"""Opt-in pool subsets through configuration, CLI, policy and recorded plots."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from adaptive_mh_selection.config import RunConfig, POPULATION_POOL, TRAJECTORY_POOL
from adaptive_mh_selection.orchestrator import SelectionPolicy, isolated_rng, run_pipeline
from adaptive_mh_selection.tests.test_mkp_pipeline import instance, epoch
from adaptive_mh_selection.tests.test_selection_plot import (
    capture_probability_figure, decision, native_result,
)
from hybrid_mkp import orchestrator as native

ROOT = Path(__file__).resolve().parents[2]
POP = ('GA', 'PSO', 'ACO')
TRAJ = ('SA', 'ILS', 'VNS')


def reduced(**kwargs):
    return RunConfig(population_pool=POP, trajectory_pool=TRAJ, **kwargs)


class ReducedPoolTests(unittest.TestCase):
    def test_config_defaults_normalization_and_validation(self):
        default = RunConfig()
        self.assertEqual((default.population_pool, default.trajectory_pool),
                         (POPULATION_POOL, TRAJECTORY_POOL))
        self.assertEqual(RunConfig(population_pool=None, trajectory_pool=None), default)
        names = list(POP)
        config = RunConfig(population_pool=names, trajectory_pool=['SA'])
        names.append('GWO')
        self.assertEqual(config.population_pool, POP)
        self.assertEqual(config.trajectory_pool, ('SA',))
        for field, invalids in (
            ('population_pool', [(), ('GA', 'GA'), ('SA',), ('unknown',), 'GA', (1,), [['GA']]]),
            ('trajectory_pool', [(), ('SA', 'SA'), ('GA',), ('unknown',), 'SA', (1,), [['SA']]]),
        ):
            for value in invalids:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, f'^{field} must be a nonempty unique subset of '):
                        RunConfig(**{field: value})
        self.assertEqual((tuple(native.POOL_POBLACIONAL), tuple(native.POOL_TRAYECTORIA)),
                         (POPULATION_POOL, TRAJECTORY_POOL))

    def test_policy_checks_native_pool_but_selects_subset_and_preserves_lifecycle(self):
        for mode in ('uniform', 'adaptive', 'adaptive_exploratory'):
            policy = SelectionPolicy(reduced(selector=mode, seed=60))
            pop = policy.selectors['poblacional']
            traj = policy.selectors['trayectoria']
            before = (pop.scores.copy(), pop.uses.copy(), pop.rng.getstate(), random.getstate())
            for invalid in (POP, TRAJECTORY_POOL, tuple(reversed(POPULATION_POOL))):
                with self.assertRaisesRegex(ValueError, '^Policy requires native pools and one completed block per choice$'):
                    policy.choose(invalid, 'poblacional')
                self.assertEqual((pop.scores, pop.uses, pop.rng.getstate(), random.getstate()), before)
                self.assertIsNone(policy.pending)
                self.assertEqual(policy.records, [])
                self.assertEqual(policy.iterations, 0)
            with isolated_rng(60):
                chosen = policy.choose(POPULATION_POOL, 'poblacional')
            self.assertIn(chosen, POP)
            self.assertEqual(pop.names, POP)
            self.assertEqual(traj.names, TRAJ)
            pending = policy.pending
            with self.assertRaises(ValueError):
                policy.choose(TRAJECTORY_POOL, 'trayectoria')
            self.assertEqual(policy.pending, pending)
            with self.assertRaises(ValueError):
                policy.observe(chosen, 'poblacional', -math.inf, epoch(n=0), 0, 10.)
            self.assertEqual(pop.uses, dict.fromkeys(POP, 0))
            policy.observe(chosen, 'poblacional', -math.inf, epoch(n=1), 1, 10.)
            self.assertEqual(sum(pop.uses.values()), 1)
            self.assertEqual(traj.uses, dict.fromkeys(TRAJ, 0))
            self.assertEqual(policy.records[0]['probabilities_before'], dict.fromkeys(POP, 1 / 3))
            self.assertEqual(policy.records[0]['scores_after'], dict.fromkeys(POP, 0.))
            self.assertEqual(policy.records[0]['reward'], 0.)
            self.assertIsNone(policy.pending)
            self.assertEqual(policy.iterations, 1)

    def test_mocked_pipeline_records_subsets_without_changing_native_pools(self):
        config = reduced(selector='adaptive_exploratory', seed=60, max_iters=4)
        with patch.object(native, '_ejecutar_mh', side_effect=[epoch(), epoch(12.)]) as dispatch:
            result = run_pipeline(instance(), config)
        self.assertIn(dispatch.call_args_list[0].kwargs['mh_nombre'], POP)
        self.assertIn(dispatch.call_args_list[1].kwargs['mh_nombre'], TRAJ)
        self.assertEqual(result['summary']['settings'], asdict(config))
        self.assertEqual([set(r['probabilities_before']) for r in result['selection_records']],
                         [set(POP), set(TRAJ)])
        self.assertEqual((tuple(native.POOL_POBLACIONAL), tuple(native.POOL_TRAYECTORIA)),
                         (POPULATION_POOL, TRAJECTORY_POOL))

    def test_default_uniform_preserves_native_random_stream(self):
        policy = SelectionPolicy(RunConfig(selector='uniform'))
        with isolated_rng(60):
            expected = random.choice(POPULATION_POOL)
            expected_state = random.getstate()
        with isolated_rng(60):
            self.assertEqual(policy.choose(POPULATION_POOL, 'poblacional'), expected)
            self.assertEqual(random.getstate(), expected_state)

    def test_cli_dry_run_records_subsets_and_rejects_invalid_input_without_writes(self):
        from adaptive_mh_selection.run import parser_for_run
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'not-created'
            common = ['--dry-run', '--output', str(output)]
            for flags, expected in [([], (POPULATION_POOL, TRAJECTORY_POOL)),
                                    (['--population-pool', *POP, '--trajectory-pool', *TRAJ], (POP, TRAJ))]:
                args = [*common, '--selectors', 'adaptive_exploratory', '--seed', '60',
                        '--iterations', '1000', '--modes', 'ddtw', '--window', '20', *flags]
                result = self.invoke(*args)
                self.assertEqual((result.returncode, result.stderr), (0, ''), result.stderr)
                protocol = json.loads(result.stdout)
                self.assertEqual(result.stdout, json.dumps(protocol, sort_keys=True, allow_nan=False) + '\n')
                run = protocol['runs'][0]
                self.assertEqual((run['population_pool'], run['trajectory_pool']), tuple(map(list, expected)))
                self.assertEqual((run['seed'], run['selector'], run['monitor']['window'], run['use_ddtw']),
                                 (60, 'adaptive_exploratory', 20, True))
            for field, flags, message in [
                ('population_pool', ['--population-pool', 'GA', 'GA'],
                 'population_pool must be a nonempty unique subset of ' + ', '.join(POPULATION_POOL)),
                ('trajectory_pool', ['--trajectory-pool', 'GA'],
                 'trajectory_pool must be a nonempty unique subset of ' + ', '.join(TRAJECTORY_POOL)),
                ('population_pool', ['--population-pool', 'unknown'],
                 'population_pool must be a nonempty unique subset of ' + ', '.join(POPULATION_POOL)),
                ('population_pool', ['--population-pool'], 'argument --population-pool: expected at least one argument'),
            ]:
                with self.subTest(field=field, flags=flags):
                    result = self.invoke(*common, *flags)
                    self.assertEqual((result.returncode, result.stdout), (2, ''))
                    parser = parser_for_run()
                    parser.prog = (f'{Path(sys.executable).name} -m adaptive_mh_selection.run'
                                   if sys.version_info >= (3, 14) else 'run.py')
                    self.assertEqual(result.stderr, parser.format_usage() + f'{parser.prog}: error: {message}\n')
            self.assertEqual(list(Path(tmp).iterdir()), [])
        help_result = self.invoke('--help')
        self.assertEqual((help_result.returncode, help_result.stderr), (0, ''))
        self.assertIn('--population-pool', help_result.stdout)
        self.assertIn('--trajectory-pool', help_result.stdout)

    def invoke(self, *args):
        return subprocess.run([sys.executable, '-B', '-m', 'adaptive_mh_selection.run', *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)

    def test_plots_only_six_recorded_mhs_and_keeps_missing_pool_explicit(self):
        from adaptive_mh_selection.plotting import plot_selection_probabilities, plot_run
        records = [decision(0, 'poblacional', 'GA', 0, dict.fromkeys(POP, 1 / 3)),
                   decision(1, 'trayectoria', 'SA', 10, dict.fromkeys(TRAJ, 1 / 3))]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            _, figures = capture_probability_figure(lambda: plot_selection_probabilities(records, output))
            self.assertEqual(len(figures[0]['axes']), 6)
            self.assertEqual({a['title'].split(' — ')[1] for a in figures[0]['axes']}, set(POP + TRAJ))
            for axis in figures[0]['axes']:
                self.assertEqual(axis['lines'][0]['x'], [1.])
                self.assertAlmostEqual(axis['lines'][0]['y'][0], 100 / 3)
            self.assertTrue((output / 'selection_probabilities.png').is_file())
            self.assertTrue((output / 'selection_probabilities.pdf').is_file())
            metadata = {'settings': asdict(reduced())}
            _, figures = capture_probability_figure(
                lambda: plot_run(native_result(), output / 'missing', metadata,
                                 selection_records=records[:1]))
            self.assertEqual(len(figures[0]['axes']), 6)
            missing = [a for a in figures[0]['axes'] if a['title'].startswith('Trayectoria')]
            self.assertEqual(len(missing), 3)
            self.assertTrue(all(not a['lines'] and a['texts'] == ['No recorded decisions'] for a in missing))
            self.assertIsNone(plot_selection_probabilities([{'pool': 'poblacional', 'solver': 'GA'}],
                                                         output / 'absent'))
            self.assertFalse((output / 'absent').exists())
