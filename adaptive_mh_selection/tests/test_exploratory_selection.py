"""Opt-in exploration, public dry-runs and unchanged baseline contracts."""
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from adaptive_mh_selection import selection
from adaptive_mh_selection.config import RunConfig
from adaptive_mh_selection.orchestrator import SelectionPolicy

ROOT = Path(__file__).resolve().parents[2]


class ExploratorySelectionTests(unittest.TestCase):
    def make_selector(self, names=('A', 'B', 'C'), **kwargs):
        return selection.AdaptiveExploratorySelector(names, random.Random(17), **kwargs)

    def assert_distribution(self, probabilities):
        self.assertTrue(all(math.isfinite(p) and p > 0 for p in probabilities.values()))
        self.assertAlmostEqual(sum(probabilities.values()), 1.)

    def test_zero_scores_use_inverse_square_root_counts_not_uniform(self):
        selector = self.make_selector()
        self.assertEqual(selector.probabilities(), dict.fromkeys(selector.names, 1 / 3))
        for name, count in [('A', 8), ('B', 3)]:
            for _ in range(count):
                selector.update(name, 0.)
        probabilities = selector.probabilities()
        weights = {'A': 1 / 3, 'B': 1 / 2, 'C': 1.}
        for name, weight in weights.items():
            self.assertAlmostEqual(probabilities[name], weight / sum(weights.values()))
        self.assertEqual(selector.scores, dict.fromkeys(selector.names, 0.))
        self.assertEqual(selector.uses, {'A': 8, 'B': 3, 'C': 0})
        self.assert_distribution(probabilities)

    def test_no_reward_reduces_relative_probability_with_and_without_scores(self):
        for initial_reward in (0., 1.):
            with self.subTest(initial_reward=initial_reward):
                selector = self.make_selector()
                selector.update('A', initial_reward)
                before = selector.probabilities()
                scores = selector.scores.copy()
                selector.update('A', 0.)
                after = selector.probabilities()
                self.assertLess(after['A'], before['A'])
                for other in ('B', 'C'):
                    self.assertGreater(after[other], before[other])
                    self.assertLess(after['A'] / after[other], before['A'] / before[other])
                    self.assertEqual(selector.scores[other], scores[other])
                self.assertAlmostEqual(selector.scores['A'], .7 * scores['A'])
                self.assertEqual(selector.uses, {'A': 2, 'B': 0, 'C': 0})
                self.assert_distribution(after)

    def test_positive_scores_mix_normalized_exploitation_and_count_exploration(self):
        selector = self.make_selector(alpha=.4, epsilon=.25)
        selector.update('A', 1.)
        selector.update('A', 0.)
        selector.update('B', .5)
        scores = {'A': .24, 'B': .2, 'C': 0.}
        weights = {'A': 1 / math.sqrt(3), 'B': 1 / math.sqrt(2), 'C': 1.}
        for name, probability in selector.probabilities().items():
            self.assertAlmostEqual(selector.scores[name], scores[name])
            expected = .25 * weights[name] / sum(weights.values()) + .75 * scores[name] / .44
            self.assertAlmostEqual(probability, expected)
        self.assert_distribution(selector.probabilities())

    def test_boundaries_remain_positive_and_normalized(self):
        for names in [('A',), ('A', 'B', 'C')]:
            for alpha in (.3, 1.):
                for epsilon in (math.ulp(0.), .2, 1.):
                    with self.subTest(names=names, alpha=alpha, epsilon=epsilon):
                        selector = self.make_selector(names, alpha=alpha, epsilon=epsilon)
                        selector.update('A', 1.)
                        self.assert_distribution(selector.probabilities())
                        for _ in range(40):
                            selector.update('A', 0.)
                        self.assert_distribution(selector.probabilities())
                        if epsilon == 1.:
                            self.assertAlmostEqual(selector.probabilities()['A'],
                                                   (1 / math.sqrt(42)) /
                                                   (1 / math.sqrt(42) + len(names) - 1))

    def test_validation_is_inherited_and_rejected_updates_leave_state_unchanged(self):
        for names in ((), ('A', 'A')):
            with self.assertRaisesRegex(ValueError, '^A nonempty unique solver pool is required$'):
                self.make_selector(names)
        for field in ('alpha', 'epsilon'):
            for value in (0., -1., 2., math.nan, math.inf):
                with self.assertRaisesRegex(ValueError, rf'^{field} must be finite and in \(0, 1\]$'):
                    self.make_selector(**{field: value})
        selector = self.make_selector()
        selector.update('A', .5)
        before = (selector.scores.copy(), selector.uses.copy(), selector.probabilities())
        for name, reward, message in [
            ('missing', 0., 'Unknown solver: missing'),
            *[('A', value, r'reward must be finite and in \[0, 1\]')
              for value in (-1., 2., math.nan, math.inf)],
        ]:
            with self.assertRaisesRegex(ValueError, '^' + message + '$'):
                selector.update(name, reward)
            self.assertEqual((selector.scores, selector.uses, selector.probabilities()), before)

    def test_choices_are_reproducible_without_consuming_global_rng(self):
        a, b = self.make_selector(), self.make_selector()
        before = random.getstate()
        for reward in (0., .5, 0., 1.) * 5:
            chosen, probabilities = a.choose()
            self.assertEqual((chosen, probabilities), b.choose())
            self.assertEqual(probabilities, a.probabilities())
            a.update(chosen, reward)
            b.update(chosen, reward)
        self.assertEqual(random.getstate(), before)

    def test_policy_reuses_completed_block_lifecycle_and_isolates_pools_and_runs(self):
        config = RunConfig(selector='adaptive_exploratory', alpha=.4, epsilon=.25)
        policy, fresh = SelectionPolicy(config), SelectionPolicy(config)
        pop, trajectory = policy.selectors['poblacional'], policy.selectors['trayectoria']
        self.assertIsInstance(pop, selection.AdaptiveExploratorySelector)
        self.assertEqual((pop.alpha, pop.epsilon), (.4, .25))
        untouched = trajectory.probabilities()
        chosen = policy.choose(pop.names, 'poblacional')
        before = pop.probabilities()
        self.assertEqual(sum(pop.uses.values()), 0)
        policy.observe(chosen, 'poblacional', -math.inf,
                       SimpleNamespace(historial=[10.]), 1, 10.)
        self.assertLess(pop.probabilities()[chosen], before[chosen])
        self.assertEqual(sum(pop.uses.values()), 1)
        self.assertEqual(trajectory.probabilities(), untouched)
        self.assertEqual(sum(trajectory.uses.values()), 0)
        self.assertEqual(sum(fresh.selectors['poblacional'].uses.values()), 0)
        self.assertEqual(policy.records[0]['probabilities_before'], before)
        self.assertEqual(policy.records[0]['reward'], 0.)
        self.assertTrue(policy.records[0]['warmup'])
        self.assertIsNone(policy.pending)
        self.assertEqual(policy.iterations, 1)

    def test_baseline_policy_classes_and_no_reward_behavior_are_unchanged(self):
        for name, cls in [('adaptive', selection.AdaptiveSelector), ('uniform', selection.UniformSelector)]:
            policy = SelectionPolicy(RunConfig(selector=name))
            self.assertTrue(all(type(selector) is cls for selector in policy.selectors.values()))
            selector = cls(('A', 'B'), random.Random(17))
            selector.update('A', 1.)
            before = selector.probabilities()
            selector.update('A', 0.)
            self.assertEqual(selector.probabilities(), before)
            self.assertEqual(selector.uses, {'A': 2, 'B': 0})
            if name == 'adaptive':
                self.assertEqual(before, {'A': .9, 'B': .1})
                self.assertAlmostEqual(selector.scores['A'], .21)
            else:
                self.assertEqual(before, {'A': .5, 'B': .5})
                self.assertEqual(selector.scores, {'A': 0., 'B': 0.})


class SelectorCLITests(unittest.TestCase):
    def invoke(self, module, *args):
        return subprocess.run([sys.executable, '-B', '-m', f'adaptive_mh_selection.{module}',
                               *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=60)

    def test_public_dry_runs_opt_in_and_preserve_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'not-created'
            for module in ('run', 'campaign'):
                extra = ['--families', '1', '--indices', '0', '--runs', '1'] if module == 'campaign' else []
                common = ['--dry-run', '--output', output, *extra]
                for selectors in ([], ['uniform', 'adaptive'],
                                  ['adaptive_exploratory', 'uniform', 'adaptive']):
                    with self.subTest(module=module, selectors=selectors):
                        result = self.invoke(module, *common, *(['--selectors', *selectors] if selectors else []))
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stderr, '')
                        protocol = json.loads(result.stdout)
                        self.assertEqual(result.stdout, json.dumps(protocol, sort_keys=True, allow_nan=False) + '\n')
                        expected = selectors or ['adaptive']
                        if module == 'run':
                            self.assertEqual([run['selector'] for run in protocol['runs']], expected)
                            for run in protocol['runs']:
                                self.assertEqual((run['seed'], run['max_iters'], run['alpha'], run['epsilon'],
                                                  run['use_ddtw'], run['monitor']['window']),
                                                 (42, 1000, .3, .2, True, 70))
                        else:
                            expected = [name for name in ('adaptive', 'uniform', 'adaptive_exploratory') if name in expected]
                            self.assertEqual(protocol['selectors'], expected)
                            self.assertEqual(protocol['conditions'], [name + '_ddtw' for name in expected])
                            self.assertEqual(protocol['planned_runs'], len(expected))
                            self.assertEqual(protocol['seeds'], [42])
                            self.assertEqual((protocol['iterations'], protocol['alpha'], protocol['epsilon'],
                                              protocol['monitor']['window']), (1000, .3, .2, 70))
                        self.assertFalse(output.exists())
                self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_explicit_protocol_parameters_survive_campaign_reconstruction(self):
        from adaptive_mh_selection.campaign import make_config

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'not-created'
            configs = []
            for module in ('run', 'campaign'):
                extra = ['--families', '1', '--indices', '0'] if module == 'campaign' else []
                result = self.invoke(module, '--dry-run', '--output', output, *extra,
                                     '--selectors', 'adaptive_exploratory', '--runs', '1',
                                     '--seed', '60', '--iterations', '1000', '--modes', 'ddtw',
                                     '--window', '20', '--alpha', '.4', '--epsilon', '.25')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, '')
                protocol = json.loads(result.stdout)
                if module == 'campaign':
                    config = make_config(protocol, 60, protocol['selectors'][0], 'ddtw')
                else:
                    from dtw_stagnation import StagnationConfig
                    fields = protocol['runs'][0]
                    fields['monitor'] = StagnationConfig(**fields['monitor'])
                    config = RunConfig(**fields)
                configs.append(config)
                self.assertEqual((config.selector, config.seed, config.max_iters, config.alpha,
                                  config.epsilon, config.use_ddtw, config.monitor.window),
                                 ('adaptive_exploratory', 60, 1000, .4, .25, True, 20))
                self.assertIsNone(config.max_epoch_iters)
                self.assertFalse(output.exists())
            self.assertEqual(configs[0], configs[1])
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_public_help_names_opt_in_and_exploration_mixture(self):
        for module in ('run', 'campaign'):
            with self.subTest(module=module):
                result = self.invoke(module, '--help')
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stderr, '')
                self.assertIn('adaptive_exploratory', result.stdout)
                self.assertIn('Exploration mixture', result.stdout)
