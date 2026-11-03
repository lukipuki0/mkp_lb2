import math
import random
import unittest

from adaptive_mh_selection.selection import AdaptiveSelector, UniformSelector


class SelectionTests(unittest.TestCase):
    def test_uniform_start_ewma_and_exploration(self):
        selector = AdaptiveSelector(('A', 'B'), random.Random(1))
        self.assertEqual(selector.probabilities(), {'A': .5, 'B': .5})
        selector.update('A', .5)
        self.assertEqual(selector.scores, {'A': .15, 'B': 0.})
        self.assertEqual(selector.uses, {'A': 1, 'B': 0})
        self.assertAlmostEqual(selector.probabilities()['A'], .9)
        selector.update('A', 0.)
        self.assertAlmostEqual(selector.scores['A'], .105)
        self.assertAlmostEqual(sum(selector.probabilities().values()), 1.)
        self.assertGreaterEqual(min(selector.probabilities().values()), .1)

    def test_uniform_control_and_independent_reproducible_streams(self):
        a = AdaptiveSelector(('A', 'B'), random.Random(17))
        b = AdaptiveSelector(('A', 'B'), random.Random(17))
        self.assertEqual([a.choose() for _ in range(20)], [b.choose() for _ in range(20)])
        a.update('A', 1.)
        self.assertEqual(b.scores['A'], 0.)
        uniform = UniformSelector(('A', 'B'), random.Random(1))
        uniform.update('A', 1.)
        self.assertEqual(uniform.probabilities(), {'A': .5, 'B': .5})
        self.assertEqual(uniform.scores, {'A': 0., 'B': 0.})
        self.assertEqual(uniform.uses['A'], 1)

    def test_invalid_inputs_do_not_mutate(self):
        for pool in [(), ('A', 'A')]:
            with self.assertRaises(ValueError):
                AdaptiveSelector(pool, random.Random(1))
        for field in ('alpha', 'epsilon'):
            for value in (0., -1., 2., math.nan, math.inf):
                with self.assertRaises(ValueError):
                    AdaptiveSelector(('A',), random.Random(1), **{field: value})
        for cls in (AdaptiveSelector, UniformSelector):
            selector = cls(('A',), random.Random(1))
            for value in (-1., 2., math.nan, math.inf):
                with self.assertRaises(ValueError):
                    selector.update('A', value)
            self.assertEqual(selector.scores, {'A': 0.})
            self.assertEqual(selector.uses, {'A': 0})
