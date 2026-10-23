import math
import unittest


class RewardTests(unittest.TestCase):
    def test_fixed_first_block_scale_native_iterations_and_saturation(self):
        from adaptive_mh_selection.reward import compute_reward
        self.assertEqual(compute_reward(1000., 1100., 1000., 200), (100., .0005, .0005))
        self.assertEqual(compute_reward(2000., 2100., 1000., 400), (100., .00025, .00025))
        self.assertEqual(compute_reward(1100., 1000., 1000., 200), (0., 0., 0.))
        self.assertEqual(compute_reward(0., 100., 0., 1), (100., 100., 1.))

    def test_nonfinite_and_invalid_iterations_are_rejected(self):
        from adaptive_mh_selection.reward import compute_reward
        for bad in (math.nan, math.inf, -math.inf):
            for args in ((bad, 2., 1., 1), (1., bad, 1., 1), (1., 2., bad, 1)):
                with self.assertRaises(ValueError):
                    compute_reward(*args)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                compute_reward(1., 2., 1., count)
