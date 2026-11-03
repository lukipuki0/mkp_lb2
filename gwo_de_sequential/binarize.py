"""LB2 (default) and sigmoid maps from a real vector to a repaired bit string."""

import numpy as np

from .problem import reparar


def lb2_binarize(real, current, inst, rng, g1, g2, g3, v_max):
    """Flip current bits with L1 and L2, repair both, keep the better. L2 wins ties."""
    slope = v_max - g2
    if abs(slope) < 1e-12:
        slope = 1e-12
    low = np.clip(-g1 * real / slope + g3, 0.0, 1.0)
    high = np.clip(g1 * real / slope + g3, 0.0, 1.0)
    current = np.asarray(current, dtype=int)
    first = np.where(rng.random(current.shape[0]) < low, 1 - current, current)
    second = np.where(rng.random(current.shape[0]) < high, 1 - current, current)
    bits_low, fit_low = reparar(first, inst)
    bits_high, fit_high = reparar(second, inst)
    if fit_low > fit_high:
        return bits_low, fit_low
    return bits_high, fit_high


def sigmoid_binarize(real, inst, rng):
    """Sample each bit with probability sigmoid(v), then repair. One evaluation."""
    probability = 1.0 / (1.0 + np.exp(-np.asarray(real, dtype=float)))
    bits = (rng.random(probability.shape[0]) < probability).astype(int)
    return reparar(bits, inst)
