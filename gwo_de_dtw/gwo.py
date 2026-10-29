"""
Grey Wolf Optimizer (GWO) operator on real vectors.

The individual x is pulled towards each of the three leaders (alpha, beta,
delta). For leader X_k:

    A = 2a * r1 - a,   C = 2 * r2,   D = |C * X_k - x|,   X_k' = X_k - A * D

with r1, r2 uniform in [0, 1] per coordinate. The candidate is the mean of the
three guided positions, clipped to [-v_max, v_max]. Large `a` (|A| > 1) moves
away from the leaders (exploration); small `a` stays close (exploitation).

Spec: odd/tasks/gwo-de-dtw-hybrid.md (S2).
"""

import numpy as np


def gwo_candidate(
    x: np.ndarray,
    leaders: np.ndarray,
    a: float,
    rng: np.random.Generator,
    v_max: float,
) -> np.ndarray:
    """One GWO candidate for `x` guided by the three rows of `leaders`."""
    n = x.shape[0]
    guided = np.empty((3, n))
    for k in range(3):
        A = 2.0 * a * rng.random(n) - a
        C = 2.0 * rng.random(n)
        D = np.abs(C * leaders[k] - x)
        guided[k] = leaders[k] - A * D
    return np.clip(guided.mean(axis=0), -v_max, v_max)
