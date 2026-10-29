"""
Differential Evolution operator DE/rand/1/bin on real vectors.

For individual i, three distinct individuals r1, r2, r3 (all different from i)
build the mutant

    v = x_r1 + F * (x_r2 - x_r3),   clipped to [-v_max, v_max]

and binomial crossover takes each coordinate from v with probability CR,
otherwise from x_i. One random coordinate (j_rand) always comes from v, so the
candidate differs from x_i in at least one position.

Spec: odd/tasks/gwo-de-dtw-hybrid.md (S2).
"""

import numpy as np


def de_candidate(
    i: int,
    population: np.ndarray,
    F: float,
    CR: float,
    rng: np.random.Generator,
    v_max: float,
) -> np.ndarray:
    """One DE/rand/1/bin candidate for row `i` of `population`."""
    pop_size, n = population.shape
    others = [j for j in range(pop_size) if j != i]
    r1, r2, r3 = rng.choice(others, 3, replace=False)
    v = population[r1] + F * (population[r2] - population[r3])
    v = np.clip(v, -v_max, v_max)
    mask = rng.random(n) < CR
    mask[rng.integers(0, n)] = True
    return np.where(mask, v, population[i])
