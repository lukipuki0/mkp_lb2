"""GWO and DE/rand/1/bin moves on clipped real vectors."""

import numpy as np


def gwo_candidate(x, leaders, a, rng, v_max):
    """Mean of the three leader-guided positions, clipped to [-v_max, v_max]."""
    guided = np.empty((3, x.shape[0]))
    for leader in range(3):
        coefficient_a = 2.0 * a * rng.random(x.shape[0]) - a
        coefficient_c = 2.0 * rng.random(x.shape[0])
        distance = np.abs(coefficient_c * leaders[leader] - x)
        guided[leader] = leaders[leader] - coefficient_a * distance
    return np.clip(guided.mean(axis=0), -v_max, v_max)


def de_candidate(index, population, factor, crossover, rng, v_max):
    """DE/rand/1/bin. Reads ``population`` at call time, so in-place updates are visible."""
    others = [row for row in range(population.shape[0]) if row != index]
    base, left, right = rng.choice(others, 3, replace=False)
    mutant = np.clip(population[base] + factor * (population[left] - population[right]),
                     -v_max, v_max)
    mask = rng.random(population.shape[1]) < crossover
    mask[int(rng.integers(0, population.shape[1]))] = True
    return np.where(mask, mutant, population[index])
