"""Improvement per native iteration, scaled by the first completed block."""

import math
from .config import positive_integer


def compute_reward(fitness_before, fitness_after, first_block_fitness, n_iters):
    """Return (gain, raw_reward, reward) for a subsequent MKP block.

    The policy handles the first block separately: inherited best is -inf, so it
    establishes the fixed scale and receives zero reward (warmup).
    """
    if not all(math.isfinite(value) for value in (fitness_before, fitness_after, first_block_fitness)):
        raise ValueError('Fitness values must be finite')
    positive_integer(n_iters, 'n_iters')
    gain = max(0., fitness_after - fitness_before)
    raw_reward = gain / max(1., abs(first_block_fitness)) / n_iters
    if not math.isfinite(gain) or not math.isfinite(raw_reward):
        raise ValueError('Reward calculation overflowed')
    return float(gain), float(raw_reward), float(min(1., raw_reward))
