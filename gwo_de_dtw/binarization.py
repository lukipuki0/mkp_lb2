"""
Binarizations of the GWO-DE hybrid: LB2 (default) and sigmoid.

LB2 (port of lb2/binarization.py::binarizar_posicion).

The real vector is not turned into bits from scratch: it gives, per item, the
probability of flipping the individual's current bit.

    L1 = clip(-G1 * v / (v_max - G2) + G3, 0, 1)
    L2 = clip( G1 * v / (v_max - G2) + G3, 0, 1)

Two candidates are built from the current solution (one flipped with L1, one
with L2), both are repaired, and the better one is returned. On a tie L2 wins,
as in the original (`val_1 > val_2`). Each call costs two evaluations.

Differences with the original: flips use the run's numpy Generator instead of
the global `np.random` (reproducible per seed), and repair is
`gwo_de_dtw.problem.reparar`.

Sigmoid: each bit is sampled with P(bit = 1) = 1 / (1 + exp(-v)) using the
run's numpy Generator, then repaired. Each call costs one evaluation.

Spec: odd/tasks/gwo-de-dtw-hybrid.md (S11, S14).
"""

from typing import Tuple

import numpy as np

from .problem import reparar

LB2_EVALUATIONS = 2
SIGMOID_EVALUATIONS = 1


def lb2_probabilities(
    v: np.ndarray, v_max: float, G1: float, G2: float, G3: float
) -> Tuple[np.ndarray, np.ndarray]:
    denom = v_max - G2
    if abs(denom) < 1e-12:
        denom = 1e-12
    L1 = np.clip(-G1 * v / denom + G3, 0.0, 1.0)
    L2 = np.clip(G1 * v / denom + G3, 0.0, 1.0)
    return L1, L2


def lb2_binarize(
    v: np.ndarray,
    current: np.ndarray,
    inst: dict,
    rng: np.random.Generator,
    G1: float,
    G2: float,
    G3: float,
    v_max: float,
) -> Tuple[np.ndarray, float]:
    """Return the better repaired LB2 candidate (L2 on ties)."""
    L1, L2 = lb2_probabilities(v, v_max, G1, G2, G3)
    current = np.asarray(current, dtype=int)

    x1 = np.where(rng.random(current.shape[0]) < L1, 1 - current, current)
    x2 = np.where(rng.random(current.shape[0]) < L2, 1 - current, current)

    x1, val_1 = reparar(x1, inst)
    x2, val_2 = reparar(x2, inst)

    if val_1 > val_2:
        return x1, val_1
    return x2, val_2


def sigmoid_binarize(
    v: np.ndarray, inst: dict, rng: np.random.Generator
) -> Tuple[np.ndarray, float]:
    """Sample bits with P(1) = sigmoid(v) and return them repaired."""
    prob = 1.0 / (1.0 + np.exp(-np.asarray(v, dtype=float)))
    bits = (rng.random(prob.shape[0]) < prob).astype(int)
    return reparar(bits, inst)
