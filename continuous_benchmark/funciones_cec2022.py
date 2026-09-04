"""IEEE CEC 2022 real-parameter benchmark suite.

The implementation follows the official Python reference distributed in
``2022-SO-BO`` and uses its shift, rotation and shuffle files from
``continuous_benchmark/input_data``.  The files are deliberately required:
silently generating replacement matrices would make results incomparable
with CEC2022 publications.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), "input_data")
SUPPORTED_DIMENSIONS = (2, 10, 20)
CEC_BIASES = {
    1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0,
    6: 1800.0, 7: 2000.0, 8: 2200.0, 9: 2300.0,
    10: 2400.0, 11: 2600.0, 12: 2700.0,
}
# Backwards-compatible public name used by the existing verification script.
OFFICIAL_BIASES = CEC_BIASES


@dataclass
class ContinuousFunction:
    """Descriptor consumed by the continuous metaheuristics."""

    name: str
    func: Callable[[np.ndarray], float]
    lb: float
    ub: float
    optimum: float
    n_dim: int


# ---------------------------------------------------------------------------
# Mathematical functions used by the official CEC implementation
# ---------------------------------------------------------------------------

def _zakharov(z: np.ndarray) -> float:
    i = np.arange(1, z.size + 1, dtype=float)
    s2 = np.sum(0.5 * i * z)
    return float(np.sum(z * z) + s2**2 + s2**4)


def _ellips(z: np.ndarray) -> float:
    if z.size <= 1:
        return float(np.sum(z * z))
    exponents = 6.0 * np.arange(z.size) / (z.size - 1)
    return float(np.sum(10.0**exponents * z * z))


def _bent_cigar(z: np.ndarray) -> float:
    return float(z[0] ** 2 + 1.0e6 * np.sum(z[1:] ** 2))


def _discus(z: np.ndarray) -> float:
    return float(1.0e6 * z[0] ** 2 + np.sum(z[1:] ** 2))


def _rosenbrock(z: np.ndarray) -> float:
    y = z + 1.0
    if y.size <= 1:
        return 0.0
    return float(np.sum(100.0 * (y[:-1] ** 2 - y[1:]) ** 2 + (y[:-1] - 1.0) ** 2))


def _ackley(z: np.ndarray) -> float:
    n = z.size
    return float(
        np.e - 20.0 * np.exp(-0.2 * np.sqrt(np.sum(z * z) / n))
        - np.exp(np.sum(np.cos(2.0 * np.pi * z)) / n) + 20.0
    )


def _griewank(z: np.ndarray) -> float:
    i = np.arange(1, z.size + 1, dtype=float)
    return float(1.0 + np.sum(z * z) / 4000.0 - np.prod(np.cos(z / np.sqrt(i))))


def _rastrigin(z: np.ndarray) -> float:
    return float(np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))


def _step_rastrigin(z: np.ndarray) -> float:
    # Official reference rule: floor rather than np.round at half-integers.
    stepped = np.where(np.abs(z) > 0.5, np.floor(2.0 * z + 0.5) / 2.0, z)
    return _rastrigin(stepped)


def _schwefel(z: np.ndarray) -> float:
    f = 0.0
    for value in z + 4.209687462275036e2:
        if value > 500.0:
            remainder = np.fmod(value, 500.0)
            f -= (500.0 - remainder) * np.sin(np.sqrt(500.0 - remainder))
            f += ((value - 500.0) / 100.0) ** 2 / z.size
        elif value < -500.0:
            remainder = np.fmod(abs(value), 500.0)
            f -= (-500.0 + remainder) * np.sin(np.sqrt(500.0 - remainder))
            f += ((value + 500.0) / 100.0) ** 2 / z.size
        else:
            f -= value * np.sin(np.sqrt(abs(value)))
    return float(f + 4.189828872724338e2 * z.size)


def _grie_rosen(z: np.ndarray) -> float:
    y = z + 1.0
    if y.size <= 1:
        return 0.0
    total = 0.0
    for i in range(y.size - 1):
        temp = 100.0 * (y[i] ** 2 - y[i + 1]) ** 2 + (y[i] - 1.0) ** 2
        total += temp * temp / 4000.0 - np.cos(temp) + 1.0
    temp = 100.0 * (y[-1] ** 2 - y[0]) ** 2 + (y[-1] - 1.0) ** 2
    total += temp * temp / 4000.0 - np.cos(temp) + 1.0
    return float(total)


def _expanded_schaffer_f6(z: np.ndarray) -> float:
    if z.size <= 1:
        return 0.0
    pair_sq = np.concatenate((z[:-1] ** 2 + z[1:] ** 2, [z[-1] ** 2 + z[0] ** 2]))
    return float(np.sum(0.5 + (np.sin(np.sqrt(pair_sq)) ** 2 - 0.5) / (1.0 + 0.001 * pair_sq) ** 2))


def _schaffer_f7(z: np.ndarray) -> float:
    if z.size <= 1:
        return 0.0
    radius = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
    terms = np.sqrt(radius) * (1.0 + np.sin(50.0 * radius**0.2) ** 2)
    return float(np.sum(terms) ** 2 / (z.size - 1) ** 2)


def _happycat(z: np.ndarray) -> float:
    y = z - 1.0
    r2 = np.sum(y * y)
    return float(abs(r2 - z.size) ** 0.25 + (0.5 * r2 + np.sum(y)) / z.size + 0.5)


def _hgbat(z: np.ndarray) -> float:
    y = z - 1.0
    r2 = np.sum(y * y)
    sum_y = np.sum(y)
    return float(abs(r2**2 - sum_y**2) ** 0.5 + (0.5 * r2 + sum_y) / z.size + 0.5)


def _katsuura(z: np.ndarray) -> float:
    product = 1.0
    for i, value in enumerate(z, start=1):
        term = 0.0
        for j in range(1, 33):
            power = 2.0**j
            term += abs(power * value - np.floor(power * value + 0.5)) / power
        product *= (1.0 + i * term) ** (10.0 / (z.size**1.2))
    scale = 10.0 / (z.size * z.size)
    return float(product * scale - scale)


# Public standalone helpers retained for compatibility with earlier code.
def bent_cigar(x: np.ndarray) -> float:
    return _bent_cigar(np.asarray(x, dtype=float))


def discus(x: np.ndarray) -> float:
    return _discus(np.asarray(x, dtype=float))


def zakharov(x: np.ndarray) -> float:
    return _zakharov(np.asarray(x, dtype=float))


def rosenbrock(x: np.ndarray) -> float:
    return _rosenbrock(np.asarray(x, dtype=float) * (2.048 / 100.0))


def schaffer_f6(x_i: float, x_j: float) -> float:
    q = x_i * x_i + x_j * x_j
    return float(0.5 + (np.sin(np.sqrt(q)) ** 2 - 0.5) / (1.0 + 0.001 * q) ** 2)


def expanded_schaffer_f6(x: np.ndarray) -> float:
    return _expanded_schaffer_f6(np.asarray(x, dtype=float))


def rastrigin(x: np.ndarray) -> float:
    return _rastrigin(np.asarray(x, dtype=float) * (5.12 / 100.0))


def non_continuous_rastrigin(x: np.ndarray) -> float:
    return _step_rastrigin(np.asarray(x, dtype=float) * (5.12 / 100.0))


def levy(x: np.ndarray) -> float:
    z = np.asarray(x, dtype=float)
    w = 1.0 + z / 4.0
    return float(
        np.sin(np.pi * w[0]) ** 2
        + np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
        + (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    )


def ackley(x: np.ndarray) -> float:
    return _ackley(np.asarray(x, dtype=float))


def griewank(x: np.ndarray) -> float:
    return _griewank(np.asarray(x, dtype=float) * (600.0 / 100.0))


def schwefel(x: np.ndarray) -> float:
    return _schwefel(np.asarray(x, dtype=float) * (1000.0 / 100.0))


def katsuura(x: np.ndarray) -> float:
    return _katsuura(np.asarray(x, dtype=float) * (5.0 / 100.0))


def happycat(x: np.ndarray) -> float:
    return _happycat(np.asarray(x, dtype=float) * (5.0 / 100.0))


def hgbat(x: np.ndarray) -> float:
    return _hgbat(np.asarray(x, dtype=float) * (5.0 / 100.0))


def grie_rosen(x: np.ndarray) -> float:
    return _grie_rosen(np.asarray(x, dtype=float) * (5.0 / 100.0))


# ---------------------------------------------------------------------------
# Official data loading and transformations
# ---------------------------------------------------------------------------

def shift(x: np.ndarray, o: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) - np.asarray(o, dtype=float)


def rotate(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float) @ np.asarray(x, dtype=float)


def _transform(
    x: np.ndarray,
    o: Optional[np.ndarray],
    matrix: Optional[np.ndarray],
    rate: float,
    rotate_flag: bool = True,
) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    if o is not None:
        z = z - np.asarray(o, dtype=float)
    z = z * rate
    if rotate_flag and matrix is not None:
        z = np.asarray(matrix, dtype=float) @ z
    return z


def _read_required(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Falta el archivo oficial de CEC2022: {path}. "
            "Instala los archivos de input_data antes de ejecutar el benchmark."
        )
    return np.loadtxt(path)


def get_shift_matrix_data(
    func_num: int, n_dim: int, num_subfuncs: int = 1
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load the official shifts and rotation matrices for a function."""
    raw_shift = np.asarray(_read_required(os.path.join(DATA_DIR, f"shift_data_{func_num}.txt")), dtype=float)
    raw_matrix = np.asarray(_read_required(os.path.join(DATA_DIR, f"M_{func_num}_D{n_dim}.txt")), dtype=float)
    if num_subfuncs == 1:
        return [raw_shift.reshape(-1)[:n_dim]], [raw_matrix.reshape(-1, n_dim)[:n_dim, :n_dim]]
    shift_rows = raw_shift.reshape(-1, 100)[:num_subfuncs, :n_dim]
    matrix_rows = raw_matrix.reshape(-1, n_dim)
    shifts = [row.copy() for row in shift_rows]
    matrices = [matrix_rows[i * n_dim : (i + 1) * n_dim, :n_dim].copy() for i in range(num_subfuncs)]
    return shifts, matrices


def _get_shuffle_data(func_num: int, n_dim: int) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"shuffle_data_{func_num}_D{n_dim}.txt")
    return np.asarray(_read_required(path), dtype=int).reshape(-1)


def _validate_dimension(func_num: int, n_dim: int) -> None:
    if n_dim not in SUPPORTED_DIMENSIONS:
        raise ValueError(f"CEC2022 solo define D={SUPPORTED_DIMENSIONS}; se recibió D={n_dim}.")
    if func_num in (6, 7, 8) and n_dim == 2:
        raise ValueError(f"CEC2022 F{func_num} no está definida para D=2.")


# ---------------------------------------------------------------------------
# CEC2022 functions
# ---------------------------------------------------------------------------

_RATES = {
    "zakharov": 1.0, "rosenbrock": 2.048 / 100.0, "schaffer_f7": 1.0,
    "step_rastrigin": 5.12 / 100.0, "levy": 1.0, "bent_cigar": 1.0,
    "hgbat": 5.0 / 100.0, "rastrigin": 5.12 / 100.0, "katsuura": 5.0 / 100.0,
    "ackley": 1.0, "schwefel": 1000.0 / 100.0, "happycat": 5.0 / 100.0,
    "grie_rosen": 5.0 / 100.0, "escaffer6": 1.0, "griewank": 600.0 / 100.0,
    "ellips": 1.0, "discus": 1.0,
}

_RAW_FUNCTIONS = {
    "zakharov": _zakharov, "rosenbrock": _rosenbrock, "schaffer_f7": _schaffer_f7,
    "step_rastrigin": _step_rastrigin, "levy": levy, "bent_cigar": _bent_cigar,
    "hgbat": _hgbat, "rastrigin": _rastrigin, "katsuura": _katsuura,
    "ackley": _ackley, "schwefel": _schwefel, "happycat": _happycat,
    "grie_rosen": _grie_rosen, "escaffer6": _expanded_schaffer_f6,
    "griewank": _griewank, "ellips": _ellips, "discus": _discus,
}


def _component(
    name: str,
    x: np.ndarray,
    o: Optional[np.ndarray] = None,
    matrix: Optional[np.ndarray] = None,
    rotate_flag: bool = True,
) -> float:
    z = _transform(x, o, matrix, _RATES[name], rotate_flag)
    return float(_RAW_FUNCTIONS[name](z))


def _hybrid(
    x: np.ndarray,
    o: np.ndarray,
    matrix: np.ndarray,
    shuffle_data: np.ndarray,
    names: Sequence[str],
    proportions: Sequence[float],
) -> float:
    z = _transform(x, o, matrix, 1.0)
    shuffled = z[shuffle_data.astype(int) - 1]
    sizes = [int(np.ceil(p * z.size)) for p in proportions[:-1]]
    sizes.append(z.size - sum(sizes))
    total = 0.0
    start = 0
    for name, size in zip(names, sizes):
        total += _component(name, shuffled[start : start + size])
        start += size
    return float(total)


def _composition(
    x: np.ndarray,
    shifts: Sequence[np.ndarray],
    matrices: Sequence[np.ndarray],
    names: Sequence[str],
    deltas: Sequence[float],
    component_biases: Sequence[float],
    normalizers: Sequence[float],
    rotate_flags: Optional[Sequence[bool]] = None,
) -> float:
    if rotate_flags is None:
        rotate_flags = [True] * len(names)
    fits = np.empty(len(names), dtype=float)
    weights = np.empty(len(names), dtype=float)
    for i, (name, o, matrix, delta, bias, normalizer, rotate_flag) in enumerate(
        zip(names, shifts, matrices, deltas, component_biases, normalizers, rotate_flags)
    ):
        fits[i] = 10000.0 * _component(name, x, o, matrix, rotate_flag) / normalizer + bias
        distance2 = float(np.sum((x - o) ** 2))
        if distance2 == 0.0:
            weights[i] = np.inf
        else:
            weights[i] = (1.0 / np.sqrt(distance2)) * np.exp(
                -distance2 / (2.0 * x.size * delta**2)
            )
    infinite = np.flatnonzero(np.isinf(weights))
    if infinite.size:
        return float(fits[infinite[0]])
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum == 0.0:
        weights.fill(1.0 / len(weights))
    else:
        weights /= weight_sum
    return float(np.dot(weights, fits))


def _f1(x, data):
    return _component("zakharov", x, data[0][0], data[1][0]) + CEC_BIASES[1]


def _f2(x, data):
    return _component("rosenbrock", x, data[0][0], data[1][0]) + CEC_BIASES[2]


def _f3(x, data):
    return _component("schaffer_f7", x, data[0][0], data[1][0]) + CEC_BIASES[3]


def _f4(x, data):
    return _component("step_rastrigin", x, data[0][0], data[1][0]) + CEC_BIASES[4]


def _f5(x, data):
    return _component("levy", x, data[0][0], data[1][0]) + CEC_BIASES[5]


def _f6(x, data):
    return _hybrid(x, data[0][0], data[1][0], _get_shuffle_data(6, x.size),
                   ("bent_cigar", "hgbat", "rastrigin"), (0.4, 0.4, 0.2)) + CEC_BIASES[6]


def _f7(x, data):
    return _hybrid(x, data[0][0], data[1][0], _get_shuffle_data(7, x.size),
                   ("hgbat", "katsuura", "ackley", "rastrigin", "schwefel", "schaffer_f7"),
                   (0.1, 0.2, 0.2, 0.2, 0.1, 0.2)) + CEC_BIASES[7]


def _f8(x, data):
    return _hybrid(x, data[0][0], data[1][0], _get_shuffle_data(8, x.size),
                   ("katsuura", "happycat", "grie_rosen", "schwefel", "ackley"),
                   (0.3, 0.2, 0.2, 0.1, 0.2)) + CEC_BIASES[8]


def _f9(x, data):
    return _composition(
        x, data[0], data[1],
        ("rosenbrock", "ellips", "bent_cigar", "discus", "ellips"),
        (10, 20, 30, 40, 50), (0, 200, 300, 100, 400),
        (1e4, 1e10, 1e30, 1e10, 1e10), (True, True, True, True, False),
    ) + CEC_BIASES[9]


def _f10(x, data):
    return _composition(
        x, data[0], data[1], ("schwefel", "rastrigin", "hgbat"),
        (20, 10, 10), (0, 200, 100), (10000, 10000, 10000), (False, True, True),
    ) + CEC_BIASES[10]


def _f11(x, data):
    return _composition(
        x, data[0], data[1], ("escaffer6", "schwefel", "griewank", "rosenbrock", "rastrigin"),
        (20, 20, 30, 30, 20), (0, 200, 300, 400, 200),
        (2e7, 10000, 1000, 10000, 1000),
    ) + CEC_BIASES[11]


def _f12(x, data):
    return _composition(
        x, data[0], data[1],
        ("hgbat", "rastrigin", "schwefel", "bent_cigar", "ellips", "escaffer6"),
        (10, 20, 30, 40, 50, 60), (0, 300, 500, 100, 400, 200),
        (1000, 1000, 4000, 1e30, 1e10, 2e7),
    ) + CEC_BIASES[12]


_NUM_SUBFUNCTIONS = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
    9: 5, 10: 3, 11: 5, 12: 6,
}
_FUNCTIONS = {1: _f1, 2: _f2, 3: _f3, 4: _f4, 5: _f5, 6: _f6, 7: _f7, 8: _f8,
              9: _f9, 10: _f10, 11: _f11, 12: _f12}
_DATA_CACHE: Dict[Tuple[int, int], Tuple[List[np.ndarray], List[np.ndarray]]] = {}


def get_cec2022_data(func_num: int, n_dim: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if func_num not in _FUNCTIONS:
        raise ValueError(f"Número de función inválido: {func_num}. Debe estar entre 1 y 12.")
    _validate_dimension(func_num, n_dim)
    key = (func_num, n_dim)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = get_shift_matrix_data(func_num, n_dim, _NUM_SUBFUNCTIONS[func_num])
    return _DATA_CACHE[key]


def cec2022_func(x: np.ndarray, func_num: int, n_dim: Optional[int] = None) -> float:
    """Evaluate one official CEC2022 function for a single decision vector."""
    if func_num not in _FUNCTIONS:
        raise ValueError(f"Número de función inválido: {func_num}. Debe estar entre 1 y 12.")
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("cec2022_func espera un vector unidimensional.")
    if n_dim is None:
        n_dim = x_arr.size
    if x_arr.size != n_dim:
        raise ValueError(f"La dimensión de x ({x_arr.size}) no coincide con n_dim ({n_dim}).")
    data = get_cec2022_data(func_num, n_dim)
    return float(_FUNCTIONS[func_num](x_arr, data))


def get_cec2022_optimum_point(func_num: int, n_dim: int) -> np.ndarray:
    """Return the first official shift vector used as the known optimum point."""
    return get_cec2022_data(func_num, n_dim)[0][0].copy()


def get_test_functions(n_dim: int = 20) -> List[ContinuousFunction]:
    """Return descriptors for the 12 CEC2022 functions."""
    names = [
        "F1_Shifted_Rotated_Zakharov",
        "F2_Shifted_Rotated_Rosenbrock",
        "F3_Shifted_Rotated_Expanded_Schaffers_F7",
        "F4_Shifted_Rotated_NonContinuous_Rastrigin",
        "F5_Shifted_Rotated_Levy",
        "F6_Hybrid_Function_1",
        "F7_Hybrid_Function_2",
        "F8_Hybrid_Function_3",
        "F9_Composition_Function_1",
        "F10_Composition_Function_2",
        "F11_Composition_Function_3",
        "F12_Composition_Function_4",
    ]
    _validate_dimension(1, n_dim)
    if n_dim == 2:
        raise ValueError("get_test_functions requiere D=10 o D=20 porque F6-F8 no están definidas para D=2.")
    return [
        ContinuousFunction(
            name=name,
            func=lambda x, f_num=num: cec2022_func(x, f_num, n_dim),
            lb=-100.0,
            ub=100.0,
            optimum=CEC_BIASES[num],
            n_dim=n_dim,
        )
        for num, name in enumerate(names, start=1)
    ]
