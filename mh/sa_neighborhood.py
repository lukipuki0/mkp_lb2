"""
mh/sa_neighborhood.py
---------------------
Operadores de vecindad para Simulated Annealing (SA).

  - flip_bits  : invierte `num_flip` bits al azar (por defecto).
  - swap_bits  : intercambia un bit 1 con un bit 0.
  - block_flip : invierte un bloque contiguo de bits.
"""

from __future__ import annotations

import random

from mkp_core.problem import MKPInstance
from mkp_core.repair  import reparar_solucion


def flip_bits(
    sol_actual: list[int],
    inst: MKPInstance,
    num_flip: int = 3,
) -> tuple[list[int], float]:
    """Genera un vecino invirtiendo `num_flip` bits al azar y lo repara."""
    vecino = sol_actual.copy()
    indices = random.sample(range(inst.n), min(num_flip, inst.n))
    for idx in indices:
        vecino[idx] = 1 - vecino[idx]
    vecino, valor = reparar_solucion(vecino, inst)
    return vecino, valor


def swap_bits(
    sol_actual: list[int],
    inst: MKPInstance,
    num_flip: int = 1,
) -> tuple[list[int], float]:
    """Intercambia un bit en 1 por un bit en 0 (swap de estado)."""
    vecino = sol_actual.copy()
    unos  = [i for i, b in enumerate(vecino) if b == 1]
    ceros = [i for i, b in enumerate(vecino) if b == 0]
    if unos and ceros:
        i_on  = random.choice(unos)
        i_off = random.choice(ceros)
        vecino[i_on]  = 0
        vecino[i_off] = 1
    vecino, valor = reparar_solucion(vecino, inst)
    return vecino, valor


def block_flip(
    sol_actual: list[int],
    inst: MKPInstance,
    num_flip: int = 3,
) -> tuple[list[int], float]:
    """Invierte un bloque contiguo de `num_flip` bits al azar."""
    vecino = sol_actual.copy()
    n = inst.n
    tam = min(num_flip, n)
    inicio = random.randint(0, n - tam)
    for i in range(inicio, inicio + tam):
        vecino[i] = 1 - vecino[i]
    vecino, valor = reparar_solucion(vecino, inst)
    return vecino, valor


# ── Registro de operadores ────────────────────────────────────────────────────

OPERATORS: dict[str, callable] = {
    "flip_bits": flip_bits,
    "swap_bits": swap_bits,
    "block_flip": block_flip,
}


def get_operator(nombre: str) -> callable:
    """Devuelve el operador de vecindad registrado bajo *nombre*."""
    if nombre not in OPERATORS:
        raise KeyError(
            f"Operador '{nombre}' no encontrado. "
            f"Disponibles: {list(OPERATORS.keys())}"
        )
    return OPERATORS[nombre]

