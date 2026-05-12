"""
neighborhood.py
───────────────
Operadores de vecindad para SA binario aplicado al MKP.

Actualmente implementa:
  - flip_bits : perturba k bits aleatorios y repara la solución resultante.

Se puede extender con otros operadores (swap, segment-flip, etc.) siguiendo
la misma firma: (solucion, inst, **kwargs) → (vecino, valor).
"""

from __future__ import annotations

import random

from mkp_core.problem import MKPInstance
from mkp_core.repair import reparar_solucion


# ── Operador principal ────────────────────────────────────────────────────────

def flip_bits(
    solucion: list[int],
    inst: MKPInstance,
    num_flip: int = 3,
) -> tuple[list[int], float]:
    """Genera un vecino flipando *num_flip* bits aleatorios y reparando.

    Parameters
    ----------
    solucion : list[int]
        Solución binaria actual (factible).
    inst : MKPInstance
        Instancia del problema.
    num_flip : int
        Cantidad de bits a perturbar.

    Returns
    -------
    vecino : list[int]
        Solución vecina factible.
    valor : float
        Ganancia del vecino.
    """
    vecino = solucion.copy()
    indices = random.sample(range(len(vecino)), num_flip)
    for idx in indices:
        vecino[idx] = 1 - vecino[idx]
    vecino, valor = reparar_solucion(vecino, inst)
    return vecino, valor


def swap_bits(
    solucion: list[int],
    inst: MKPInstance,
    num_flip: int = 1,
) -> tuple[list[int], float]:
    """Genera un vecino intercambiando num_flip pares de (1 y 0) aleatorios y reparando."""
    vecino = solucion.copy()
    ones = [i for i, v in enumerate(vecino) if v == 1]
    zeros = [i for i, v in enumerate(vecino) if v == 0]
    
    if not ones or not zeros:
        return vecino, inst.evaluar(vecino)
        
    for _ in range(min(num_flip, len(ones), len(zeros))):
        i = random.choice(ones)
        j = random.choice(zeros)
        vecino[i] = 0
        vecino[j] = 1
        ones.remove(i)
        zeros.remove(j)
        
    vecino, valor = reparar_solucion(vecino, inst)
    return vecino, valor

def block_flip(
    solucion: list[int],
    inst: MKPInstance,
    num_flip: int = 3,
) -> tuple[list[int], float]:
    """Voltea un bloque contiguo de tamaño num_flip."""
    vecino = solucion.copy()
    n = len(vecino)
    if n == 0:
        return vecino, inst.evaluar(vecino)
    
    start_idx = random.randint(0, n - 1)
    for i in range(num_flip):
        idx = (start_idx + i) % n
        vecino[idx] = 1 - vecino[idx]
        
    vecino, valor = reparar_solucion(vecino, inst)
    return vecino, valor


# ── Registro de operadores ────────────────────────────────────────────────────
# Permite seleccionar el operador por nombre desde config o CLI.

OPERATORS: dict[str, callable] = {
    "flip_bits": flip_bits,
    "swap_bits": swap_bits,
    "block_flip": block_flip,
}


def get_operator(nombre: str) -> callable:
    """Devuelve el operador de vecindad registrado bajo *nombre*.

    Raises
    ------
    KeyError
        Si el operador no está registrado.
    """
    if nombre not in OPERATORS:
        raise KeyError(
            f"Operador '{nombre}' no encontrado. "
            f"Disponibles: {list(OPERATORS.keys())}"
        )
    return OPERATORS[nombre]
