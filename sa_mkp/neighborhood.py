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


# ── Registro de operadores ────────────────────────────────────────────────────
# Permite seleccionar el operador por nombre desde config o CLI.

OPERATORS: dict[str, callable] = {
    "flip_bits": flip_bits,
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
