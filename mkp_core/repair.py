"""
repair.py
─────────
Función de reparación para soluciones binarias del MKP.

Estrategia (idéntica a la del notebook PSO original):
  1. Fase de EXPULSIÓN – mientras la solución sea infactible, elimina ítems
     empezando por los de menor densidad.
  2. Fase de INSERCIÓN – agrega ítems en orden descendente de densidad siempre
     que la restricción de capacidad lo permita.

La función es pura: no modifica la solución de entrada.
"""

from __future__ import annotations

import numpy as np

from mkp_core.problem import MKPInstance


def reparar_solucion(
    solucion: list[int],
    inst: MKPInstance,
) -> tuple[list[int], float]:
    """Repara *solucion* y devuelve (solucion_reparada, valor_total).

    Parameters
    ----------
    solucion : list[int]
        Solución binaria (puede ser infactible).
    inst : MKPInstance
        Instancia del problema con densidades precalculadas.

    Returns
    -------
    solucion_reparada : list[int]
        Solución factible.
    valor_total : float
        Ganancia acumulada de la solución reparada.
    """
    sol = np.array(solucion, dtype=np.int8)
    uso = inst.r @ sol
    indices_asc = inst.indices_ascendentes
    b = inst.b
    r = inst.r
    p = inst.p

    # ── Fase 1: Expulsión ──────────────────────────────────────────────────
    # Recorre ítems de menor a mayor densidad y los desactiva hasta lograr factibilidad.
    for idx in indices_asc:
        if np.all(uso <= b):
            break
        if sol[idx] == 1:
            sol[idx] = 0
            uso -= r[:, idx]

    # ── Fase 2: Inserción ─────────────────────────────────────────────────
    # Recorre ítems de mayor a menor densidad y los activa si caben.
    for idx in indices_asc[::-1]:
        if sol[idx] == 0:
            w = r[:, idx]
            if np.all(uso + w <= b):
                sol[idx] = 1
                uso += w

    valor_total = float(sol @ p)
    return sol.tolist(), valor_total
