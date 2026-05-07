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
    sol = list(solucion)   # copia de trabajo

    # ── Fase 1: Expulsión ──────────────────────────────────────────────────
    # Recorre ítems de menor a mayor densidad y los desactiva hasta lograr
    # factibilidad.
    for idx in inst.indices_ascendentes:
        if np.all(np.dot(sol, inst.r.T) <= inst.b):
            break                        # ya es factible
        sol[int(idx)] = 0

    # ── Fase 2: Inserción ─────────────────────────────────────────────────
    # Recorre ítems de mayor a menor densidad y los activa si caben.
    for idx in inst.indices_ascendentes[::-1]:
        candidato = sol.copy()
        candidato[int(idx)] = 1
        if np.all(np.dot(candidato, inst.r.T) <= inst.b):
            sol = candidato

    valor_total = inst.evaluar(sol)
    return sol, valor_total
