"""
mh/ts_neighborhood.py
---------------------
Generación de vecindarios para Tabu Search (TS).
Evalúa vecinos a distancia Hamming 1 (o 2) y aplica reparación.
"""

from __future__ import annotations

import random

from mkp_core.problem import MKPInstance
from mkp_core.repair  import reparar_solucion


def obtener_mejor_vecino(
    sol_actual   : list[int],
    inst         : MKPInstance,
    tabu_list    : dict[int, int],  # item_idx -> iteración_donde_expira
    iter_actual  : int,
    mejor_global : float,
    max_evals    : int = 0,         # Si 0, evalúa todos (n). Si > 0, elige un subset
    num_flips    : int = 1,         # 1-bit flip o 2-bit flip
) -> tuple[list[int], float, int]:
    """
    Genera vecinos, filtra tabú (salvo aspiración), los repara y devuelve
    la mejor solución vecina válida, su valor y el bit principal que se modificó.
    """
    mejor_vecino = None
    mejor_valor  = -float("inf")
    mejor_mov    = -1

    # Determinar qué items probar (vecindario)
    if max_evals > 0 and max_evals < inst.n:
        candidatos = random.sample(range(inst.n), max_evals)
    else:
        candidatos = list(range(inst.n))

    for idx in candidatos:
        vecino = sol_actual.copy()
        vecino[idx] = 1 - vecino[idx]

        # Opcional: flip extra si num_flips == 2
        extra_idx = -1
        if num_flips == 2:
            extra_idx = random.choice([i for i in range(inst.n) if i != idx])
            vecino[extra_idx] = 1 - vecino[extra_idx]

        # Reparar
        vecino, valor = reparar_solucion(vecino, inst)

        # Chequeo Tabú
        es_tabu = False
        if tabu_list.get(idx, 0) > iter_actual:
            es_tabu = True
        if num_flips == 2 and extra_idx != -1 and tabu_list.get(extra_idx, 0) > iter_actual:
            es_tabu = True

        # Criterio de Aspiración: si es tabú pero mejora el MEJOR GLOBAL HISTÓRICO, se acepta
        if es_tabu and valor > mejor_global:
            es_tabu = False

        if not es_tabu:
            if valor > mejor_valor:
                mejor_valor  = valor
                mejor_vecino = vecino
                mejor_mov    = idx

    # Si todos los vecinos son tabú y ninguno aspira, elegimos uno al azar violando tabú.
    if mejor_vecino is None:
        idx = random.choice(candidatos)
        vecino = sol_actual.copy()
        vecino[idx] = 1 - vecino[idx]
        vecino, valor = reparar_solucion(vecino, inst)
        return vecino, valor, idx

    return mejor_vecino, mejor_valor, mejor_mov

