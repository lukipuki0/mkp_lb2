"""
ts_mkp/rescues.py
-----------------
Lógicas de rescate avanzadas para Tabu Search (TS).
Se usan para modificar agresivamente la solucion actual
y escapar de estancamientos largos.
"""

from __future__ import annotations

import random
import numpy as np
from scipy.optimize import linprog

from mkp_core.problem import MKPInstance
from mkp_core.repair  import reparar_solucion


def salto_heuristico(inst: MKPInstance) -> tuple[list[int], float]:
    """V5: Genera una solucion guiada por densidad con ruido."""
    sol = [0] * inst.n
    ruido = np.random.uniform(0.85, 1.15, size=inst.n)
    orden = np.argsort(inst.density * ruido)[::-1]
    
    for j in orden:
        candidato = sol.copy()
        candidato[int(j)] = 1
        if inst.es_factible(candidato):
            sol = candidato
            
    val = inst.evaluar(sol)
    return sol, val


def salto_lp(inst: MKPInstance, sol_actual: list[int]) -> tuple[list[int], float, list[int]]:
    """
    V6/V7: Resuelve LP y fuerza a la solucion a adoptar los bits que LP sugiere.
    Devuelve la solucion modificada y una lista de los indices que fueron forzados.
    """
    res = linprog(-inst.p, A_ub=inst.r, b_ub=inst.b, bounds=(0, 1), method='highs')
    duales = res.ineqlin.marginals if res.success and hasattr(res.ineqlin, 'marginals') else np.ones(inst.m)
    
    utilidad = np.array([
        inst.p[j] / max(sum(inst.r[i][j] * abs(duales[i]) for i in range(inst.m)), 1e-9)
        for j in range(inst.n)
    ])
    prob_activar = utilidad / (np.max(utilidad) + 1e-9)
    
    nuevo = sol_actual.copy()
    indices_forzados = []
    
    for j in range(inst.n):
        if prob_activar[j] > 0.9:  # LP esta muy seguro de prenderlo
            if nuevo[j] == 0:
                nuevo[j] = 1
                indices_forzados.append(j)
        elif prob_activar[j] < 0.1: # LP esta muy seguro de apagarlo
            if nuevo[j] == 1:
                nuevo[j] = 0
                indices_forzados.append(j)
                
    nuevo, val = reparar_solucion(nuevo, inst)
    return nuevo, val, indices_forzados


def ruin_and_recreate_sol(sol_actual: list[int], inst: MKPInstance) -> tuple[list[int], float]:
    """V8: Destruye el 50% de los items activos y reconstruye vorazmente."""
    sol = sol_actual.copy()
    activos = [i for i, v in enumerate(sol) if v == 1]
    if activos:
        destruir = random.sample(activos, k=max(1, len(activos) // 2))
        for j in destruir:
            sol[j] = 0
            
    for j in inst.indices_ascendentes[::-1]:
        candidato = sol.copy()
        candidato[int(j)] = 1
        if inst.es_factible(candidato):
            sol = candidato
            
    val = inst.evaluar(sol)
    return sol, val
