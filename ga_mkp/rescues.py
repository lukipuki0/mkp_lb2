"""
ga_mkp/rescues.py
-----------------
Estrategias avanzadas de rescate poblacional para el GA cuando
el StagnationMonitor detecta estancamiento.

Equivalencias con las variantes del notebook PSO original:
  V1 - Hill Climbing en el top 10%
  V5 - Inyeccion heuristica (densidad con ruido)
  V6 - Mutacion guiada por relajacion lineal (LP)
  V7 - Igual a V6 con memoria tabu de genes
  V8 - Ruin & Recreate en el top 30%
"""

from __future__ import annotations

import random
import numpy as np
from scipy.optimize import linprog

from mkp_core.problem import MKPInstance
from mkp_core.repair  import reparar_solucion


# ── V1: Hill Climbing local en el top ─────────────────────────────────────────

def hill_climb_elite(
    poblacion : list[list[int]],
    fitnesses : list[float],
    inst      : MKPInstance,
    top_frac  : float = 0.10,
) -> tuple[list[list[int]], list[float]]:
    """Aplica Hill Climbing de 1 bit al top `top_frac` de la poblacion."""
    n_elite = max(1, int(len(poblacion) * top_frac))
    # Indices del top (los mejores)
    top_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:n_elite]

    for idx in top_indices:
        ind = poblacion[idx].copy()
        for bit in range(inst.n):
            vecino        = ind.copy()
            vecino[bit]   = 1 - vecino[bit]
            vecino, val   = reparar_solucion(vecino, inst)
            if val > fitnesses[idx]:
                ind             = vecino
                fitnesses[idx]  = val
        poblacion[idx] = ind

    return poblacion, fitnesses


# ── V5: Inyeccion heuristica ─────────────────────────────────────────────────

def inyectar_heuristicos(
    poblacion : list[list[int]],
    fitnesses : list[float],
    inst      : MKPInstance,
    frac      : float = 0.50,
) -> tuple[list[list[int]], list[float]]:
    """Reemplaza la peor `frac` de la poblacion por individuos guiados por densidad."""
    n_reemplazar = int(len(poblacion) * frac)
    peores = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:n_reemplazar]

    for idx in peores:
        sol = [0] * inst.n
        ruido = np.random.uniform(0.85, 1.15, size=inst.n)
        orden = np.argsort(inst.density * ruido)[::-1]
        for j in orden:
            candidato      = sol.copy()
            candidato[int(j)] = 1
            if inst.es_factible(candidato):
                sol = candidato
        fitnesses[idx]  = inst.evaluar(sol)
        poblacion[idx]  = sol

    return poblacion, fitnesses


# ── V6 / V7: Mutacion LP-Dirigida ────────────────────────────────────────────

def _resolver_lp(inst: MKPInstance) -> np.ndarray:
    """Resuelve la relajacion LP del MKP y devuelve las utilidades por item."""
    res    = linprog(-inst.p, A_ub=inst.r, b_ub=inst.b, bounds=(0, 1), method='highs')
    duales = res.ineqlin.marginals if res.success and hasattr(res.ineqlin, 'marginals') else np.ones(inst.m)
    utilidad = np.array([
        inst.p[j] / max(sum(inst.r[i][j] * abs(duales[i]) for i in range(inst.m)), 1e-9)
        for j in range(inst.n)
    ])
    return utilidad / (np.max(utilidad) + 1e-9)


def mutacion_lp(
    poblacion : list[list[int]],
    fitnesses : list[float],
    inst      : MKPInstance,
    tabu_genes: list[int] | None = None,
) -> tuple[list[list[int]], list[float]]:
    """Muta toda la poblacion guiada por precios sombra del LP (V6/V7)."""
    prob_activar = _resolver_lp(inst)
    tabu_set     = set(tabu_genes or [])

    for idx in range(len(poblacion)):
        nuevo = poblacion[idx].copy()
        for j in range(inst.n):
            if j in tabu_set:
                continue
            if random.random() < prob_activar[j]:
                nuevo[j] = 1
            elif random.random() > prob_activar[j]:
                nuevo[j] = 0
        nuevo, val     = reparar_solucion(nuevo, inst)
        if val >= fitnesses[idx]:
            poblacion[idx] = nuevo
            fitnesses[idx] = val

    return poblacion, fitnesses


# ── V8: Ruin & Recreate ──────────────────────────────────────────────────────

def ruin_and_recreate_elite(
    poblacion : list[list[int]],
    fitnesses : list[float],
    inst      : MKPInstance,
    frac      : float = 0.30,
) -> tuple[list[list[int]], list[float]]:
    """Aplica Ruin & Recreate al top `frac` de la poblacion."""
    n_elite  = max(1, int(len(poblacion) * frac))
    top_idx  = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:n_elite]

    for idx in top_idx:
        sol = poblacion[idx].copy()
        # RUIN: apagar la mitad de los bits activos aleatoriamente
        activos = [i for i, v in enumerate(sol) if v == 1]
        if activos:
            destruir = random.sample(activos, k=max(1, len(activos) // 2))
            for j in destruir:
                sol[j] = 0
        # RECREATE: llenar vorazmente con orden de densidad descendente
        for j in inst.indices_ascendentes[::-1]:
            candidato     = sol.copy()
            candidato[int(j)] = 1
            if inst.es_factible(candidato):
                sol = candidato

        val            = inst.evaluar(sol)
        if val >= fitnesses[idx]:
            poblacion[idx]  = sol
            fitnesses[idx]  = val

    return poblacion, fitnesses
