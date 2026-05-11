"""
gwo_mkp/rescues.py
------------------
Estrategias de rescate avanzadas para el GWO cuando el
StagnationMonitor detecta estancamiento.

Contiene las logicas correspondientes a las versiones V5-V8 y
funciones auxiliares reutilizadas por V1-V4 en el algorithm.py.
"""

from __future__ import annotations

import random
import numpy as np
from scipy.optimize import linprog

from mkp_core.problem import MKPInstance
from mkp_core.repair  import reparar_solucion


# ── V5: Inyeccion Heuristica (reemplaza peores lobos) ────────────────────────

def inyectar_heuristicos(
    poblacion_bin: list[list[int]],
    fitnesses: list[float],
    posiciones: np.ndarray,
    inst: MKPInstance,
    frac: float = 0.50,
    v_max: float = 6.0,
) -> tuple[list[list[int]], list[float], np.ndarray]:
    """Reemplaza la peor `frac` de la manada con soluciones guiadas por densidad."""
    n_reemplazar = int(len(poblacion_bin) * frac)
    peores = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:n_reemplazar]

    for idx in peores:
        sol = [0] * inst.n
        ruido = np.random.uniform(0.85, 1.15, size=inst.n)
        orden = np.argsort(inst.density * ruido)[::-1]
        for j in orden:
            candidato = sol.copy()
            candidato[int(j)] = 1
            if inst.es_factible(candidato):
                sol = candidato
        fitnesses[idx]     = inst.evaluar(sol)
        poblacion_bin[idx] = sol
        # Resetear posicion continua del lobo
        posiciones[idx]    = np.random.uniform(-v_max, v_max, size=inst.n)

    return poblacion_bin, fitnesses, posiciones


# ── V6 / V7: Mutacion LP-Dirigida ────────────────────────────────────────────

def _resolver_lp(inst: MKPInstance) -> np.ndarray:
    """Resuelve la relajacion LP del MKP y devuelve probabilidades de activacion."""
    res    = linprog(-inst.p, A_ub=inst.r, b_ub=inst.b, bounds=(0, 1), method='highs')
    duales = res.ineqlin.marginals if res.success and hasattr(res.ineqlin, 'marginals') else np.ones(inst.m)
    utilidad = np.array([
        inst.p[j] / max(sum(inst.r[i][j] * abs(duales[i]) for i in range(inst.m)), 1e-9)
        for j in range(inst.n)
    ])
    return utilidad / (np.max(utilidad) + 1e-9)


def mutacion_lp(
    poblacion_bin: list[list[int]],
    fitnesses: list[float],
    inst: MKPInstance,
    tabu_genes: list[int] | None = None,
) -> tuple[list[list[int]], list[float]]:
    """Muta toda la manada guiada por precios sombra del LP (V6/V7)."""
    prob_activar = _resolver_lp(inst)
    tabu_set     = set(tabu_genes or [])

    for idx in range(len(poblacion_bin)):
        nuevo = poblacion_bin[idx].copy()
        for j in range(inst.n):
            if j in tabu_set:
                continue
            if random.random() < prob_activar[j]:
                nuevo[j] = 1
            elif random.random() > prob_activar[j]:
                nuevo[j] = 0
        nuevo, val = reparar_solucion(nuevo, inst)
        if val >= fitnesses[idx]:
            poblacion_bin[idx] = nuevo
            fitnesses[idx]     = val

    return poblacion_bin, fitnesses


# ── V8: Ruin & Recreate ──────────────────────────────────────────────────────

def ruin_and_recreate_manada(
    poblacion_bin: list[list[int]],
    fitnesses: list[float],
    inst: MKPInstance,
    frac: float = 0.30,
) -> tuple[list[list[int]], list[float]]:
    """Aplica Ruin & Recreate al top `frac` de la manada."""
    n_elite = max(1, int(len(poblacion_bin) * frac))
    top_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:n_elite]

    for idx in top_idx:
        sol = poblacion_bin[idx].copy()
        # RUIN: apagar la mitad de los bits activos
        activos = [i for i, v in enumerate(sol) if v == 1]
        if activos:
            destruir = random.sample(activos, k=max(1, len(activos) // 2))
            for j in destruir:
                sol[j] = 0
        # RECREATE: llenar vorazmente con orden de densidad descendente
        for j in inst.indices_ascendentes[::-1]:
            candidato = sol.copy()
            candidato[int(j)] = 1
            if inst.es_factible(candidato):
                sol = candidato

        val = inst.evaluar(sol)
        if val >= fitnesses[idx]:
            poblacion_bin[idx] = sol
            fitnesses[idx]     = val

    return poblacion_bin, fitnesses
