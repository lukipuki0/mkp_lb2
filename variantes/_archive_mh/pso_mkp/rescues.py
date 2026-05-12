"""
pso_mkp/rescues.py
------------------
Estrategias de rescate avanzadas para el PSO cuando el
StagnationMonitor detecta estancamiento.

Contiene las lógicas correspondientes a las versiones V5-V8 y
funciones auxiliares reutilizadas por V1-V4 en algorithm.py.
"""

from __future__ import annotations

import random
import numpy as np
from scipy.optimize import linprog

from mkp_core.problem import MKPInstance
from mkp_core.repair  import reparar_solucion


def _actualizar_particula(particula: dict, nueva_sol: list[int], inst: MKPInstance) -> None:
    """Actualiza la solución de la partícula y sus mejores históricos si corresponde."""
    val = inst.evaluar(nueva_sol)
    if val >= particula['valor']:
        particula['solucion'] = nueva_sol
        particula['valor'] = val
        if val > particula['mejor_valor_personal']:
            particula['mejor_valor_personal'] = val
            particula['mejor_solucion_personal'] = nueva_sol


# ── V5: Inyección Heurística (reemplaza peores partículas) ────────────────────

def inyectar_heuristicos(
    particulas: list[dict],
    inst: MKPInstance,
    frac: float = 0.50,
) -> list[dict]:
    """Reemplaza la peor `frac` de las partículas con soluciones guiadas por densidad."""
    n_reemplazar = int(len(particulas) * frac)
    peores = sorted(range(len(particulas)), key=lambda i: particulas[i]['valor'])[:n_reemplazar]

    for idx in peores:
        sol = [0] * inst.n
        ruido = np.random.uniform(0.85, 1.15, size=inst.n)
        orden = np.argsort(inst.density * ruido)[::-1]
        for j in orden:
            candidato = sol.copy()
            candidato[int(j)] = 1
            if inst.es_factible(candidato):
                sol = candidato
        
        # Reiniciar completamente la partícula para darle un nuevo punto de partida
        val = inst.evaluar(sol)
        particulas[idx]['solucion'] = sol
        particulas[idx]['valor'] = val
        particulas[idx]['mejor_solucion_personal'] = sol
        particulas[idx]['mejor_valor_personal'] = val

    return particulas


# ── V6 / V7: Mutación LP-Dirigida ────────────────────────────────────────────

def _resolver_lp(inst: MKPInstance) -> np.ndarray:
    """Resuelve la relajación LP del MKP y devuelve probabilidades de activación."""
    res    = linprog(-inst.p, A_ub=inst.r, b_ub=inst.b, bounds=(0, 1), method='highs')
    duales = res.ineqlin.marginals if res.success and hasattr(res.ineqlin, 'marginals') else np.ones(inst.m)
    utilidad = np.array([
        inst.p[j] / max(sum(inst.r[i][j] * abs(duales[i]) for i in range(inst.m)), 1e-9)
        for j in range(inst.n)
    ])
    return utilidad / (np.max(utilidad) + 1e-9)


def mutacion_lp(
    particulas: list[dict],
    inst: MKPInstance,
    tabu_genes: list[int] | None = None,
) -> list[dict]:
    """Muta toda la población guiada por precios sombra del LP (V6/V7)."""
    prob_activar = _resolver_lp(inst)
    tabu_set     = set(tabu_genes or [])

    for idx in range(len(particulas)):
        nuevo = particulas[idx]['solucion'].copy()
        for j in range(inst.n):
            if j in tabu_set:
                continue
            if random.random() < prob_activar[j]:
                nuevo[j] = 1
            elif random.random() > prob_activar[j]:
                nuevo[j] = 0
        nuevo, val = reparar_solucion(nuevo, inst)
        if val >= particulas[idx]['valor']:
            _actualizar_particula(particulas[idx], nuevo, inst)

    return particulas


# ── V8: Ruin & Recreate ──────────────────────────────────────────────────────

def ruin_and_recreate_enjambre(
    particulas: list[dict],
    inst: MKPInstance,
    frac: float = 0.30,
) -> list[dict]:
    """Aplica Ruin & Recreate al top `frac` de la población."""
    n_elite = max(1, int(len(particulas) * frac))
    top_idx = sorted(range(len(particulas)), key=lambda i: particulas[i]['valor'], reverse=True)[:n_elite]

    for idx in top_idx:
        sol = particulas[idx]['solucion'].copy()
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
        if val >= particulas[idx]['valor']:
            _actualizar_particula(particulas[idx], sol, inst)

    return particulas
