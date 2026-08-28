"""
mezclas_mh/ga_sa/algoritmos/mdg_gasa.py
---------------------------------------
MDG-GASA: Momentum Diversity-Guided Genetic Algorithm - Simulated Annealing (Variante D / Combinada B + C).

Lógica:
  1. Utiliza la Diversidad Poblacional (Variante C) para decidir el modo de operación en cada paso.
  2. Modula la intensidad de perturbación y la aceptación de soluciones mediante un esquema
     de Temperatura Adaptativa y Filtro de Metrópolis (Variante B).
  3. Aplica un Cruce Térmico Guiado donde el ruido de recombinación decae exponencialmente
     con la temperatura del sistema.

Soporta:
  - Problema Discreto MKP (reparación factible).
  - Problema Continuo (CEC2022 / HRES / Hypertuning).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from mkp_core.problem import MKPInstance
from mkp_core.repair import reparar_solucion
from mh.ga_operators import torneo, get_crossover, get_mutation
from dtw_stagnation import StagnationConfig, StagnationMonitor


@dataclass
class MDGGASAParams:
    """Hiperparámetros de MDG-GASA (Variante D)."""
    pop_size        : int   = 50
    generations     : int   = 300
    epochs          : int   = 1
    elitism         : int   = 2
    tournament_size : int   = 3
    crossover_rate  : float = 0.85
    crossover_op    : str   = "uniform"
    mutation_op     : str   = "bitflip"
    # Parámetros Térmicos
    T_inicial       : float = 1000.0
    T_final         : float = 0.05
    # Parámetros de Diversidad
    umbral_init     : float = 0.60
    umbral_final    : float = 0.10
    # Parámetros de perturbación
    mutation_rate_max: float = 0.15
    mutation_rate_min: float = 0.02
    mutation_sigma  : float = 0.15
    blx_alpha       : float = 0.5
    # Pipeline híbrido
    injection_mode  : str   = "mixed"
    use_stagnation  : bool  = True
    stag_cfg        : StagnationConfig | None = None


@dataclass
class MDGGASAEpochResult:
    """Resultado de un epoch de MDG-GASA."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[int] | np.ndarray = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)
    dtw_info_hist    : list[dict]  = field(default_factory=list)
    historial_div    : list[float] = field(default_factory=list)


# ── Helpers Discretos (MKP) ───────────────────────────────────────────────────

def _inicializar_poblacion_mkp(inst: MKPInstance, pop_size: int) -> tuple[list[list[int]], list[float]]:
    poblacion = []
    fitnesses = []
    for _ in range(pop_size):
        sol = [random.randint(0, 1) for _ in range(inst.n)]
        sol, val = reparar_solucion(sol, inst)
        poblacion.append(sol)
        fitnesses.append(val)
    return poblacion, fitnesses


def _calcular_diversidad_binaria(poblacion: list[list[int]]) -> float:
    arr = np.array(poblacion, dtype=float)
    centroide = np.mean(arr, axis=0)
    return float(np.mean(np.linalg.norm(arr - centroide, axis=1)))


# ── Ejecución Discreta MKP ────────────────────────────────────────────────────

def ejecutar_epoch(
    inst          : MKPInstance,
    params        : MDGGASAParams,
    epoch_idx     : int = 0,
    verbose       : bool = True,
    sol_inyectada : list[int] | None = None,
) -> MDGGASAEpochResult:
    """Ejecuta MDG-GASA para MKP."""
    pop_size = params.pop_size
    poblacion, fitnesses = _inicializar_poblacion_mkp(inst, pop_size)

    if sol_inyectada is not None:
        sol_rep, val_rep = reparar_solucion(list(sol_inyectada), inst)
        poblacion[0] = sol_rep
        fitnesses[0] = val_rep

    div_0 = max(1e-9, _calcular_diversidad_binaria(poblacion))
    cross_fn = get_crossover(params.crossover_op)
    mut_fn   = get_mutation(params.mutation_op)

    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    mejor_val = fitnesses[best_idx]
    mejor_sol = list(poblacion[best_idx])

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    dtw_info_hist  = []
    historial_div  = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.generations):
        frac = it / max(1, params.generations - 1)
        T_t = params.T_inicial * ((params.T_final / max(1e-9, params.T_inicial)) ** frac)

        # 1. Diversidad y Umbral
        div_act = _calcular_diversidad_binaria(poblacion)
        div_norm = div_act / div_0
        historial_div.append(div_norm)
        umbral_t = params.umbral_init * (1.0 - frac) + params.umbral_final * frac

        # Tasa de mutación dependiente de temperatura y déficit de diversidad
        factor_div = max(0.0, (umbral_t - div_norm) / max(1e-5, umbral_t)) if div_norm < umbral_t else 0.0
        p_m_base = params.mutation_rate_min + (params.mutation_rate_max - params.mutation_rate_min) * (T_t / max(1e-9, params.T_inicial))
        p_m_t = min(0.35, p_m_base + 0.15 * factor_div)

        # 2. Elitismo
        indices_ordenados = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        nueva_pob = [list(poblacion[i]) for i in indices_ordenados[:params.elitism]]
        nuevos_fit = [fitnesses[i] for i in indices_ordenados[:params.elitism]]

        # 3. Cruce + Mutación Térmica con Filtro Metrópolis
        while len(nueva_pob) < pop_size:
            p1 = torneo(poblacion, fitnesses, params.tournament_size)
            p2 = torneo(poblacion, fitnesses, params.tournament_size)

            if random.random() < params.crossover_rate:
                h1, h2 = cross_fn(p1, p2)
            else:
                h1, h2 = list(p1), list(p2)

            h1 = mut_fn(h1, p_m_t)
            h2 = mut_fn(h2, p_m_t)

            h1, v1 = reparar_solucion(h1, inst)
            h2, v2 = reparar_solucion(h2, inst)

            v_p1 = np.dot(p1, inst.p)
            delta1 = v1 - v_p1
            if delta1 >= 0 or random.random() < math.exp(delta1 / max(1e-9, T_t)):
                nueva_pob.append(h1)
                nuevos_fit.append(v1)
            else:
                nueva_pob.append(list(p1))
                nuevos_fit.append(v_p1)

            if len(nueva_pob) < pop_size:
                v_p2 = np.dot(p2, inst.p)
                delta2 = v2 - v_p2
                if delta2 >= 0 or random.random() < math.exp(delta2 / max(1e-9, T_t)):
                    nueva_pob.append(h2)
                    nuevos_fit.append(v2)
                else:
                    nueva_pob.append(list(p2))
                    nuevos_fit.append(v_p2)

        poblacion = nueva_pob
        fitnesses = nuevos_fit

        iter_best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        fit_iter_best = fitnesses[iter_best_idx]

        if fit_iter_best > mejor_val:
            mejor_val = fit_iter_best
            mejor_sol = list(poblacion[iter_best_idx])

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)

        dtw_info = {}
        if monitor is not None:
            status = monitor.update(mejor_val)
            dtw_info = status.copy()
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                dtw_info_hist.append(dtw_info)
                if verbose:
                    print(f"    [Stagnation] Fire #{stag_fires} @ gen {it + 1} -> ABORT")
                break
        dtw_info_hist.append(dtw_info)

    return MDGGASAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
        dtw_info_hist    = dtw_info_hist,
        historial_div    = historial_div,
    )


# ── Ejecución Continua (CEC2022 / HRES / Hypertuning) ─────────────────────────

def ejecutar_epoch_continuo(
    func      : Callable[[np.ndarray], float],
    dim       : int,
    bounds    : tuple[float, float],
    params    : MDGGASAParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> MDGGASAEpochResult:
    """Ejecuta MDG-GASA en optimización continua (minimización)."""
    lb, ub = bounds
    pop_size = params.pop_size

    poblacion = np.random.uniform(lb, ub, (pop_size, dim))
    fitnesses = [float(func(ind)) for ind in poblacion]

    if sol_inyectada is not None:
        poblacion[0] = np.clip(sol_inyectada.copy(), lb, ub)
        fitnesses[0] = float(func(poblacion[0]))

    centroide_0 = np.mean(poblacion, axis=0)
    div_0 = max(1e-9, float(np.mean(np.linalg.norm(poblacion - centroide_0, axis=1))))

    best_idx = int(np.argmin(fitnesses))
    mejor_val = fitnesses[best_idx]
    mejor_sol = poblacion[best_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    dtw_info_hist  = []
    historial_div  = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.generations):
        frac = it / max(1, params.generations - 1)
        T_t = params.T_inicial * ((params.T_final / max(1e-9, params.T_inicial)) ** frac)

        # 1. Diversidad y Umbral
        centroide = np.mean(poblacion, axis=0)
        div_act = float(np.mean(np.linalg.norm(poblacion - centroide, axis=1)))
        div_norm = div_act / div_0
        historial_div.append(div_norm)
        umbral_t = params.umbral_init * (1.0 - frac) + params.umbral_final * frac

        factor_div = max(0.0, (umbral_t - div_norm) / max(1e-5, umbral_t)) if div_norm < umbral_t else 0.0
        sigma_t = params.mutation_sigma * (T_t / max(1e-9, params.T_inicial)) * (1.0 + 1.5 * factor_div)

        # 2. Elitismo
        indices_ordenados = np.argsort(fitnesses)
        nueva_pob = [poblacion[i].copy() for i in indices_ordenados[:params.elitism]]
        nuevos_fit = [fitnesses[i] for i in indices_ordenados[:params.elitism]]

        # 3. Cruce Térmico Guiado + Metrópolis
        while len(nueva_pob) < pop_size:
            t_idx1 = random.sample(range(pop_size), params.tournament_size)
            idx_p1 = min(t_idx1, key=lambda i: fitnesses[i])
            p1 = poblacion[idx_p1].copy()
            fit_p1 = fitnesses[idx_p1]

            t_idx2 = random.sample(range(pop_size), params.tournament_size)
            idx_p2 = min(t_idx2, key=lambda i: fitnesses[i])
            p2 = poblacion[idx_p2].copy()
            fit_p2 = fitnesses[idx_p2]

            if random.random() < params.crossover_rate:
                diff = np.abs(p1 - p2)
                min_val = np.minimum(p1, p2) - params.blx_alpha * diff
                max_val = np.maximum(p1, p2) + params.blx_alpha * diff
                h1 = np.random.uniform(min_val, max_val)
                h2 = np.random.uniform(min_val, max_val)
            else:
                h1, h2 = p1.copy(), p2.copy()

            # Ruido térmico guiado
            h1 += np.random.normal(0, max(1e-5, sigma_t) * (ub - lb), size=dim)
            h2 += np.random.normal(0, max(1e-5, sigma_t) * (ub - lb), size=dim)

            h1 = np.clip(h1, lb, ub)
            h2 = np.clip(h2, lb, ub)

            v1 = float(func(h1))
            v2 = float(func(h2))

            delta1 = v1 - fit_p1
            if delta1 <= 0 or random.random() < math.exp(-delta1 / max(1e-9, T_t)):
                nueva_pob.append(h1)
                nuevos_fit.append(v1)
            else:
                nueva_pob.append(p1)
                nuevos_fit.append(fit_p1)

            if len(nueva_pob) < pop_size:
                delta2 = v2 - fit_p2
                if delta2 <= 0 or random.random() < math.exp(-delta2 / max(1e-9, T_t)):
                    nueva_pob.append(h2)
                    nuevos_fit.append(v2)
                else:
                    nueva_pob.append(p2)
                    nuevos_fit.append(fit_p2)

        poblacion = np.array(nueva_pob)
        fitnesses = nuevos_fit

        iter_best_idx = int(np.argmin(fitnesses))
        fit_iter_best = fitnesses[iter_best_idx]

        if fit_iter_best < mejor_val:
            mejor_val = fit_iter_best
            mejor_sol = poblacion[iter_best_idx].copy()

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)

        dtw_info = {}
        if monitor is not None:
            status = monitor.update(-mejor_val)
            dtw_info = status.copy()
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                dtw_info_hist.append(dtw_info)
                if verbose:
                    print(f"    [Stagnation] Fire #{stag_fires} @ gen {it + 1} -> ABORT")
                break
        dtw_info_hist.append(dtw_info)

    return MDGGASAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
        dtw_info_hist    = dtw_info_hist,
        historial_div    = historial_div,
    )
