"""
mezclas_mh/ga_sa/algoritmos/variante_a.py
-----------------------------------------
Variante A: Algoritmo Memético Clásico (GA con Refinamiento Local SA).

Lógica:
  1. En cada generación ejecuta selección, cruce y mutación estándar del GA.
  2. Sobre los Top-K individuos (o sobre la población élite), ejecuta un Micro-SA
     de corta duración con perturbación en vecindario y criterio de aceptación de Metrópolis.
  3. Reinserta las soluciones mejoradas en la población.

Soporta:
  - Problema Discreto MKP (reparación factible).
  - Problema Continuo / HRES / Hypertuning / CEC2022 (con límites y función objetivo genérica).
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


# ── Hiperparámetros y Dataclasses ─────────────────────────────────────────────

@dataclass
class VariantAParams:
    """Hiperparámetros de la Variante A (Memético GA + Micro-SA)."""
    pop_size        : int   = 50
    generations     : int   = 300
    epochs          : int   = 1
    elitism         : int   = 2
    tournament_size : int   = 3
    crossover_rate  : float = 0.85
    mutation_rate   : float = 0.05
    crossover_op    : str   = "uniform"   # "uniform" | "1point" | "2point"
    mutation_op     : str   = "bitflip"   # "bitflip" | "swap"
    # Parámetros Micro-SA
    sa_k_elite      : int   = 5           # Número de mejores individuos a refinar con SA
    sa_steps        : int   = 15          # Pasos de búsqueda local SA por individuo
    T_inicial       : float = 100.0       # Temperatura inicial de SA
    alpha_sa        : float = 0.90        # Enfriamiento geométrico
    # Parámetros Continuos
    blx_alpha       : float = 0.5
    mutation_sigma  : float = 0.1
    # Pipeline híbrido
    injection_mode  : str   = "mixed"
    use_stagnation  : bool  = True
    stag_cfg        : StagnationConfig | None = None


@dataclass
class VariantAEpochResult:
    """Resultado de un epoch de la Variante A."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[int] | np.ndarray = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)
    dtw_info_hist    : list[dict]  = field(default_factory=list)


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


def _mutar_solucion_mkp(sol: list[int], inst: MKPInstance) -> tuple[list[int], float]:
    copia = list(sol)
    n = len(copia)
    n_flips = random.randint(1, 3)
    indices = random.sample(range(n), min(n_flips, n))
    for idx in indices:
        copia[idx] = 1 - copia[idx]
    return reparar_solucion(copia, inst)


def _micro_sa_mkp(
    sol_inicial: list[int],
    val_inicial: float,
    inst: MKPInstance,
    steps: int,
    T0: float,
    alpha: float,
) -> tuple[list[int], float]:
    """Ejecuta un ciclo corto de Simulated Annealing sobre una solución MKP."""
    actual_sol = list(sol_inicial)
    actual_val = val_inicial
    mejor_sol  = list(sol_inicial)
    mejor_val  = val_inicial
    T = T0

    for _ in range(steps):
        # Generar vecino con 1-flip o 2-flip
        vecino = list(actual_sol)
        k = random.randint(1, 2)
        idx_flips = random.sample(range(inst.n), k)
        for idx in idx_flips:
            vecino[idx] = 1 - vecino[idx]
        vecino, val_vecino = reparar_solucion(vecino, inst)

        # Criterio de Metrópolis (Maximización en MKP)
        delta = val_vecino - actual_val
        if delta >= 0:
            actual_sol = vecino
            actual_val = val_vecino
            if val_vecino > mejor_val:
                mejor_sol = list(vecino)
                mejor_val = val_vecino
        else:
            p_acept = math.exp(delta / max(1e-9, T))
            if random.random() < p_acept:
                actual_sol = vecino
                actual_val = val_vecino

        T *= alpha

    return mejor_sol, mejor_val


# ── Ejecución Discreta MKP ────────────────────────────────────────────────────

def ejecutar_epoch(
    inst          : MKPInstance,
    params        : VariantAParams,
    epoch_idx     : int = 0,
    verbose       : bool = True,
    sol_inyectada : list[int] | None = None,
) -> VariantAEpochResult:
    """Ejecuta la Variante A (Memético GA + SA) para el problema MKP."""
    n = inst.n
    pop_size = params.pop_size
    poblacion, fitnesses = _inicializar_poblacion_mkp(inst, pop_size)

    # Inyección de solución si existe
    if sol_inyectada is not None:
        sol_rep, val_rep = reparar_solucion(list(sol_inyectada), inst)
        if params.injection_mode == "random":
            peor_idx = min(range(pop_size), key=lambda i: fitnesses[i])
            poblacion[peor_idx] = sol_rep
            fitnesses[peor_idx] = val_rep
        elif params.injection_mode in ("mutated", "mixed"):
            poblacion[0] = sol_rep
            fitnesses[0] = val_rep
            n_mut = pop_size // 2 if params.injection_mode == "mixed" else pop_size
            for i in range(1, n_mut):
                msol, mval = _mutar_solucion_mkp(sol_rep, inst)
                poblacion[i] = msol
                fitnesses[i] = mval

    cross_fn = get_crossover(params.crossover_op)
    mut_fn   = get_mutation(params.mutation_op)

    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    mejor_val = fitnesses[best_idx]
    mejor_sol = list(poblacion[best_idx])

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    dtw_info_hist  = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.generations):
        # 1. Elitismo
        indices_ordenados = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        nueva_pob = [list(poblacion[i]) for i in indices_ordenados[:params.elitism]]
        nuevos_fit = [fitnesses[i] for i in indices_ordenados[:params.elitism]]

        # 2. Selección, Cruce y Mutación GA
        while len(nueva_pob) < pop_size:
            p1 = torneo(poblacion, fitnesses, params.tournament_size)
            p2 = torneo(poblacion, fitnesses, params.tournament_size)

            if random.random() < params.crossover_rate:
                h1, h2 = cross_fn(p1, p2)
            else:
                h1, h2 = list(p1), list(p2)

            h1 = mut_fn(h1, params.mutation_rate)
            h2 = mut_fn(h2, params.mutation_rate)

            h1, v1 = reparar_solucion(h1, inst)
            h2, v2 = reparar_solucion(h2, inst)

            nueva_pob.append(h1)
            nuevos_fit.append(v1)
            if len(nueva_pob) < pop_size:
                nueva_pob.append(h2)
                nuevos_fit.append(v2)

        # 3. Refinamiento Micro-SA sobre Top-K élites
        top_k_indices = sorted(range(pop_size), key=lambda i: nuevos_fit[i], reverse=True)[:params.sa_k_elite]
        T_actual_sa = params.T_inicial * (1.0 - it / max(1, params.generations))

        for idx_k in top_k_indices:
            sol_ref, val_ref = _micro_sa_mkp(
                sol_inicial = nueva_pob[idx_k],
                val_inicial = nuevos_fit[idx_k],
                inst        = inst,
                steps       = params.sa_steps,
                T0          = max(0.1, T_actual_sa),
                alpha       = params.alpha_sa,
            )
            nueva_pob[idx_k] = sol_ref
            nuevos_fit[idx_k] = val_ref

        poblacion = nueva_pob
        fitnesses = nuevos_fit

        iter_best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        fit_iter_best = fitnesses[iter_best_idx]

        if fit_iter_best > mejor_val:
            mejor_val = fit_iter_best
            mejor_sol = list(poblacion[iter_best_idx])

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)

        # Stagnation monitor
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

    return VariantAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
        dtw_info_hist    = dtw_info_hist,
    )


# ── Ejecución Continua (CEC2022 / HRES / Hypertuning) ─────────────────────────

def _micro_sa_continuo(
    sol_inicial : np.ndarray,
    val_inicial : float,
    func        : Callable[[np.ndarray], float],
    dim         : int,
    bounds      : tuple[float, float],
    steps       : int,
    T0          : float,
    alpha       : float,
    sigma       : float,
) -> tuple[np.ndarray, float]:
    """Micro-SA en espacio continuo (minimizacion)."""
    actual_sol = sol_inicial.copy()
    actual_val = val_inicial
    mejor_sol  = sol_inicial.copy()
    mejor_val  = val_inicial
    T = T0
    lb, ub = bounds

    for _ in range(steps):
        perturb = np.random.normal(0, sigma * (ub - lb), size=dim)
        vecino = np.clip(actual_sol + perturb, lb, ub)
        val_vecino = float(func(vecino))

        delta = val_vecino - actual_val  # Minimizacion: deseamos delta <= 0
        if delta <= 0:
            actual_sol = vecino.copy()
            actual_val = val_vecino
            if val_vecino < mejor_val:
                mejor_sol = vecino.copy()
                mejor_val = val_vecino
        else:
            p_acept = math.exp(-delta / max(1e-9, T))
            if random.random() < p_acept:
                actual_sol = vecino.copy()
                actual_val = val_vecino

        T *= alpha

    return mejor_sol, mejor_val


def ejecutar_epoch_continuo(
    func      : Callable[[np.ndarray], float],
    dim       : int,
    bounds    : tuple[float, float],
    params    : VariantAParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> VariantAEpochResult:
    """Ejecuta la Variante A en optimización continua (minimización)."""
    lb, ub = bounds
    pop_size = params.pop_size

    # Inicializar población continua
    poblacion = np.random.uniform(lb, ub, (pop_size, dim))
    fitnesses = [float(func(ind)) for ind in poblacion]

    if sol_inyectada is not None:
        poblacion[0] = np.clip(sol_inyectada.copy(), lb, ub)
        fitnesses[0] = float(func(poblacion[0]))

    best_idx = int(np.argmin(fitnesses))
    mejor_val = fitnesses[best_idx]
    mejor_sol = poblacion[best_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    dtw_info_hist  = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.generations):
        # 1. Elitismo
        indices_ordenados = np.argsort(fitnesses)
        nueva_pob = [poblacion[i].copy() for i in indices_ordenados[:params.elitism]]
        nuevos_fit = [fitnesses[i] for i in indices_ordenados[:params.elitism]]

        # 2. Recombinación BLX-alpha + Mutación Gaussiana
        while len(nueva_pob) < pop_size:
            # Torneo (minimización)
            t_idx1 = random.sample(range(pop_size), params.tournament_size)
            p1 = poblacion[min(t_idx1, key=lambda i: fitnesses[i])].copy()
            t_idx2 = random.sample(range(pop_size), params.tournament_size)
            p2 = poblacion[min(t_idx2, key=lambda i: fitnesses[i])].copy()

            if random.random() < params.crossover_rate:
                # BLX-alpha
                diff = np.abs(p1 - p2)
                min_val = np.minimum(p1, p2) - params.blx_alpha * diff
                max_val = np.maximum(p1, p2) + params.blx_alpha * diff
                h1 = np.random.uniform(min_val, max_val)
                h2 = np.random.uniform(min_val, max_val)
            else:
                h1, h2 = p1.copy(), p2.copy()

            # Mutación Gaussiana
            if random.random() < params.mutation_rate:
                h1 += np.random.normal(0, params.mutation_sigma * (ub - lb), size=dim)
            if random.random() < params.mutation_rate:
                h2 += np.random.normal(0, params.mutation_sigma * (ub - lb), size=dim)

            h1 = np.clip(h1, lb, ub)
            h2 = np.clip(h2, lb, ub)

            nueva_pob.append(h1)
            nuevos_fit.append(float(func(h1)))
            if len(nueva_pob) < pop_size:
                nueva_pob.append(h2)
                nuevos_fit.append(float(func(h2)))

        poblacion = np.array(nueva_pob)
        fitnesses = nuevos_fit

        # 3. Micro-SA continuo sobre Top-K élites
        top_k_indices = np.argsort(fitnesses)[:params.sa_k_elite]
        T_actual_sa = params.T_inicial * (1.0 - it / max(1, params.generations))

        for idx_k in top_k_indices:
            sol_ref, val_ref = _micro_sa_continuo(
                sol_inicial = poblacion[idx_k],
                val_inicial = fitnesses[idx_k],
                func        = func,
                dim         = dim,
                bounds      = bounds,
                steps       = params.sa_steps,
                T0          = max(0.01, T_actual_sa),
                alpha       = params.alpha_sa,
                sigma       = params.mutation_sigma,
            )
            poblacion[idx_k] = sol_ref
            fitnesses[idx_k] = val_ref

        iter_best_idx = int(np.argmin(fitnesses))
        fit_iter_best = fitnesses[iter_best_idx]

        if fit_iter_best < mejor_val:
            mejor_val = fit_iter_best
            mejor_sol = poblacion[iter_best_idx].copy()

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)

        # Monitor de estancamiento (minimizacion -> valor negativo para rampa)
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

    return VariantAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
        dtw_info_hist    = dtw_info_hist,
    )
