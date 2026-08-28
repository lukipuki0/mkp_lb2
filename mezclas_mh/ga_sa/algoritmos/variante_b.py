"""
mezclas_mh/ga_sa/algoritmos/variante_b.py
-----------------------------------------
Variante B: Enfriamiento Térmico Global (Annealing-Guided GA).

Lógica:
  1. Un programa de enfriamiento geométrico global T(t) = T_0 * (T_f / T_0) ** (t / MaxGen)
     gobierna las probabilidades y comportamiento del GA.
  2. Modulación dinámica de la tasa de mutación:
     p_m(t) = p_m_min + (p_m_max - p_m_min) * (T(t) / T_0)
  3. Filtro de reemplazo generacional por Criterio de Metrópolis:
     Hijos peores que sus progenitores son aceptados probabilísticamente si r < exp(-Delta / T(t)),
     fomentando exploración temprana y convergencia elitista estricta en etapas tardías.

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
class VariantBParams:
    """Hiperparámetros de la Variante B (GA Térmico Global)."""
    pop_size        : int   = 50
    generations     : int   = 300
    epochs          : int   = 1
    elitism         : int   = 2
    tournament_size : int   = 3
    crossover_rate  : float = 0.85
    mutation_rate_max: float = 0.15  # Tasa de mutación a T_inicial
    mutation_rate_min: float = 0.01  # Tasa de mutación a T_final
    crossover_op    : str   = "uniform"
    mutation_op     : str   = "bitflip"
    # Parámetros Térmicos
    T_inicial       : float = 1000.0
    T_final         : float = 0.01
    # Continuo
    blx_alpha       : float = 0.5
    mutation_sigma  : float = 0.1
    # Pipeline híbrido
    injection_mode  : str   = "mixed"
    use_stagnation  : bool  = True
    stag_cfg        : StagnationConfig | None = None


@dataclass
class VariantBEpochResult:
    """Resultado de un epoch de la Variante B."""
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


# ── Ejecución Discreta MKP ────────────────────────────────────────────────────

def ejecutar_epoch(
    inst          : MKPInstance,
    params        : VariantBParams,
    epoch_idx     : int = 0,
    verbose       : bool = True,
    sol_inyectada : list[int] | None = None,
) -> VariantBEpochResult:
    """Ejecuta la Variante B (GA con Enfriamiento Térmico Global) para MKP."""
    n = inst.n
    pop_size = params.pop_size
    poblacion, fitnesses = _inicializar_poblacion_mkp(inst, pop_size)

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
        # 1. Calcular Temperatura Actual y Tasa de Mutación Térmica
        frac = it / max(1, params.generations - 1)
        T_t = params.T_inicial * ((params.T_final / max(1e-9, params.T_inicial)) ** frac)
        p_m_t = params.mutation_rate_min + (params.mutation_rate_max - params.mutation_rate_min) * (T_t / max(1e-9, params.T_inicial))

        # 2. Elitismo
        indices_ordenados = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        nueva_pob = [list(poblacion[i]) for i in indices_ordenados[:params.elitism]]
        nuevos_fit = [fitnesses[i] for i in indices_ordenados[:params.elitism]]

        # 3. Cruce + Mutación con Reemplazo por Metrópolis
        while len(nueva_pob) < pop_size:
            idx1 = torneo(poblacion, fitnesses, params.tournament_size, return_index=True) if "return_index" in torneo.__code__.co_varnames else None
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

            # Criterio Metrópolis frente a los padres para ingresar
            v_padre1 = inst.evaluar(p1) if hasattr(inst, "evaluar") else np.dot(p1, inst.p)
            delta1 = v1 - v_padre1
            if delta1 >= 0 or random.random() < math.exp(delta1 / max(1e-9, T_t)):
                nueva_pob.append(h1)
                nuevos_fit.append(v1)
            else:
                nueva_pob.append(list(p1))
                nuevos_fit.append(v_padre1)

            if len(nueva_pob) < pop_size:
                v_padre2 = inst.evaluar(p2) if hasattr(inst, "evaluar") else np.dot(p2, inst.p)
                delta2 = v2 - v_padre2
                if delta2 >= 0 or random.random() < math.exp(delta2 / max(1e-9, T_t)):
                    nueva_pob.append(h2)
                    nuevos_fit.append(v2)
                else:
                    nueva_pob.append(list(p2))
                    nuevos_fit.append(v_padre2)

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

    return VariantBEpochResult(
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

def ejecutar_epoch_continuo(
    func      : Callable[[np.ndarray], float],
    dim       : int,
    bounds    : tuple[float, float],
    params    : VariantBParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> VariantBEpochResult:
    """Ejecuta la Variante B en optimización continua (minimización)."""
    lb, ub = bounds
    pop_size = params.pop_size

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
        frac = it / max(1, params.generations - 1)
        T_t = params.T_inicial * ((params.T_final / max(1e-9, params.T_inicial)) ** frac)
        p_m_t = params.mutation_rate_min + (params.mutation_rate_max - params.mutation_rate_min) * (T_t / max(1e-9, params.T_inicial))

        # 1. Elitismo
        indices_ordenados = np.argsort(fitnesses)
        nueva_pob = [poblacion[i].copy() for i in indices_ordenados[:params.elitism]]
        nuevos_fit = [fitnesses[i] for i in indices_ordenados[:params.elitism]]

        # 2. Cruce BLX-alpha + Mutación + Aceptación de Metrópolis
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

            # Perturbación adaptativa por temperatura
            sigma_t = params.mutation_sigma * (T_t / max(1e-9, params.T_inicial))
            if random.random() < p_m_t:
                h1 += np.random.normal(0, max(1e-5, sigma_t) * (ub - lb), size=dim)
            if random.random() < p_m_t:
                h2 += np.random.normal(0, max(1e-5, sigma_t) * (ub - lb), size=dim)

            h1 = np.clip(h1, lb, ub)
            h2 = np.clip(h2, lb, ub)

            v1 = float(func(h1))
            v2 = float(func(h2))

            # Filtro Metrópolis frente a los padres
            delta1 = v1 - fit_p1  # Minimizacion: queremos delta1 <= 0
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

    return VariantBEpochResult(
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
