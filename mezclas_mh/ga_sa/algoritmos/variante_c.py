"""
mezclas_mh/ga_sa/algoritmos/variante_c.py
-----------------------------------------
Variante C: Conmutación Dinámica por Diversidad Poblacional (GA ↔ SA Multi-Agente).

Lógica:
  1. Monitorea en cada generación la distancia espacial normalizada al centroide:
     Div(t) = (1/N) * sum(|| x_i - x_mean ||)
     Div_norm(t) = Div(t) / Div(0)
  2. Compara Div_norm(t) frente a un umbral adaptativo decreciente umbral(t).
  3. Decisión de Régimen:
     - Si Div_norm(t) <= umbral(t) (Población sobre-agrupada):
       Activa SA Multi-Agente a Alta Temperatura sobre toda la población para
       inyectar perturbaciones amplias y romper el agrupamiento prematuro.
     - Si Div_norm(t) > umbral(t) (Buena diversidad):
       Ejecuta GA estándar (cruce y mutación) para recombinación y convergencia rápida.

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
class VariantCParams:
    """Hiperparámetros de la Variante C (Switching por Diversidad GA-SA)."""
    pop_size        : int   = 50
    generations     : int   = 300
    epochs          : int   = 1
    elitism         : int   = 2
    tournament_size : int   = 3
    crossover_rate  : float = 0.85
    mutation_rate   : float = 0.05
    crossover_op    : str   = "uniform"
    mutation_op     : str   = "bitflip"
    # Parámetros de Diversidad
    umbral_init     : float = 0.60
    umbral_final    : float = 0.10
    # Parámetros de SA Shake
    T_shake         : float = 500.0
    sa_shake_steps  : int   = 5
    # Continuo
    blx_alpha       : float = 0.5
    mutation_sigma  : float = 0.1
    # Pipeline híbrido
    injection_mode  : str   = "mixed"
    use_stagnation  : bool  = True
    stag_cfg        : StagnationConfig | None = None


@dataclass
class VariantCEpochResult:
    """Resultado de un epoch de la Variante C."""
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
    historial_fases  : list[str]   = field(default_factory=list)  # "GA" o "SA_SHAKE"


# ── Helpers Diversidad y Discreto ─────────────────────────────────────────────

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
    distancias = np.linalg.norm(arr - centroide, axis=1)
    return float(np.mean(distancias))


def _sa_shake_mkp(
    poblacion: list[list[int]],
    fitnesses: list[float],
    inst: MKPInstance,
    steps: int,
    T: float,
) -> tuple[list[list[int]], list[float]]:
    """Aplica perturbaciones SA con criterio de Metrópolis a toda la población para romper estancamiento."""
    nueva_pob = []
    nuevos_fit = []
    for sol, val in zip(poblacion, fitnesses):
        actual_sol = list(sol)
        actual_val = val
        for _ in range(steps):
            vecino = list(actual_sol)
            num_flips = random.randint(2, max(2, inst.n // 10))
            for idx in random.sample(range(inst.n), num_flips):
                vecino[idx] = 1 - vecino[idx]
            vecino, val_v = reparar_solucion(vecino, inst)

            delta = val_v - actual_val  # Maximización
            if delta >= 0 or random.random() < math.exp(delta / max(1e-9, T)):
                actual_sol = vecino
                actual_val = val_v
        nueva_pob.append(actual_sol)
        nuevos_fit.append(actual_val)
    return nueva_pob, nuevos_fit


# ── Ejecución Discreta MKP ────────────────────────────────────────────────────

def ejecutar_epoch(
    inst          : MKPInstance,
    params        : VariantCParams,
    epoch_idx     : int = 0,
    verbose       : bool = True,
    sol_inyectada : list[int] | None = None,
) -> VariantCEpochResult:
    """Ejecuta la Variante C (Switching por Diversidad) para MKP."""
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

    historial       = []
    historial_inst  = []
    dtw_deltas      = []
    dtw_info_hist   = []
    historial_div   = []
    historial_fases = []
    stag_fires      = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.generations):
        # 1. Medir Diversidad y Umbral
        div_act = _calcular_diversidad_binaria(poblacion)
        div_norm = div_act / div_0
        historial_div.append(div_norm)

        frac = it / max(1, params.generations - 1)
        umbral_t = params.umbral_init * (1.0 - frac) + params.umbral_final * frac

        # 2. Conmutación de Régimen
        if div_norm <= umbral_t:
            # Fase SA SHAKE (dispersión térmica)
            historial_fases.append("SA_SHAKE")
            poblacion, fitnesses = _sa_shake_mkp(
                poblacion, fitnesses, inst,
                steps = params.sa_shake_steps,
                T     = params.T_shake * (1.0 - frac * 0.5),
            )
            # Reinsertar al mejor histórico si se perdió
            peor_idx = min(range(pop_size), key=lambda i: fitnesses[i])
            poblacion[peor_idx] = list(mejor_sol)
            fitnesses[peor_idx] = mejor_val
        else:
            # Fase GA (Recombinación estándar)
            historial_fases.append("GA")
            indices_ordenados = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
            nueva_pob = [list(poblacion[i]) for i in indices_ordenados[:params.elitism]]
            nuevos_fit = [fitnesses[i] for i in indices_ordenados[:params.elitism]]

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

    return VariantCEpochResult(
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
        historial_fases  = historial_fases,
    )


# ── Ejecución Continua (CEC2022 / HRES / Hypertuning) ─────────────────────────

def ejecutar_epoch_continuo(
    func      : Callable[[np.ndarray], float],
    dim       : int,
    bounds    : tuple[float, float],
    params    : VariantCParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> VariantCEpochResult:
    """Ejecuta la Variante C en optimización continua (minimización)."""
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

    historial       = []
    historial_inst  = []
    dtw_deltas      = []
    dtw_info_hist   = []
    historial_div   = []
    historial_fases = []
    stag_fires      = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.generations):
        centroide = np.mean(poblacion, axis=0)
        div_act = float(np.mean(np.linalg.norm(poblacion - centroide, axis=1)))
        div_norm = div_act / div_0
        historial_div.append(div_norm)

        frac = it / max(1, params.generations - 1)
        umbral_t = params.umbral_init * (1.0 - frac) + params.umbral_final * frac

        if div_norm <= umbral_t:
            # Fase SA SHAKE Continuo
            historial_fases.append("SA_SHAKE")
            T = params.T_shake * (1.0 - frac * 0.5)
            for i in range(pop_size):
                act_sol = poblacion[i].copy()
                act_val = fitnesses[i]
                for _ in range(params.sa_shake_steps):
                    vecino = np.clip(act_sol + np.random.normal(0, params.mutation_sigma * 2.0 * (ub - lb), size=dim), lb, ub)
                    v_vecino = float(func(vecino))
                    delta = v_vecino - act_val
                    if delta <= 0 or random.random() < math.exp(-delta / max(1e-9, T)):
                        act_sol = vecino.copy()
                        act_val = v_vecino
                poblacion[i] = act_sol
                fitnesses[i] = act_val

            peor_idx = int(np.argmax(fitnesses))
            poblacion[peor_idx] = mejor_sol.copy()
            fitnesses[peor_idx] = mejor_val
        else:
            # Fase GA Continuo
            historial_fases.append("GA")
            indices_ordenados = np.argsort(fitnesses)
            nueva_pob = [poblacion[i].copy() for i in indices_ordenados[:params.elitism]]
            nuevos_fit = [fitnesses[i] for i in indices_ordenados[:params.elitism]]

            while len(nueva_pob) < pop_size:
                t_idx1 = random.sample(range(pop_size), params.tournament_size)
                p1 = poblacion[min(t_idx1, key=lambda i: fitnesses[i])].copy()
                t_idx2 = random.sample(range(pop_size), params.tournament_size)
                p2 = poblacion[min(t_idx2, key=lambda i: fitnesses[i])].copy()

                if random.random() < params.crossover_rate:
                    diff = np.abs(p1 - p2)
                    min_val = np.minimum(p1, p2) - params.blx_alpha * diff
                    max_val = np.maximum(p1, p2) + params.blx_alpha * diff
                    h1 = np.random.uniform(min_val, max_val)
                    h2 = np.random.uniform(min_val, max_val)
                else:
                    h1, h2 = p1.copy(), p2.copy()

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

    return VariantCEpochResult(
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
        historial_fases  = historial_fases,
    )
