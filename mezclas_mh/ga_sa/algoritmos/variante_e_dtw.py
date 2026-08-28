"""
mezclas_mh/ga_sa/algoritmos/variante_e_dtw.py
---------------------------------------------
Variante E: DTW-Adaptive GA-SA (Control Adaptativo por DTW / DDTW).

Lógica:
  1. Un monitor DTW compara la serie temporal de fitness reciente contra
     patrones de progreso (Rampa) y estancamiento (Constante).
  2. Conmutación dinámica:
     - Si Delta_DTW >= theta_delta (Progreso activo): Opera GA estándar de alta eficiencia.
     - Si Delta_DTW < theta_delta (Desaceleración / Meseta): Se inyecta un ciclo de
       Simulated Annealing con Thermal Reheating (re-calentamiento térmico) para
       sacudir las soluciones y saltar barreras de energía.
  3. Annealing Blast (Mecanismo de Rescate):
     Ante alerta crítica (fire == True), se preserva el mejor individuo global y se aplica
     un reinicio térmico de alta temperatura al 40% peor de la población.

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
class DTWGASAParams:
    """Hiperparámetros de la Variante E (DTW-Adaptive GA-SA)."""
    pop_size        : int   = 50
    generations     : int   = 300
    epochs          : int   = 1
    elitism         : int   = 2
    tournament_size : int   = 3
    crossover_rate  : float = 0.85
    mutation_rate   : float = 0.05
    crossover_op    : str   = "uniform"
    mutation_op     : str   = "bitflip"
    # Parámetros Térmicos DTW
    T_base          : float = 200.0
    reheat_factor   : float = 3.0   # Multiplicador de temperatura ante desaceleración
    sa_blast_ratio  : float = 0.40  # Porcentaje de población a resetear en fire
    sa_boost_steps  : int   = 10
    # Continuo
    blx_alpha       : float = 0.5
    mutation_sigma  : float = 0.1
    # Pipeline híbrido
    injection_mode  : str   = "mixed"
    use_stagnation  : bool  = True
    stag_cfg        : StagnationConfig | None = None


@dataclass
class DTWGASAEpochResult:
    """Resultado de un epoch de DTW-Adaptive GA-SA."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[int] | np.ndarray = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)
    dtw_info_hist    : list[dict]  = field(default_factory=list)
    thermal_boosts   : int = 0


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


def _sa_reheat_step_mkp(
    sol: list[int],
    val: float,
    inst: MKPInstance,
    T: float,
    steps: int,
) -> tuple[list[int], float]:
    act_sol = list(sol)
    act_val = val
    for _ in range(steps):
        vecino = list(act_sol)
        k = random.randint(1, 3)
        for idx in random.sample(range(inst.n), k):
            vecino[idx] = 1 - vecino[idx]
        vecino, v_vec = reparar_solucion(vecino, inst)
        delta = v_vec - act_val
        if delta >= 0 or random.random() < math.exp(delta / max(1e-9, T)):
            act_sol = vecino
            act_val = v_vec
    return act_sol, act_val


# ── Ejecución Discreta MKP ────────────────────────────────────────────────────

def ejecutar_epoch(
    inst          : MKPInstance,
    params        : DTWGASAParams,
    epoch_idx     : int = 0,
    verbose       : bool = True,
    sol_inyectada : list[int] | None = None,
) -> DTWGASAEpochResult:
    """Ejecuta la Variante E (DTW-Adaptive GA-SA) para MKP."""
    pop_size = params.pop_size
    poblacion, fitnesses = _inicializar_poblacion_mkp(inst, pop_size)

    if sol_inyectada is not None:
        sol_rep, val_rep = reparar_solucion(list(sol_inyectada), inst)
        poblacion[0] = sol_rep
        fitnesses[0] = val_rep

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
    thermal_boosts = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.generations):
        # 1. Elitismo
        indices_ordenados = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        nueva_pob = [list(poblacion[i]) for i in indices_ordenados[:params.elitism]]
        nuevos_fit = [fitnesses[i] for i in indices_ordenados[:params.elitism]]

        # 2. Generación GA estándar
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

        # 3. Monitor DTW e Inyección Térmica Adaptativa
        dtw_info = {}
        if monitor is not None:
            status = monitor.update(mejor_val)
            dtw_info = status.copy()
            if status.get("ready"):
                delta_d = status.get("delta", 0.0)
                theta_d = status.get("theta_delta", 0.0)
                dtw_deltas.append(delta_d)

                # Desaceleración -> Reheating térmico
                if delta_d < theta_d:
                    thermal_boosts += 1
                    T_reheat = params.T_base * params.reheat_factor
                    top_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:10]
                    for idx_t in top_indices:
                        poblacion[idx_t], fitnesses[idx_t] = _sa_reheat_step_mkp(
                            poblacion[idx_t], fitnesses[idx_t], inst, T_reheat, steps=params.sa_boost_steps
                        )

            # Alerta crítica -> Annealing Blast
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [DTW Blast] Fire #{stag_fires} @ gen {it + 1} -> Re-inyección térmica")
                # Resetear el peor ratio de la población con alta temperatura
                n_blast = int(pop_size * params.sa_blast_ratio)
                peores_idx = sorted(range(pop_size), key=lambda i: fitnesses[i])[:n_blast]
                for idx_b in peores_idx:
                    sol_rand = [random.randint(0, 1) for _ in range(inst.n)]
                    sol_rand, val_rand = reparar_solucion(sol_rand, inst)
                    poblacion[idx_b] = sol_rand
                    fitnesses[idx_b] = val_rand

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)
        dtw_info_hist.append(dtw_info)

    return DTWGASAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
        dtw_info_hist    = dtw_info_hist,
        thermal_boosts   = thermal_boosts,
    )


# ── Ejecución Continua (CEC2022 / HRES / Hypertuning) ─────────────────────────

def ejecutar_epoch_continuo(
    func      : Callable[[np.ndarray], float],
    dim       : int,
    bounds    : tuple[float, float],
    params    : DTWGASAParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> DTWGASAEpochResult:
    """Ejecuta la Variante E en optimización continua (minimización)."""
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
    thermal_boosts = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.generations):
        # 1. Elitismo
        indices_ordenados = np.argsort(fitnesses)
        nueva_pob = [poblacion[i].copy() for i in indices_ordenados[:params.elitism]]
        nuevos_fit = [fitnesses[i] for i in indices_ordenados[:params.elitism]]

        # 2. Cruce BLX-alpha + Mutación
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

        # 3. Monitor DTW e Inyección Térmica Continua
        dtw_info = {}
        if monitor is not None:
            status = monitor.update(-mejor_val)
            dtw_info = status.copy()
            if status.get("ready"):
                delta_d = status.get("delta", 0.0)
                theta_d = status.get("theta_delta", 0.0)
                dtw_deltas.append(delta_d)

                if delta_d < theta_d:
                    thermal_boosts += 1
                    T_reheat = params.T_base * params.reheat_factor
                    top_indices = np.argsort(fitnesses)[:10]
                    for idx_t in top_indices:
                        act_s = poblacion[idx_t].copy()
                        act_v = fitnesses[idx_t]
                        for _ in range(params.sa_boost_steps):
                            vec = np.clip(act_s + np.random.normal(0, params.mutation_sigma * 1.5 * (ub - lb), size=dim), lb, ub)
                            v_vec = float(func(vec))
                            delta = v_vec - act_v
                            if delta <= 0 or random.random() < math.exp(-delta / max(1e-9, T_reheat)):
                                act_s = vec.copy()
                                act_v = v_vec
                        poblacion[idx_t] = act_s
                        fitnesses[idx_t] = act_v

            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [DTW Blast] Fire #{stag_fires} @ gen {it + 1} -> Re-inyección térmica continua")
                n_blast = int(pop_size * params.sa_blast_ratio)
                peores_idx = np.argsort(fitnesses)[-n_blast:]
                for idx_b in peores_idx:
                    poblacion[idx_b] = np.random.uniform(lb, ub, dim)
                    fitnesses[idx_b] = float(func(poblacion[idx_b]))

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)
        dtw_info_hist.append(dtw_info)

    return DTWGASAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
        dtw_info_hist    = dtw_info_hist,
        thermal_boosts   = thermal_boosts,
    )
