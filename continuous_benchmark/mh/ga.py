"""
continuous_benchmark/mh/ga.py
-----------------------------
Algoritmo Genetico (GA) para funciones continuas (minimizacion).

Operadores:
  - Seleccion por torneo
  - Cruce BLX-alpha (blend crossover) – adecuado para dominio continuo
  - Mutacion gaussiana
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dtw_stagnation import StagnationConfig, StagnationMonitor


@dataclass
class GAParams:
    pop_size        : int   = 50
    generations     : int   = 300
    epochs          : int   = 1
    elitism         : int   = 2
    tournament_size : int   = 3
    crossover_rate  : float = 0.85
    mutation_rate   : float = 0.10
    mutation_sigma  : float = 0.1   # desviacion estandar relativa al rango
    blx_alpha       : float = 0.5   # parametro del cruce BLX-alpha
    injection_mode  : str   = "random"
    use_stagnation  : bool  = True
    stag_cfg        : StagnationConfig | None = None


@dataclass
class GAEpochResult:
    epoch_idx        : int
    mejor_valor      : float
    generaciones     : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[float] = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)


# ── Operadores geneticos continuos ────────────────────────────────────────────

def _torneo(poblacion: np.ndarray, fitnesses: np.ndarray, k: int) -> np.ndarray:
    indices = random.sample(range(len(poblacion)), k)
    mejor = min(indices, key=lambda i: fitnesses[i])
    return poblacion[mejor].copy()


def _cruce_blx(p1: np.ndarray, p2: np.ndarray, alpha: float,
               lb: float, ub: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.abs(p1 - p2)
    low  = np.minimum(p1, p2) - alpha * d
    high = np.maximum(p1, p2) + alpha * d
    hijo_a = np.random.uniform(low, high)
    hijo_b = np.random.uniform(low, high)
    hijo_a = np.clip(hijo_a, lb, ub)
    hijo_b = np.clip(hijo_b, lb, ub)
    return hijo_a, hijo_b


def _mutacion_gaussiana(ind: np.ndarray, rate: float, sigma: float,
                        lb: float, ub: float) -> np.ndarray:
    mask = np.random.rand(len(ind)) < rate
    rango = ub - lb
    ind[mask] += np.random.normal(0, sigma * rango, size=int(mask.sum()))
    return np.clip(ind, lb, ub)


def _mutar_solucion(sol: np.ndarray, lb: float, ub: float, n_dim: int) -> np.ndarray:
    copia = sol.copy()
    n_perturb = random.randint(1, max(1, n_dim // 10))
    for idx in random.sample(range(n_dim), n_perturb):
        copia[idx] = np.random.uniform(lb, ub)
    return copia


# ── Epoch ─────────────────────────────────────────────────────────────────────

def ejecutar_epoch(
    func,
    params    : GAParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> GAEpochResult:

    n = func.n_dim
    lb, ub = func.lb, func.ub

    poblacion = np.random.uniform(lb, ub, size=(params.pop_size, n))
    fitnesses = np.array([func.func(p) for p in poblacion])

    # Inyeccion
    if sol_inyectada is not None:
        sol_rep = np.clip(sol_inyectada, lb, ub)
        val_rep = func.func(sol_rep)
        mode = params.injection_mode
        if mode == "random":
            peor_idx = np.argmax(fitnesses)
            poblacion[peor_idx] = sol_rep; fitnesses[peor_idx] = val_rep
        elif mode == "mutated":
            poblacion[0] = sol_rep; fitnesses[0] = val_rep
            for i in range(1, params.pop_size):
                msol = _mutar_solucion(sol_rep, lb, ub, n)
                poblacion[i] = msol; fitnesses[i] = func.func(msol)
        elif mode == "mixed":
            poblacion[0] = sol_rep; fitnesses[0] = val_rep
            for i in range(1, params.pop_size // 2):
                msol = _mutar_solucion(sol_rep, lb, ub, n)
                poblacion[i] = msol; fitnesses[i] = func.func(msol)

    mejor_idx = np.argmin(fitnesses)
    mejor_val = fitnesses[mejor_idx]
    mejor_sol = poblacion[mejor_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for gen in range(params.generations):
        # Elitismo
        elite_idx  = np.argsort(fitnesses)[:params.elitism]
        nueva_pob  = list(poblacion[elite_idx])
        nuevos_fit = list(fitnesses[elite_idx])

        # Reproduccion
        while len(nueva_pob) < params.pop_size:
            padre_a = _torneo(poblacion, fitnesses, params.tournament_size)
            padre_b = _torneo(poblacion, fitnesses, params.tournament_size)

            if random.random() < params.crossover_rate:
                hijo_a, hijo_b = _cruce_blx(padre_a, padre_b, params.blx_alpha, lb, ub)
            else:
                hijo_a, hijo_b = padre_a.copy(), padre_b.copy()

            hijo_a = _mutacion_gaussiana(hijo_a, params.mutation_rate, params.mutation_sigma, lb, ub)
            hijo_b = _mutacion_gaussiana(hijo_b, params.mutation_rate, params.mutation_sigma, lb, ub)

            nueva_pob.append(hijo_a)
            nuevos_fit.append(func.func(hijo_a))
            nueva_pob.append(hijo_b)
            nuevos_fit.append(func.func(hijo_b))

        poblacion = np.array(nueva_pob[:params.pop_size])
        fitnesses = np.array(nuevos_fit[:params.pop_size])

        gen_mejor_idx = np.argmin(fitnesses)
        fit_gen_actual = fitnesses[gen_mejor_idx]
        if fit_gen_actual < mejor_val:
            mejor_val = fit_gen_actual
            mejor_sol = poblacion[gen_mejor_idx].copy()

        historial.append(mejor_val)
        historial_inst.append(fit_gen_actual)

        if monitor is not None:
            status = monitor.update(-mejor_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [GA Stagnation] Fire #{stag_fires} @ gen {gen} -> ABORT")
                break

    return GAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        generaciones     = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol.tolist(),
        dtw_deltas       = dtw_deltas,
    )
