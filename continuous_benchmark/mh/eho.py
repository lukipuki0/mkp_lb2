"""
continuous_benchmark/mh/eho.py
------------------------------
Elk Herd Optimizer (EHO) para funciones continuas (minimizacion).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dtw_stagnation import StagnationConfig, StagnationMonitor


@dataclass
class EHOParams:
    pop_size       : int   = 30
    iterations     : int   = 300
    epochs         : int   = 1
    bull_ratio     : float = 0.2
    injection_mode : str   = "random"
    use_stagnation : bool  = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class EHOEpochResult:
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[float] = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)


def _mutar_solucion(sol: np.ndarray, lb: float, ub: float, n_dim: int) -> np.ndarray:
    copia = sol.copy()
    n_perturb = random.randint(1, max(1, n_dim // 10))
    for idx in random.sample(range(n_dim), n_perturb):
        copia[idx] = np.random.uniform(lb, ub)
    return copia


def ejecutar_epoch(
    func,
    params    : EHOParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> EHOEpochResult:

    n = func.n_dim
    lb, ub = func.lb, func.ub
    pop_size = params.pop_size
    num_bulls = max(1, round(pop_size * params.bull_ratio))

    posiciones = np.random.uniform(lb, ub, size=(pop_size, n))
    fitnesses  = np.array([func.func(p) for p in posiciones])

    # Inyeccion
    if sol_inyectada is not None:
        sol_rep = np.clip(sol_inyectada, lb, ub)
        val_rep = func.func(sol_rep)
        mode = params.injection_mode
        if mode == "random":
            peor_idx = np.argmax(fitnesses)
            posiciones[peor_idx] = sol_rep; fitnesses[peor_idx] = val_rep
        elif mode == "mutated":
            posiciones[0] = sol_rep; fitnesses[0] = val_rep
            for i in range(1, pop_size):
                msol = _mutar_solucion(sol_rep, lb, ub, n)
                posiciones[i] = msol; fitnesses[i] = func.func(msol)
        elif mode == "mixed":
            posiciones[0] = sol_rep; fitnesses[0] = val_rep
            for i in range(1, pop_size // 2):
                msol = _mutar_solucion(sol_rep, lb, ub, n)
                posiciones[i] = msol; fitnesses[i] = func.func(msol)

    mejor_idx = np.argmin(fitnesses)
    mejor_val = fitnesses[mejor_idx]
    mejor_sol = posiciones[mejor_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        # Ordenar por fitness (ascendente = mejor primero en minimizacion)
        sorted_indices = np.argsort(fitnesses)

        bull_indices = sorted_indices[:num_bulls]

        # Seleccion por ruleta para familias (invertida para minimizacion)
        bull_fitness = fitnesses[bull_indices]
        # Invertir: los menores fitness obtienen mayor probabilidad
        max_fit = np.max(bull_fitness)
        inverted = max_fit - bull_fitness + 1e-10
        total = np.sum(inverted)
        selection_probs = inverted / total if total > 0 else np.ones(num_bulls) / num_bulls

        Families = np.zeros(pop_size, dtype=int)
        for rank in range(num_bulls, pop_size):
            female_index = sorted_indices[rank]
            selected_bull = np.random.choice(bull_indices, p=selection_probs)
            Families[female_index] = selected_bull

        # Reproduccion
        offspring_pos = []
        offspring_fit = []

        for i in range(pop_size):
            if i in bull_indices:
                # Bull
                h = np.random.randint(0, pop_size)
                alpha = np.random.rand()
                new_pos = posiciones[i] + alpha * (posiciones[h] - posiciones[i])
            else:
                # Female
                male_index = Families[i]
                h = np.random.randint(0, num_bulls)
                random_bull = bull_indices[h]
                gamma_vec = np.random.uniform(-2, 2, size=n)
                beta = 1.0
                new_pos = (posiciones[i]
                           + beta * (posiciones[male_index] - posiciones[i])
                           + gamma_vec * (posiciones[random_bull] - posiciones[i]))

            new_pos = np.clip(new_pos, lb, ub)
            new_val = func.func(new_pos)
            offspring_pos.append(new_pos)
            offspring_fit.append(new_val)

        # Merge padres + hijos y seleccionar los mejores
        merged_pos = np.concatenate([posiciones, np.array(offspring_pos)])
        merged_fit = np.concatenate([fitnesses, np.array(offspring_fit)])

        # Ascendente: los mejores (menores) primero
        selected = np.argsort(merged_fit)[:pop_size]
        posiciones = merged_pos[selected]
        fitnesses  = merged_fit[selected]

        mejor_actual_val = fitnesses[0]
        if mejor_actual_val < mejor_val:
            mejor_val = mejor_actual_val
            mejor_sol = posiciones[0].copy()

        historial.append(mejor_val)
        historial_inst.append(mejor_actual_val)

        if monitor is not None:
            status = monitor.update(-mejor_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [EHO Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return EHOEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol.tolist(),
        dtw_deltas       = dtw_deltas,
    )
