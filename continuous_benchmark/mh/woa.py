"""
continuous_benchmark/mh/woa.py
------------------------------
Whale Optimization Algorithm (WOA) para funciones continuas (minimizacion).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dtw_stagnation import StagnationConfig, StagnationMonitor


@dataclass
class WOAParams:
    pop_size       : int   = 30
    iterations     : int   = 300
    epochs         : int   = 1
    b              : float = 1.0   # constante espiral
    injection_mode : str   = "random"
    use_stagnation : bool  = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class WOAEpochResult:
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[float] = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)
    dtw_info_hist    : list[dict]  = field(default_factory=list)


def _mutar_solucion(sol: np.ndarray, lb: float, ub: float, n_dim: int) -> np.ndarray:
    copia = sol.copy()
    n_perturb = random.randint(1, max(1, n_dim // 10))
    for idx in random.sample(range(n_dim), n_perturb):
        copia[idx] = np.random.uniform(lb, ub)
    return copia


def ejecutar_epoch(
    func,
    params    : WOAParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> WOAEpochResult:

    n = func.n_dim
    lb, ub = func.lb, func.ub

    posiciones = np.random.uniform(lb, ub, size=(params.pop_size, n))
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
            for i in range(1, params.pop_size):
                msol = _mutar_solucion(sol_rep, lb, ub, n)
                posiciones[i] = msol; fitnesses[i] = func.func(msol)
        elif mode == "mixed":
            posiciones[0] = sol_rep; fitnesses[0] = val_rep
            for i in range(1, params.pop_size // 2):
                msol = _mutar_solucion(sol_rep, lb, ub, n)
                posiciones[i] = msol; fitnesses[i] = func.func(msol)

    best_idx  = np.argmin(fitnesses)
    mejor_val = fitnesses[best_idx]
    mejor_sol = posiciones[best_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    dtw_info_hist  = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        a  = 2.0 - it * (2.0 / params.iterations)
        a2 = -1.0 + it * (-1.0 / params.iterations)

        X_best = posiciones[best_idx].copy()

        for i in range(params.pop_size):
            r1, r2 = random.random(), random.random()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = random.random()

            if p < 0.5:
                if abs(A) < 1.0:
                    D = np.abs(C * X_best - posiciones[i])
                    X_new = X_best - A * D
                else:
                    rand_idx = random.randint(0, params.pop_size - 1)
                    X_rand = posiciones[rand_idx]
                    D = np.abs(C * X_rand - posiciones[i])
                    X_new = X_rand - A * D
            else:
                # Spiral bubble-net
                D_prime = np.abs(X_best - posiciones[i])
                l = random.uniform(-1.0, 1.0)
                X_new = D_prime * np.exp(params.b * l) * np.cos(2.0 * math.pi * l) + X_best

            X_new = np.clip(X_new, lb, ub)
            nueva_val = func.func(X_new)

            if nueva_val < fitnesses[i]:
                posiciones[i] = X_new
                fitnesses[i]  = nueva_val

        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < mejor_val:
            mejor_val = fitnesses[best_idx]
            mejor_sol = posiciones[best_idx].copy()

        historial.append(mejor_val)
        historial_inst.append(fitnesses[best_idx])

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
                    print(f"    [WOA Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break
        dtw_info_hist.append(dtw_info)

    return WOAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol.tolist(),
        dtw_deltas       = dtw_deltas,
        dtw_info_hist    = dtw_info_hist,
    )
