"""
continuous_benchmark/mh/gwo.py
------------------------------
Grey Wolf Optimizer (GWO) para funciones continuas (minimizacion).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dtw_stagnation import StagnationConfig, StagnationMonitor


@dataclass
class GWOParams:
    pop_size       : int  = 30
    iterations     : int  = 300
    epochs         : int  = 1
    injection_mode : str  = "random"
    use_stagnation : bool = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class GWOEpochResult:
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
    params    : GWOParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> GWOEpochResult:

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

    # Jerarquia (minimizacion: los menores son los mejores)
    sorted_idx = np.argsort(fitnesses)
    alpha_idx, beta_idx, delta_idx = sorted_idx[0], sorted_idx[1], sorted_idx[2]

    mejor_val = fitnesses[alpha_idx]
    mejor_sol = posiciones[alpha_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        a = 2.0 - it * (2.0 / params.iterations)

        X_alpha = posiciones[alpha_idx].copy()
        X_beta  = posiciones[beta_idx].copy()
        X_delta = posiciones[delta_idx].copy()

        for i in range(params.pop_size):
            if i in (alpha_idx, beta_idx, delta_idx):
                continue

            r1_a, r2_a = np.random.random(n), np.random.random(n)
            r1_b, r2_b = np.random.random(n), np.random.random(n)
            r1_d, r2_d = np.random.random(n), np.random.random(n)

            A1 = 2.0 * a * r1_a - a;  C1 = 2.0 * r2_a
            A2 = 2.0 * a * r1_b - a;  C2 = 2.0 * r2_b
            A3 = 2.0 * a * r1_d - a;  C3 = 2.0 * r2_d

            X1 = X_alpha - A1 * np.abs(C1 * X_alpha - posiciones[i])
            X2 = X_beta  - A2 * np.abs(C2 * X_beta  - posiciones[i])
            X3 = X_delta - A3 * np.abs(C3 * X_delta - posiciones[i])

            X_new = np.clip((X1 + X2 + X3) / 3.0, lb, ub)
            nueva_val = func.func(X_new)

            if nueva_val < fitnesses[i]:
                posiciones[i] = X_new
                fitnesses[i]  = nueva_val

        sorted_idx = np.argsort(fitnesses)
        alpha_idx, beta_idx, delta_idx = sorted_idx[0], sorted_idx[1], sorted_idx[2]

        if fitnesses[alpha_idx] < mejor_val:
            mejor_val = fitnesses[alpha_idx]
            mejor_sol = posiciones[alpha_idx].copy()

        historial.append(mejor_val)
        historial_inst.append(fitnesses[alpha_idx])

        if monitor is not None:
            status = monitor.update(-mejor_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [GWO Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return GWOEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol.tolist(),
        dtw_deltas       = dtw_deltas,
    )
