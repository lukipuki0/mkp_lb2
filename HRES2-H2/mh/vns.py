"""
HRES2-H2/mh/vns.py
-------------------
Variable Neighborhood Search (VNS) para HRES2-H2 (minimizacion continua mixta).
Sacudida (shaking) + busqueda local + k-vecindarios.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dtw_stagnation import StagnationConfig, StagnationMonitor


@dataclass
class VNSParams:
    iterations     : int   = 300
    epochs         : int   = 1
    k_max          : int   = 3
    local_evals    : int   = 10
    step_size      : float = 0.05
    use_stagnation : bool  = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class VNSEpochResult:
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[float] = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)
    dtw_info_hist    : list[dict]  = field(default_factory=list)


def _shake(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, k: int, step_scale: float) -> np.ndarray:
    rango = ub - lb
    ruido = np.random.uniform(-k * step_scale, k * step_scale, size=len(x)) * rango
    return np.clip(x + ruido, lb, ub)


def _busqueda_local(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, func, n_evals: int, step_scale: float):
    rango = ub - lb
    best_x = x.copy()
    best_val = func.func(best_x)
    for _ in range(n_evals):
        cand = np.clip(best_x + np.random.normal(0, step_scale * rango), lb, ub)
        val = func.func(cand)
        if val < best_val:
            best_val = val
            best_x = cand
    return best_x, best_val


def ejecutar_epoch(
    func,
    params    : VNSParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> VNSEpochResult:

    n = func.n_dim
    lb = getattr(func, "lb_vector", np.full(n, func.lb))
    ub = getattr(func, "ub_vector", np.full(n, func.ub))

    x_curr = np.clip(sol_inyectada, lb, ub) if sol_inyectada is not None else np.random.uniform(lb, ub)
    val_curr = func.func(x_curr)
    mejor_sol = x_curr.copy()
    mejor_val = val_curr

    historial, historial_inst, dtw_deltas, dtw_info_hist = [], [], [], []
    stag_fires = 0
    k = 1

    monitor = None
    if params.use_stagnation:
        cfg = params.stag_cfg if params.stag_cfg is not None else StagnationConfig()
        monitor = StagnationMonitor(cfg)

    for it in range(1, params.iterations + 1):
        x_shaken = _shake(x_curr, lb, ub, k, params.step_size)
        x_ls, val_ls = _busqueda_local(x_shaken, lb, ub, func, params.local_evals, params.step_size)

        if val_ls < val_curr:
            x_curr, val_curr, k = x_ls, val_ls, 1
        else:
            k = k + 1 if k < params.k_max else 1

        if val_curr < mejor_val:
            mejor_val = val_curr
            mejor_sol = x_curr.copy()

        historial.append(float(mejor_val))
        historial_inst.append(float(val_curr))

        if monitor is not None:
            stagnated, info = monitor.update(mejor_val)
            dtw_deltas.append(info.get('delta', float('nan')))
            dtw_info_hist.append(info)
            if stagnated:
                stag_fires += 1
                if verbose:
                    print(f"    [VNS Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return VNSEpochResult(
        epoch_idx=epoch_idx, mejor_valor=mejor_val, iteraciones=len(historial),
        stagnation_fires=stag_fires, historial=historial, historial_inst=historial_inst,
        mejor_solucion=mejor_sol.tolist(), dtw_deltas=dtw_deltas, dtw_info_hist=dtw_info_hist,
    )
