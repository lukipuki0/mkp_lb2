"""
HRES2-H2/mh/sa.py
------------------
Simulated Annealing (SA) para HRES2-H2 (minimizacion continua mixta).
Enfriamiento geometrico + criterio Metropolis + operadores HRES2.
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
class SAParams:
    iterations     : int   = 300
    epochs         : int   = 1
    t_initial      : float = 10.0
    cooling_rate   : float = 0.96
    step_size      : float = 0.05
    use_stagnation : bool  = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class SAEpochResult:
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[float] = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)
    dtw_info_hist    : list[dict]  = field(default_factory=list)


def _generar_vecino(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, step_scale: float) -> np.ndarray:
    """Genera un vecino con operadores mixtos HRES2."""
    vecino = x.copy()
    op = random.choice(["continuous", "battery", "electrolyzer", "balanced"])

    if op == "continuous":
        dim = 0
        rango = ub[dim] - lb[dim]
        vecino[dim] += np.random.normal(0, step_scale * rango)
    elif op == "battery":
        if random.random() < 0.5:
            vecino[2] += random.choice([-5.0, 5.0])
        else:
            vecino[3] += random.choice([-1.0, 1.0])
    elif op == "electrolyzer":
        vecino[1] += random.choice([-1.0, 1.0])
    elif op == "balanced":
        rango = ub[0] - lb[0]
        delta_w = np.random.normal(0, step_scale * rango)
        vecino[0] += delta_w
        vecino[2] += (5.0 if delta_w > 0 else -5.0)

    return np.clip(vecino, lb, ub)


def ejecutar_epoch(
    func,
    params    : SAParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> SAEpochResult:

    n = func.n_dim
    lb = getattr(func, "lb_vector", np.full(n, func.lb))
    ub = getattr(func, "ub_vector", np.full(n, func.ub))

    x_curr = np.clip(sol_inyectada, lb, ub) if sol_inyectada is not None else np.random.uniform(lb, ub)
    val_curr = func.func(x_curr)
    mejor_sol = x_curr.copy()
    mejor_val = val_curr

    historial, historial_inst, dtw_deltas, dtw_info_hist = [], [], [], []
    stag_fires = 0
    temp = params.t_initial

    monitor = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        step_scale = max(0.005, params.step_size * (temp / params.t_initial))
        vecino = _generar_vecino(x_curr, lb, ub, step_scale)
        val_vecino = func.func(vecino)

        delta = val_vecino - val_curr
        if delta < 0 or random.random() < math.exp(-delta / max(1e-8, temp)):
            x_curr = vecino.copy()
            val_curr = val_vecino

        temp *= params.cooling_rate

        if val_curr < mejor_val:
            mejor_val = val_curr
            mejor_sol = x_curr.copy()

        historial.append(mejor_val)
        historial_inst.append(val_curr)

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
                    print(f"    [SA Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break
        dtw_info_hist.append(dtw_info)

    return SAEpochResult(
        epoch_idx=epoch_idx, mejor_valor=mejor_val, iteraciones=len(historial),
        stagnation_fires=stag_fires, historial=historial, historial_inst=historial_inst,
        mejor_solucion=mejor_sol.tolist(), dtw_deltas=dtw_deltas, dtw_info_hist=dtw_info_hist,
    )
