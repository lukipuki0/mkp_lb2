"""
continuous_benchmark/mh/sa.py
------------------------------
Simulated Annealing (SA) para optimización continua y mixta (minimización).
Implementa enfriamiento geométrico, probabilidad Metropolis y monitor DTW.
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
    t_initial      : float = 10.0   # Temperatura inicial
    cooling_rate   : float = 0.96   # Factor de enfriamiento geométrico alpha
    step_size      : float = 0.05   # Amplitud inicial de paso
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


def _generar_vecino_4d(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, step_scale: float) -> np.ndarray:
    n = len(x)
    vecino = x.copy()
    op = random.choice(["continuous", "battery", "electrolyzer", "balanced"])

    if op == "continuous":
        dim = 0
        rango = ub[dim] - lb[dim]
        vecino[dim] += np.random.normal(0, step_scale * rango)

    elif op == "battery":
        if random.random() < 0.5:
            step = random.choice([-5.0, 5.0])
            vecino[2] += step
        else:
            step = random.choice([-1.0, 1.0])
            vecino[3] += step

    elif op == "electrolyzer":
        step = random.choice([-1.0, 1.0])
        vecino[1] += step

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

    if sol_inyectada is not None:
        x_curr = np.clip(sol_inyectada, lb, ub)
    else:
        x_curr = np.random.uniform(lb, ub)

    val_curr = func.func(x_curr)
    mejor_sol = x_curr.copy()
    mejor_val = val_curr

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    dtw_info_hist  = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    temp = params.t_initial

    for it in range(params.iterations):
        step_scale = max(0.005, params.step_size * (temp / params.t_initial))
        vecino = _generar_vecino_4d(x_curr, lb, ub, step_scale)
        val_vecino = func.func(vecino)

        delta = val_vecino - val_curr

        # Criterio Metropolis
        if delta < 0 or random.random() < math.exp(-delta / max(1e-8, temp)):
            x_curr = vecino.copy()
            val_curr = val_vecino

        # Enfriar
        temp *= params.cooling_rate

        # Actualizar mejor histórico
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
