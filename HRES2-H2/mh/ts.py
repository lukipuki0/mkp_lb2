"""
HRES2-H2/mh/ts.py
------------------
Tabu Search (TS) para HRES2-H2 (minimizacion continua mixta).
Lista tabu + criterio de aspiracion + operadores HRES2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dtw_stagnation import StagnationConfig, StagnationMonitor


@dataclass
class TSParams:
    iterations     : int   = 300
    epochs         : int   = 1
    tabu_tenure    : int   = 10
    neighborhood_sz: int   = 15
    step_size      : float = 0.05
    use_stagnation : bool  = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class TSEpochResult:
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
        rango = ub[0] - lb[0]
        vecino[0] += np.random.normal(0, step_scale * rango)
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


def _es_tabu(sol: np.ndarray, tabu_list: list, tol: float = 1e-3) -> bool:
    for t_sol in tabu_list:
        if np.linalg.norm(sol - t_sol) < tol:
            return True
    return False


def ejecutar_epoch(
    func,
    params    : TSParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> TSEpochResult:

    n = func.n_dim
    lb = getattr(func, "lb_vector", np.full(n, func.lb))
    ub = getattr(func, "ub_vector", np.full(n, func.ub))

    x_curr = np.clip(sol_inyectada, lb, ub) if sol_inyectada is not None else np.random.uniform(lb, ub)
    val_curr = func.func(x_curr)
    mejor_sol = x_curr.copy()
    mejor_val = val_curr

    historial, historial_inst, dtw_deltas, dtw_info_hist = [], [], [], []
    tabu_list = []
    stag_fires = 0

    monitor = None
    if params.use_stagnation:
        cfg = params.stag_cfg if params.stag_cfg is not None else StagnationConfig()
        monitor = StagnationMonitor(cfg)

    for it in range(1, params.iterations + 1):
        candidatos = []
        for _ in range(params.neighborhood_sz):
            v = _generar_vecino(x_curr, lb, ub, params.step_size)
            candidatos.append((func.func(v), v))
        candidatos.sort(key=lambda item: item[0])

        elegido_val, elegido_sol = None, None
        for val_cand, sol_cand in candidatos:
            if val_cand < mejor_val or not _es_tabu(sol_cand, tabu_list):
                elegido_val, elegido_sol = val_cand, sol_cand
                break

        if elegido_sol is None:
            elegido_val, elegido_sol = candidatos[0]

        x_curr = elegido_sol
        val_curr = elegido_val
        tabu_list.append(x_curr.copy())
        if len(tabu_list) > params.tabu_tenure:
            tabu_list.pop(0)

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
                    print(f"    [TS Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return TSEpochResult(
        epoch_idx=epoch_idx, mejor_valor=mejor_val, iteraciones=len(historial),
        stagnation_fires=stag_fires, historial=historial, historial_inst=historial_inst,
        mejor_solucion=mejor_sol.tolist(), dtw_deltas=dtw_deltas, dtw_info_hist=dtw_info_hist,
    )
