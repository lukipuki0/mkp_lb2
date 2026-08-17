"""
continuous_benchmark/mh/ils.py
------------------------------
Iterated Local Search (ILS) para optimización continua y mixta (minimización).
Implementa operadores de vecindario, perturbación ("kick") y monitor DTW de estancamiento.
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
class ILSParams:
    iterations     : int   = 300
    epochs         : int   = 1
    step_size      : float = 0.05  # Amplitud inicial de paso continuo (en % del rango)
    local_evals    : int   = 10    # Evaluaciones en vecindario local por iteración
    kick_strength  : float = 0.20  # Amplitud de perturbación al estancarse localmente
    use_stagnation : bool  = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class ILSEpochResult:
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
    """Genera un vecino aplicando uno de los operadores de movimiento (continuo o discreto)."""
    n = len(x)
    vecino = x.copy()
    op = random.choice(["continuous", "battery", "electrolyzer", "balanced"])

    if op == "continuous":
        # Refinamiento continuo en dimensión 0 (eólica) o general
        dim = 0
        rango = ub[dim] - lb[dim]
        vecino[dim] += np.random.normal(0, step_scale * rango)

    elif op == "battery":
        # Modificar batería (dim 2 o dim 3)
        if random.random() < 0.5:
            # Paso de 5 MW en potencia de batería
            step = random.choice([-5.0, 5.0])
            vecino[2] += step
        else:
            # Duración de batería
            step = random.choice([-1.0, 1.0])
            vecino[3] += step

    elif op == "electrolyzer":
        # Paso de 1 unidad de electrolizador (dim 1)
        step = random.choice([-1.0, 1.0])
        vecino[1] += step

    elif op == "balanced":
        # Movimiento coordinado eólica + batería
        rango = ub[0] - lb[0]
        delta_w = np.random.normal(0, step_scale * rango)
        vecino[0] += delta_w
        vecino[2] += (5.0 if delta_w > 0 else -5.0)

    return np.clip(vecino, lb, ub)


def _perturbar_solucion(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, strength: float) -> np.ndarray:
    """Aplica perturbación fuerte ('kick') para saltar de óptimo local."""
    vecino = x.copy()
    rango = ub - lb
    ruido = np.random.uniform(-strength, strength, size=len(x)) * rango
    return np.clip(vecino + ruido, lb, ub)


def ejecutar_epoch(
    func,
    params    : ILSParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> ILSEpochResult:

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

    local_stagnant_count = 0

    for it in range(params.iterations):
        # Búsqueda local fina en el vecindario
        best_neighbor_sol = x_curr.copy()
        best_neighbor_val = val_curr

        step_scale = max(0.01, params.step_size * (1.0 - it / params.iterations))

        for _ in range(params.local_evals):
            vecino = _generar_vecino_4d(x_curr, lb, ub, step_scale)
            v_val = func.func(vecino)
            if v_val < best_neighbor_val:
                best_neighbor_val = v_val
                best_neighbor_sol = vecino.copy()

        # Aceptación estricta de la búsqueda local
        if best_neighbor_val < val_curr:
            x_curr = best_neighbor_sol.copy()
            val_curr = best_neighbor_val
            local_stagnant_count = 0
        else:
            local_stagnant_count += 1

        # Perturbación (Kick) si hay estancamiento local repetido
        if local_stagnant_count >= 3:
            x_curr = _perturbar_solucion(x_curr, lb, ub, params.kick_strength)
            val_curr = func.func(x_curr)
            local_stagnant_count = 0

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
                    print(f"    [ILS Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break
        dtw_info_hist.append(dtw_info)

    return ILSEpochResult(
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
