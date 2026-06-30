"""
continuous_benchmark/mh/pso.py
------------------------------
Particle Swarm Optimization (PSO) para funciones continuas (minimizacion).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dtw_stagnation import StagnationConfig, StagnationMonitor


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class PSOParams:
    pop_size              : int   = 30
    iterations            : int   = 300
    epochs                : int   = 1
    inercia               : float = 0.7
    coeficiente_cognitivo : float = 1.5
    coeficiente_social    : float = 1.5
    injection_mode : str  = "random"
    use_stagnation : bool = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class PSOEpochResult:
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[float] = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mutar_solucion(sol: np.ndarray, lb: float, ub: float, n_dim: int) -> np.ndarray:
    copia = sol.copy()
    n_perturb = random.randint(1, max(1, n_dim // 10))
    indices = random.sample(range(n_dim), n_perturb)
    for idx in indices:
        copia[idx] = np.random.uniform(lb, ub)
    return copia


# ── Epoch ─────────────────────────────────────────────────────────────────────

def ejecutar_epoch(
    func,
    params    : PSOParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: np.ndarray | None = None,
) -> PSOEpochResult:

    n = func.n_dim
    lb, ub = func.lb, func.ub

    # Inicializar enjambre
    posiciones  = np.random.uniform(lb, ub, size=(params.pop_size, n))
    velocidades = np.zeros((params.pop_size, n))
    fitnesses   = np.array([func.func(p) for p in posiciones])

    pbest_pos = posiciones.copy()
    pbest_val = fitnesses.copy()

    gbest_idx = np.argmin(fitnesses)
    gbest_val = fitnesses[gbest_idx]
    gbest_pos = posiciones[gbest_idx].copy()

    # Inyeccion
    if sol_inyectada is not None:
        sol_rep = np.clip(sol_inyectada, lb, ub)
        val_rep = func.func(sol_rep)
        mode = params.injection_mode

        if mode == "random":
            peor_idx = np.argmax(fitnesses)
            posiciones[peor_idx] = sol_rep
            fitnesses[peor_idx] = val_rep
            pbest_pos[peor_idx] = sol_rep.copy()
            pbest_val[peor_idx] = val_rep
        elif mode == "mutated":
            posiciones[0] = sol_rep; fitnesses[0] = val_rep
            pbest_pos[0] = sol_rep.copy(); pbest_val[0] = val_rep
            for i in range(1, params.pop_size):
                msol = _mutar_solucion(sol_rep, lb, ub, n)
                mval = func.func(msol)
                posiciones[i] = msol; fitnesses[i] = mval
                pbest_pos[i] = msol.copy(); pbest_val[i] = mval
        elif mode == "mixed":
            posiciones[0] = sol_rep; fitnesses[0] = val_rep
            pbest_pos[0] = sol_rep.copy(); pbest_val[0] = val_rep
            n_mutados = params.pop_size // 2
            for i in range(1, n_mutados):
                msol = _mutar_solucion(sol_rep, lb, ub, n)
                mval = func.func(msol)
                posiciones[i] = msol; fitnesses[i] = mval
                pbest_pos[i] = msol.copy(); pbest_val[i] = mval

        gbest_idx = np.argmin(fitnesses)
        gbest_val = fitnesses[gbest_idx]
        gbest_pos = posiciones[gbest_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0
    v_max = (ub - lb) * 0.2

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        for i in range(params.pop_size):
            r1 = np.random.rand(n)
            r2 = np.random.rand(n)

            velocidades[i] = (params.inercia * velocidades[i]
                + params.coeficiente_cognitivo * r1 * (pbest_pos[i] - posiciones[i])
                + params.coeficiente_social    * r2 * (gbest_pos    - posiciones[i]))
            velocidades[i] = np.clip(velocidades[i], -v_max, v_max)

            posiciones[i] = posiciones[i] + velocidades[i]
            posiciones[i] = np.clip(posiciones[i], lb, ub)

            fitnesses[i] = func.func(posiciones[i])

            if fitnesses[i] < pbest_val[i]:
                pbest_val[i] = fitnesses[i]
                pbest_pos[i] = posiciones[i].copy()

            if fitnesses[i] < gbest_val:
                gbest_val = fitnesses[i]
                gbest_pos = posiciones[i].copy()

        fit_iter = float(np.min(fitnesses))
        historial.append(gbest_val)
        historial_inst.append(fit_iter)

        # Stagnation (pasamos -gbest_val para que el monitor detecte meseta en minimizacion)
        if monitor is not None:
            status = monitor.update(-gbest_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [PSO Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return PSOEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = gbest_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = gbest_pos.tolist(),
        dtw_deltas       = dtw_deltas,
    )
