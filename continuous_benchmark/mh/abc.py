"""
continuous_benchmark/mh/abc.py
------------------------------
Artificial Bee Colony (ABC) Algorithm para optimización continua (minimización).

Lógica de 4 Fases:
  1. Inicialización: Fuentes de alimento distribuidas aleatoriamente en [lb, ub].
  2. Fase de Abejas Empleadas: v_{i,j} = x_{i,j} + phi * (x_{i,j} - x_{k,j}), phi in [-1, 1],
     recorte a [lb, ub] y selección codiciosa.
  3. Fase de Abejas Observadoras: P_i = 0.9 * (fit_i / max(fit)) + 0.1, asignación por ruleta
     y búsqueda en vecindario con selección codiciosa.
  4. Fase de Abejas Exploradoras (Scouts): Si trials[i] >= limit, reinicialización aleatoria en [lb, ub].

Soporta inyección de solución y monitor DTW de estancamiento.
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
class ABCParams:
    """Hiperparámetros del ABC continuo."""
    pop_size       : int   = 30     # Número de fuentes de alimento
    iterations     : int   = 300
    epochs         : int   = 1
    limit          : int | None = None # Límite de intentos sin mejora (si None: pop_size * dim / 2)
    injection_mode : str  = "random"
    use_stagnation : bool = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class ABCEpochResult:
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[float] = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fitness_transform(f_val: float) -> float:
    """Transforma el valor objetivo de minimización a fitness estrictamente positivo (>0)."""
    if f_val >= 0:
        return 1.0 / (1.0 + f_val)
    else:
        return 1.0 + abs(f_val)


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
    params        : ABCParams,
    epoch_idx     : int = 0,
    verbose       : bool = True,
    sol_inyectada : np.ndarray | None = None,
) -> ABCEpochResult:

    n_dim = func.n_dim
    lb, ub = func.lb, func.ub
    pop_size = params.pop_size
    limit = params.limit if params.limit is not None else int(pop_size * n_dim / 2)

    # 1. Fase de Inicialización
    posiciones = np.random.uniform(lb, ub, size=(pop_size, n_dim))
    obj_vals   = np.array([func.func(p) for p in posiciones])
    trials     = np.zeros(pop_size, dtype=int)

    # Inyección de solución del orquestador
    if sol_inyectada is not None:
        sol_rep = np.clip(sol_inyectada, lb, ub)
        val_rep = func.func(sol_rep)
        mode = params.injection_mode

        if mode == "random":
            peor_idx = np.argmax(obj_vals)
            posiciones[peor_idx] = sol_rep
            obj_vals[peor_idx] = val_rep
            trials[peor_idx] = 0

        elif mode == "mutated":
            posiciones[0] = sol_rep
            obj_vals[0] = val_rep
            trials[0] = 0
            for i in range(1, pop_size):
                msol = _mutar_solucion(sol_rep, lb, ub, n_dim)
                posiciones[i] = msol
                obj_vals[i] = func.func(msol)
                trials[i] = 0

        elif mode == "mixed":
            posiciones[0] = sol_rep
            obj_vals[0] = val_rep
            trials[0] = 0
            n_mutados = pop_size // 2
            for i in range(1, n_mutados):
                msol = _mutar_solucion(sol_rep, lb, ub, n_dim)
                posiciones[i] = msol
                obj_vals[i] = func.func(msol)
                trials[i] = 0

    gbest_idx = np.argmin(obj_vals)
    gbest_val = float(obj_vals[gbest_idx])
    gbest_pos = posiciones[gbest_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):

        # 2. Fase de Abejas Empleadas
        for i in range(pop_size):
            k_candidates = [idx for idx in range(pop_size) if idx != i]
            k = random.choice(k_candidates)

            j = random.randint(0, n_dim - 1)
            phi = random.uniform(-1.0, 1.0)

            v_i = posiciones[i].copy()
            v_i[j] = v_i[j] + phi * (v_i[j] - posiciones[k][j])
            v_i = np.clip(v_i, lb, ub)

            f_v = func.func(v_i)

            # Selección codiciosa (minimización: f_v <= obj_vals[i])
            if f_v <= obj_vals[i]:
                posiciones[i] = v_i
                obj_vals[i] = f_v
                trials[i] = 0
            else:
                trials[i] += 1

        # 3. Fase de Abejas Observadoras
        fits = np.array([_fitness_transform(val) for val in obj_vals])
        max_fit = np.max(fits)
        if max_fit > 0:
            probs = 0.9 * (fits / max_fit) + 0.1
        else:
            probs = np.full(pop_size, 1.0 / pop_size)
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum

        for _ in range(pop_size):
            i = np.random.choice(pop_size, p=probs)

            k_candidates = [idx for idx in range(pop_size) if idx != i]
            k = random.choice(k_candidates)

            j = random.randint(0, n_dim - 1)
            phi = random.uniform(-1.0, 1.0)

            v_i = posiciones[i].copy()
            v_i[j] = v_i[j] + phi * (v_i[j] - posiciones[k][j])
            v_i = np.clip(v_i, lb, ub)

            f_v = func.func(v_i)

            if f_v <= obj_vals[i]:
                posiciones[i] = v_i
                obj_vals[i] = f_v
                trials[i] = 0
            else:
                trials[i] += 1

        # Actualizar mejor global
        iter_best_idx = np.argmin(obj_vals)
        fit_iter_best = float(obj_vals[iter_best_idx])

        if fit_iter_best < gbest_val:
            gbest_val = fit_iter_best
            gbest_pos = posiciones[iter_best_idx].copy()

        # 4. Fase de Abejas Exploradoras (Scouts)
        for i in range(pop_size):
            if trials[i] >= limit:
                posiciones[i] = np.random.uniform(lb, ub, size=n_dim)
                obj_vals[i] = func.func(posiciones[i])
                trials[i] = 0

                if obj_vals[i] < gbest_val:
                    gbest_val = float(obj_vals[i])
                    gbest_pos = posiciones[i].copy()

        historial.append(gbest_val)
        historial_inst.append(fit_iter_best)

        # Stagnation monitor (-gbest_val para minimización)
        if monitor is not None:
            status = monitor.update(-gbest_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [ABC Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return ABCEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = gbest_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = gbest_pos.tolist(),
        dtw_deltas       = dtw_deltas,
    )
