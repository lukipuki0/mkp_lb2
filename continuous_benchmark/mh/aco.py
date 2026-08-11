"""
continuous_benchmark/mh/aco.py
------------------------------
Ant Colony Optimization for Continuous Domains (ACOR) para funciones continuas (minimización).
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
class ACOParams:
    pop_size       : int   = 30     # Tamaño del archivo de soluciones k
    n_ants         : int   = 15     # Número de hormigas muestreadas por iteración
    iterations     : int   = 300
    epochs         : int   = 1
    q              : float = 0.5    # Parámetro de preferencia de las mejores soluciones
    xi             : float = 0.85   # Tasa de desviación estándar (velocidad de convergencia)
    injection_mode : str  = "random"
    use_stagnation : bool = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class ACOEpochResult:
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
    params        : ACOParams,
    epoch_idx     : int = 0,
    verbose       : bool = True,
    sol_inyectada : np.ndarray | None = None,
) -> ACOEpochResult:

    n = func.n_dim
    lb, ub = func.lb, func.ub
    k_archive = params.pop_size
    n_ants = params.n_ants

    # Inicializar archivo de soluciones continuas (posiciones y fitnesses)
    archivo_pos = np.random.uniform(lb, ub, size=(k_archive, n))
    archivo_fit = np.array([func.func(p) for p in archivo_pos])

    # Inyección de solución del orquestador
    if sol_inyectada is not None:
        sol_rep = np.clip(sol_inyectada, lb, ub)
        val_rep = func.func(sol_rep)
        mode = params.injection_mode

        if mode == "random":
            peor_idx = np.argmax(archivo_fit)
            archivo_pos[peor_idx] = sol_rep
            archivo_fit[peor_idx] = val_rep

        elif mode == "mutated":
            archivo_pos[0] = sol_rep
            archivo_fit[0] = val_rep
            for i in range(1, k_archive):
                msol = _mutar_solucion(sol_rep, lb, ub, n)
                archivo_pos[i] = msol
                archivo_fit[i] = func.func(msol)

        elif mode == "mixed":
            archivo_pos[0] = sol_rep
            archivo_fit[0] = val_rep
            n_mutados = k_archive // 2
            for i in range(1, n_mutados):
                msol = _mutar_solucion(sol_rep, lb, ub, n)
                archivo_pos[i] = msol
                archivo_fit[i] = func.func(msol)

    # Ordenar archivo por calidad (minimización: menor fitness primero)
    order = np.argsort(archivo_fit)
    archivo_pos = archivo_pos[order]
    archivo_fit = archivo_fit[order]

    gbest_val = float(archivo_fit[0])
    gbest_pos = archivo_pos[0].copy()

    # Precalcular pesos del archivo basados en Gaussiana discreta
    weights = (1.0 / (params.q * k_archive * np.sqrt(2.0 * np.pi))) * np.exp(
        -((np.arange(k_archive)) ** 2) / (2.0 * (params.q ** 2) * (k_archive ** 2))
    )
    probs = weights / np.sum(weights)

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        # 1. Calcular desviaciones estándar por dimensión para cada guía en el archivo
        # sigma_l^j = xi * sum_e |x_e^j - x_l^j| / (k - 1)
        sigmas = np.zeros((k_archive, n))
        for l in range(k_archive):
            dist_sum = np.sum(np.abs(archivo_pos - archivo_pos[l]), axis=0)
            sigmas[l] = params.xi * dist_sum / max(1, k_archive - 1)
            sigmas[l] = np.maximum(sigmas[l], 1e-6)

        # 2. Generar nuevas hormigas muestreando las Gaussianas de las soluciones guía
        nuevas_pos = np.zeros((n_ants, n))
        nuevos_fit = np.zeros(n_ants)

        for i in range(n_ants):
            # Seleccionar guía l según distribución de probabilidad prob
            l_guide = np.random.choice(k_archive, p=probs)
            # Muestrear cada dimensión alrededor de la posición del guía
            sample = np.random.normal(archivo_pos[l_guide], sigmas[l_guide])
            sample = np.clip(sample, lb, ub)
            nuevas_pos[i] = sample
            nuevos_fit[i] = func.func(sample)

        # 3. Combinar archivo actual + nuevas hormigas y conservar las k_archive mejores
        comb_pos = np.vstack((archivo_pos, nuevas_pos))
        comb_fit = np.concatenate((archivo_fit, nuevos_fit))

        new_order = np.argsort(comb_fit)[:k_archive]
        archivo_pos = comb_pos[new_order]
        archivo_fit = comb_fit[new_order]

        fit_iter_best = float(np.min(nuevos_fit))
        if archivo_fit[0] < gbest_val:
            gbest_val = float(archivo_fit[0])
            gbest_pos = archivo_pos[0].copy()

        historial.append(gbest_val)
        historial_inst.append(fit_iter_best)

        # Stagnation monitor (pasamos -gbest_val para minimización)
        if monitor is not None:
            status = monitor.update(-gbest_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [ACO Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return ACOEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = gbest_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = gbest_pos.tolist(),
        dtw_deltas       = dtw_deltas,
    )
