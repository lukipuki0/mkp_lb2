"""
mh/woa.py
---------
Whale Optimization Algorithm (WOA) para el MKP con binarización LB2.

Versión limpia para el pipeline híbrido: solo usa estrategia "abort"
cuando el monitor DTW detecta estancamiento.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from mkp_core.problem   import MKPInstance
from mkp_core.repair    import reparar_solucion
from dtw_stagnation     import StagnationConfig, StagnationMonitor
from lb2 import binarizar_posicion, interpolar_G


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class WOAParams:
    """Hiperparámetros del WOA."""
    pop_size       : int   = 30
    iterations     : int   = 300
    epochs         : int   = 10
    v_max          : float = 6.0
    b              : float = 1.0   # Constante de espiral logarítmica
    # LB2 params
    G1_i : float = 0.5;  G1_f : float = 1.0
    G2_i : float = 0.5;  G2_f : float = 7.2
    G3_i : float = 0.5;  G3_f : float = 0.0
    # Inyección de solución (pipeline híbrido)
    injection_mode : str  = "random"    # "random" | "mutated" | "mixed"
    # Stagnation
    use_stagnation : bool = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class WOAEpochResult:
    """Resultado de un epoch del WOA."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)  # fitness del líder (mejor ballena de la iteración)
    mejor_solucion   : list[int]  = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)


@dataclass
class WOAResult:
    """Resultado completo del WOA (todos los epochs)."""
    epochs             : list[WOAEpochResult]
    mejor_valor_global : float
    mejor_sol_global   : list[int]
    valor_optimo       : float

    @property
    def gap_pct(self) -> float | None:
        if self.valor_optimo == 0:
            return None
        return 100.0 * (self.valor_optimo - self.mejor_valor_global) / self.valor_optimo

    @property
    def valores_por_epoch(self) -> list[float]:
        return [ep.mejor_valor for ep in self.epochs]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inicializar_ballenas(
    inst: MKPInstance,
    pop_size: int,
    v_max: float,
) -> tuple[np.ndarray, list[list[int]], list[float]]:
    """Genera la población inicial: posiciones continuas + soluciones binarias factibles."""
    n = inst.n
    posiciones = np.random.uniform(-v_max, v_max, size=(pop_size, n))
    poblacion_bin = []
    fitnesses     = []

    for i in range(pop_size):
        sol = [random.randint(0, 1) for _ in range(n)]
        sol, val = reparar_solucion(sol, inst)
        poblacion_bin.append(sol)
        fitnesses.append(val)

    return posiciones, poblacion_bin, fitnesses


def _mutar_solucion(sol: list[int], inst: MKPInstance, n_flips: int = 0) -> tuple[list[int], float]:
    """Crea una copia de `sol` con bits invertidos al azar y la repara."""
    copia = list(sol)
    n = len(copia)
    if n_flips <= 0:
        n_flips = random.randint(1, 3)
    indices = random.sample(range(n), min(n_flips, n))
    for idx in indices:
        copia[idx] = 1 - copia[idx]
    copia, val = reparar_solucion(copia, inst)
    return copia, val


# ── Epoch individual ─────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst      : MKPInstance,
    params    : WOAParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: list[int] | None = None,
) -> WOAEpochResult:
    """Ejecuta un epoch completo de WOA con detección de estancamiento (abort)."""

    pop_size = params.pop_size
    n = inst.n

    # Inicializar población de ballenas
    posiciones, poblacion_bin, fitnesses = _inicializar_ballenas(
        inst, pop_size, params.v_max,
    )

    # Inyectar solución del orquestador según el modo de inyección
    if sol_inyectada is not None:
        sol_rep = list(sol_inyectada)
        sol_rep, val_rep = reparar_solucion(sol_rep, inst)
        mode = params.injection_mode

        if mode == "random":
            peor_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            poblacion_bin[peor_idx] = sol_rep
            fitnesses[peor_idx] = val_rep

        elif mode == "mutated":
            poblacion_bin[0] = sol_rep
            fitnesses[0] = val_rep
            for i in range(1, pop_size):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion_bin[i] = msol
                fitnesses[i] = mval

        elif mode == "mixed":
            poblacion_bin[0] = sol_rep
            fitnesses[0] = val_rep
            n_mutados = pop_size // 2
            for i in range(1, n_mutados):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion_bin[i] = msol
                fitnesses[i] = mval

    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    mejor_val = fitnesses[best_idx]
    mejor_sol = poblacion_bin[best_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0

    # Estado dinámico de los parámetros G
    G1 = params.G1_i
    G2 = params.G2_i
    G3 = params.G3_i

    # Inicializar monitor
    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        # Parámetro 'a' disminuye linealmente de 2 a 0
        a = 2.0 - it * (2.0 / params.iterations)

        X_best = posiciones[best_idx].copy()

        # Actualizar cada ballena
        for i in range(pop_size):
            if i == best_idx:
                continue

            X_i = posiciones[i]

            p = random.random()
            r = np.random.random(n)
            A = 2.0 * a * r - a
            C = 2.0 * np.random.random(n)

            if p < 0.5:
                # Escalar para determinar explorar o explotar
                A_scalar = 2.0 * a * random.random() - a
                if abs(A_scalar) < 1.0:
                    # Encircling prey (explotación)
                    D = np.abs(C * X_best - X_i)
                    X_new = X_best - A * D
                else:
                    # Search for prey (exploración - usando una ballena aleatoria)
                    rand_idx = random.choice([idx for idx in range(pop_size) if idx != i])
                    X_rand = posiciones[rand_idx]
                    D = np.abs(C * X_rand - X_i)
                    X_new = X_rand - A * D
            else:
                # Spiral bubble-net attack (explotación en espiral)
                D_prime = np.abs(X_best - X_i)
                l = random.uniform(-1.0, 1.0)
                X_new = D_prime * np.exp(params.b * l) * np.cos(2.0 * np.pi * l) + X_best

            # Limitar posiciones al espacio continuo permitido
            X_new = np.clip(X_new, -params.v_max, params.v_max)
            posiciones[i] = X_new

            # Binarizar posición usando LB2
            nueva_sol, nueva_val = binarizar_posicion(
                X_new, poblacion_bin[i], inst,
                G1, G2, G3, params.v_max,
            )

            # Selección Greedy
            if nueva_val >= fitnesses[i]:
                poblacion_bin[i] = nueva_sol
                fitnesses[i]     = nueva_val

        # Recalcular mejor ballena
        best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        fit_best_actual = fitnesses[best_idx]

        if fit_best_actual > mejor_val:
            mejor_val = fit_best_actual
            mejor_sol = poblacion_bin[best_idx].copy()

        historial.append(mejor_val)
        historial_inst.append(fit_best_actual)

        # ── Stagnation check ──────────────────────────────────────────────
        if monitor is not None:
            status = monitor.update(mejor_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))

            if verbose and status.get("ready"):
                dlt = status.get("delta", 0.0)
                td  = status.get("theta_delta", 0.0)
                if dlt > td: estado = "Explorar mucho"
                elif 0 <= dlt <= td: estado = "Explorar poco"
                elif -td <= dlt < 0: estado = "Explotar poco"
                else: estado = "Explotar mucho"
                print(f"i={it:03d} | Estado: {estado:<15} | Delta={dlt:6.1f} | Th_d={td:6.1f} | d1={status.get('D1_vs_ramp', 0.0):.3f} | d2={status.get('D2_vs_const', 0.0):.3f} | best={mejor_val:.1f}")

            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break
        else:
            G1 = interpolar_G(it, params.iterations, params.G1_i, params.G1_f)
            G2 = interpolar_G(it, params.iterations, params.G2_i, params.G2_f)
            G3 = interpolar_G(it, params.iterations, params.G3_i, params.G3_f)

    return WOAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = it + 1,
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
    )


# ── Ejecución multi-epoch ────────────────────────────────────────────────────

def ejecutar_woa(
    inst: MKPInstance,
    params: WOAParams,
    verbose: bool = True,
) -> WOAResult:
    """Ejecuta el WOA completo (todos los epochs) y retorna el WOAResult."""
    epochs_result    = []
    mejor_val_global = -float("inf")
    mejor_sol_global: list[int] = []

    for e in range(params.epochs):
        epoch_res = ejecutar_epoch(inst, params, epoch_idx=e, verbose=verbose)
        epochs_result.append(epoch_res)

        if epoch_res.mejor_valor > mejor_val_global:
            mejor_val_global = epoch_res.mejor_valor
            mejor_sol_global = epoch_res.mejor_solucion.copy()

    return WOAResult(
        epochs             = epochs_result,
        mejor_valor_global = mejor_val_global,
        mejor_sol_global   = mejor_sol_global,
        valor_optimo       = inst.valor_optimo,
    )
