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

from hybrid_mkp.mkp_core.problem   import MKPInstance
from hybrid_mkp.mkp_core.repair    import reparar_solucion
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
    dtw_info_hist    : list[dict]  = field(default_factory=list)


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

def _inicializar_poblacion(inst: MKPInstance, pop_size: int) -> tuple[np.ndarray, list[list[int]], list[float]]:
    n = inst.n
    posiciones = np.random.uniform(-1.0, 1.0, size=(pop_size, n))
    poblacion_bin = []
    fitnesses = []
    for i in range(pop_size):
        sol_raw = [1 if pos > 0 else 0 for pos in posiciones[i]]
        sol, val = reparar_solucion(sol_raw, inst)
        poblacion_bin.append(sol)
        fitnesses.append(val)
    return posiciones, poblacion_bin, fitnesses


def _crear_ballena(sol: list[int], val: float) -> tuple[np.ndarray, list[int], float]:
    pos = np.where(np.array(sol) == 1, 1.0, -1.0)
    return pos, sol.copy(), val


def _mutar_solucion(sol: list[int], inst: MKPInstance) -> tuple[list[int], float]:
    n = inst.n
    copia = sol.copy()
    n_flips = random.randint(1, max(1, n // 10))
    for idx in random.sample(range(n), n_flips):
        copia[idx] = 1 - copia[idx]
    return reparar_solucion(copia, inst)


# ── Epoch ─────────────────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst          : MKPInstance,
    params        : WOAParams,
    epoch_idx     : int = 0,
    verbose       : bool = True,
    sol_inyectada : list[int] | None = None,
) -> WOAEpochResult:
    """Ejecuta un único epoch del WOA."""
    posiciones, poblacion_bin, fitnesses = _inicializar_poblacion(inst, params.pop_size)

    # Inyección
    if sol_inyectada is not None:
        sol_rep, val_rep = reparar_solucion(sol_inyectada, inst)
        mode = params.injection_mode

        if mode == "random":
            peor_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            pos, sol, val = _crear_ballena(sol_rep, val_rep)
            posiciones[peor_idx]    = pos
            poblacion_bin[peor_idx] = sol
            fitnesses[peor_idx]     = val

        elif mode == "mutated":
            pos, sol, val = _crear_ballena(sol_rep, val_rep)
            posiciones[0] = pos; poblacion_bin[0] = sol; fitnesses[0] = val
            for i in range(1, params.pop_size):
                msol, mval = _mutar_solucion(sol_rep, inst)
                mpos, msol, mval = _crear_ballena(msol, mval)
                posiciones[i] = mpos; poblacion_bin[i] = msol; fitnesses[i] = mval

        elif mode == "mixed":
            pos, sol, val = _crear_ballena(sol_rep, val_rep)
            posiciones[0] = pos; poblacion_bin[0] = sol; fitnesses[0] = val
            n_mutados = params.pop_size // 2
            for i in range(1, n_mutados):
                msol, mval = _mutar_solucion(sol_rep, inst)
                mpos, msol, mval = _crear_ballena(msol, mval)
                posiciones[i] = mpos; poblacion_bin[i] = msol; fitnesses[i] = mval

    best_idx  = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    mejor_val = fitnesses[best_idx]
    mejor_sol = poblacion_bin[best_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    dtw_info_hist  = []
    stag_fires     = 0

    G1 = params.G1_i
    G2 = params.G2_i
    G3 = params.G3_i

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    pop_size = params.pop_size

    for it in range(params.iterations):
        a  = 2.0 - 2.0 * (it / max(1, params.iterations - 1))
        a2 = -1.0 + it * (-1.0 / max(1, params.iterations - 1))

        X_best = posiciones[best_idx]

        for i in range(pop_size):
            r1 = random.random()
            r2 = random.random()

            A = 2.0 * a * r1 - a
            C = 2.0 * r2

            p = random.random()
            l = (a2 - 1.0) * random.random() + 1.0

            if p < 0.5:
                if abs(A) < 1.0:
                    D_leader = np.abs(C * X_best - posiciones[i])
                    X_new = X_best - A * D_leader
                else:
                    rand_idx = random.randint(0, pop_size - 1)
                    X_rand = posiciones[rand_idx]
                    D_rand = np.abs(C * X_rand - posiciones[i])
                    X_new = X_rand - A * D_rand
            else:
                D_leader = np.abs(X_best - posiciones[i])
                X_new = D_leader * np.exp(params.b * l) * np.cos(2.0 * np.pi * l) + X_best

            X_new = np.clip(X_new, -params.v_max, params.v_max)
            posiciones[i] = X_new

            # Binarización LB2
            nueva_sol, nueva_val = binarizar_posicion(
                X_new, poblacion_bin[i], inst,
                G1, G2, G3, params.v_max,
            )

            if nueva_val >= fitnesses[i]:
                poblacion_bin[i] = nueva_sol
                fitnesses[i]     = nueva_val

        best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        fit_best_actual = fitnesses[best_idx]

        if fit_best_actual > mejor_val:
            mejor_val = fit_best_actual
            mejor_sol = poblacion_bin[best_idx].copy()

        historial.append(mejor_val)
        historial_inst.append(fit_best_actual)

        # ── Stagnation check ──────────────────────────────────────────────
        dtw_info = {}
        if monitor is not None:
            status = monitor.update(mejor_val)
            dtw_info = status.copy()
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
                dtw_info_hist.append(dtw_info)
                if verbose:
                    print(f"    [Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break
        else:
            G1 = interpolar_G(it, params.iterations, params.G1_i, params.G1_f)
            G2 = interpolar_G(it, params.iterations, params.G2_i, params.G2_f)
            G3 = interpolar_G(it, params.iterations, params.G3_i, params.G3_f)

        dtw_info_hist.append(dtw_info)

    return WOAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
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
