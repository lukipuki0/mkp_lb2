"""
mh/ts.py
--------
Tabu Search (TS) para el MKP.

Versión limpia para el pipeline híbrido: solo usa estrategia "abort"
cuando el monitor DTW detecta estancamiento.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mkp_core.problem       import MKPInstance
from mkp_core.repair        import reparar_solucion
from mh.ts_neighborhood     import obtener_mejor_vecino
from dtw_stagnation         import StagnationConfig, StagnationMonitor


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class TSParams:
    """Hiperparámetros del TS."""
    epochs          : int   = 10
    iterations      : int   = 2000
    tabu_tenure     : int   = 10
    neighborhood_sz : int   = 30
    # Stagnation
    use_stagnation  : bool  = True
    stag_cfg        : StagnationConfig | None = None


@dataclass
class TSEpochResult:
    """Resultado de un epoch del TS."""
    epoch_idx       : int
    mejor_valor     : float
    iteraciones     : int
    stagnation_fires: int
    historial       : list[float] = field(default_factory=list)
    historial_inst  : list[float] = field(default_factory=list)  # fitness de sol actual (puede bajar)
    mejor_solucion  : list[int]  = field(default_factory=list)
    dtw_deltas      : list[float] = field(default_factory=list)


@dataclass
class TSResult:
    """Resultado completo del TS (todos los epochs)."""
    epochs             : list[TSEpochResult]
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


# ── Epoch individual ──────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst      : MKPInstance,
    params    : TSParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inicial: list[int] | None = None,
) -> TSEpochResult:
    """Ejecuta un epoch completo del TS con detección de estancamiento (abort)."""

    # Solución inicial: semilla del orquestador o aleatoria
    if sol_inicial is not None:
        sol_actual = list(sol_inicial)
        sol_actual, val_actual = reparar_solucion(sol_actual, inst)
    else:
        sol_actual = [random.randint(0, 1) for _ in range(inst.n)]
        sol_actual, val_actual = reparar_solucion(sol_actual, inst)

    mejor_sol = sol_actual.copy()
    mejor_val = val_actual

    tabu_list      = {}   # index -> iteración de expiración
    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0

    monitor = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):

        # Obtener vecino
        vecino, val_vecino, flip_idx = obtener_mejor_vecino(
            sol_actual   = sol_actual,
            inst         = inst,
            tabu_list    = tabu_list,
            iter_actual  = it,
            mejor_global = mejor_val,
            max_evals    = params.neighborhood_sz,
        )

        # Moverse
        sol_actual = vecino
        val_actual = val_vecino

        # Registrar tabú
        if flip_idx != -1:
            tabu_list[flip_idx] = it + params.tabu_tenure

        # Actualizar global
        if val_actual > mejor_val:
            mejor_val = val_actual
            mejor_sol = sol_actual.copy()

        historial.append(mejor_val)
        historial_inst.append(val_actual)  # fitness real de la sol actual (puede bajar)

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
                    print(f"    [Stagnation] Fire #{stag_fires} @ iter {it + 1} -> ABORT")
                break

    return TSEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = params.iterations,
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
    )


# ── Ejecución multi-epoch ────────────────────────────────────────────────────

def ejecutar_ts(
    inst: MKPInstance,
    params: TSParams,
    verbose: bool = True,
) -> TSResult:
    """Ejecuta el TS completo (todos los epochs) y retorna el TSResult."""
    epochs_res = []
    mejor_g_v  = -float("inf")
    mejor_g_s: list[int] = []

    for e in range(params.epochs):
        res = ejecutar_epoch(inst, params, e, verbose)
        epochs_res.append(res)
        if res.mejor_valor > mejor_g_v:
            mejor_g_v = res.mejor_valor
            mejor_g_s = res.mejor_solucion.copy()

    return TSResult(
        epochs             = epochs_res,
        mejor_valor_global = mejor_g_v,
        mejor_sol_global   = mejor_g_s,
        valor_optimo       = inst.valor_optimo,
    )

