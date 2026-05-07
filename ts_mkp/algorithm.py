"""
ts_mkp/algorithm.py
-------------------
Motor principal del algoritmo Tabu Search (TS) para el MKP.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from mkp_core.problem import MKPInstance
from mkp_core.repair  import reparar_solucion
from ts_mkp.neighborhood import obtener_mejor_vecino
from ts_mkp.rescues      import salto_heuristico, salto_lp, ruin_and_recreate_sol
from dtw_stagnation   import StagnationConfig, StagnationMonitor


@dataclass
class TSParams:
    epochs          : int   = 10
    iterations      : int   = 2000
    tabu_tenure     : int   = 10
    neighborhood_sz : int   = 30
    use_stagnation  : bool  = True
    stag_cfg        : StagnationConfig | None = None
    stag_strategy   : str   = "random_restart"
    stag_max_fires  : int   = 4


@dataclass
class TSEpochResult:
    epoch_idx       : int
    mejor_valor     : float
    iteraciones     : int
    stagnation_fires: int
    historial       : list[float] = field(default_factory=list)


@dataclass
class TSResult:
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


def ejecutar_epoch(
    inst      : MKPInstance,
    params    : TSParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
) -> TSEpochResult:
    
    # 1. Generar solucion inicial aleatoria y repararla
    sol_actual = [random.randint(0, 1) for _ in range(inst.n)]
    sol_actual, val_actual = reparar_solucion(sol_actual, inst)

    mejor_sol = sol_actual.copy()
    mejor_val = val_actual

    tabu_list = {}  # index -> iteracion de expiracion
    historial = []
    stag_fires = 0

    monitor = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    # Variables dinamicas
    current_tenure = params.tabu_tenure
    num_flips      = 1
    v2_exploring   = True

    if verbose:
        print(f"\n== Epoch {epoch_idx + 1}/{params.epochs} ====================")

    for it in range(params.iterations):
        
        # Obtener vecino
        vecino, val_vecino, flip_idx = obtener_mejor_vecino(
            sol_actual   = sol_actual,
            inst         = inst,
            tabu_list    = tabu_list,
            iter_actual  = it,
            mejor_global = mejor_val,
            max_evals    = params.neighborhood_sz,
            num_flips    = num_flips,
        )

        # Moverse
        sol_actual = vecino
        val_actual = val_vecino
        
        # Registrar tabu
        if flip_idx != -1:
            tabu_list[flip_idx] = it + current_tenure

        # Actualizar global
        if val_actual > mejor_val:
            mejor_val = val_actual
            mejor_sol = sol_actual.copy()

        historial.append(mejor_val)

        # Monitoreo de estancamiento
        if monitor is not None:
            status = monitor.update(mejor_val)
            if status.get("fire"):
                max_f = params.stag_max_fires
                if max_f > 0 and stag_fires >= max_f:
                    pass
                else:
                    stag_fires += 1
                    monitor.reset()
                    
                    s = params.stag_strategy
                    if verbose:
                        print(f"    [Stagnation] Fire #{stag_fires} @ iter {it + 1} -> {s}")

                    # --- Aplicar Rescates ---
                    if s == "random_restart":
                        tabu_list.clear()
                        # Flip 5 bits random
                        for _ in range(5):
                            idx = random.randint(0, inst.n - 1)
                            sol_actual[idx] = 1 - sol_actual[idx]
                        sol_actual, val_actual = reparar_solucion(sol_actual, inst)
                        
                    elif s == "v1_exploit":
                        # Apagar tabu, usar 2-flips para rascar fondo de olla
                        current_tenure = 0
                        num_flips      = 2
                        
                    elif s == "v2_cycle":
                        if v2_exploring:
                            current_tenure = 30
                            v2_exploring   = False
                        else:
                            current_tenure = 2
                            v2_exploring   = True
                            
                    elif s == "v3_explore":
                        tabu_list.clear()
                        sol_actual = [random.randint(0, 1) for _ in range(inst.n)]
                        sol_actual, val_actual = reparar_solucion(sol_actual, inst)
                        
                    elif s == "v4_nonlinear":
                        current_tenure = int(params.tabu_tenure * math.exp(0.5 * stag_fires))
                        
                    elif s == "v5_heuristic":
                        tabu_list.clear()
                        sol_actual, val_actual = salto_heuristico(inst)
                        
                    elif s == "v6_lp":
                        sol_actual, val_actual, _ = salto_lp(inst, sol_actual)
                        
                    elif s == "v7_tabu_lp":
                        sol_actual, val_actual, forzados = salto_lp(inst, sol_actual)
                        for idx in forzados:
                            # Tabu largo para los forzados por LP
                            tabu_list[idx] = it + 100
                            
                    elif s == "v8_ruin_recreate":
                        tabu_list.clear()
                        sol_actual, val_actual = ruin_and_recreate_sol(sol_actual, inst)

                    # Restaurar dinamicas
                    if s not in ("v1_exploit", "v2_cycle", "v4_nonlinear"):
                        current_tenure = params.tabu_tenure
                        num_flips      = 1

    if verbose:
        print(f"  [Epoch {epoch_idx + 1}] iters={params.iterations} | mejor={mejor_val:.1f} | stag_fires={stag_fires}")

    return TSEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = params.iterations,
        stagnation_fires = stag_fires,
        historial        = historial,
    )


def ejecutar_ts(inst: MKPInstance, params: TSParams, verbose: bool = True) -> TSResult:
    epochs_res = []
    mejor_g_v  = -float("inf")
    mejor_g_s  = []
    
    for e in range(params.epochs):
        res = ejecutar_epoch(inst, params, e, verbose)
        epochs_res.append(res)
        if res.mejor_valor > mejor_g_v:
            mejor_g_v = res.mejor_valor
            mejor_g_s = []

    return TSResult(
        epochs             = epochs_res,
        mejor_valor_global = mejor_g_v,
        mejor_sol_global   = mejor_g_s,
        valor_optimo       = inst.valor_optimo,
    )
