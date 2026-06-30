"""
continuous_benchmark/orchestrator.py
------------------------------------
Orquestador del Pipeline Hibrido de Rotacion de Metaheuristicas
para funciones matematicas continuas (minimizacion).

Rota unicamente entre MH poblacionales: GA, PSO, GWO, WOA, EHO.
El monitor DTW detecta estancamiento y aborta el epoch para rotar
a la siguiente MH del pool.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dtw_stagnation import StagnationConfig

from continuous_benchmark.mh.ga  import GAParams,  ejecutar_epoch as _ga_epoch
from continuous_benchmark.mh.pso import PSOParams, ejecutar_epoch as _pso_epoch
from continuous_benchmark.mh.gwo import GWOParams, ejecutar_epoch as _gwo_epoch
from continuous_benchmark.mh.woa import WOAParams, ejecutar_epoch as _woa_epoch
from continuous_benchmark.mh.eho import EHOParams, ejecutar_epoch as _eho_epoch


POOL_POBLACIONAL = ["GA", "PSO", "GWO", "WOA", "EHO"]

COLORES_MH = {
    "GA" : "#4CAF50",
    "PSO": "#2196F3",
    "GWO": "#9C27B0",
    "EHO": "#00BCD4",
    "WOA": "#E040FB",
}


@dataclass
class SwitchLog:
    mh_nombre   : str
    tipo        : str
    mejor_valor : float
    t_inicio    : float
    t_fin       : float
    n_iters     : int
    dtw_deltas  : list = None


@dataclass
class PipelineResult:
    mejor_valor_global    : float
    mejor_solucion_global : list[float]
    historial_global      : list[float]
    historial_inst_global : list[float]
    dtw_deltas_global     : list[float]
    log_switches          : list[SwitchLog]
    valor_optimo          : float

    @property
    def gap_pct(self) -> float | None:
        if self.valor_optimo == 0 and self.mejor_valor_global == 0:
            return 0.0
        if self.valor_optimo == 0:
            return abs(self.mejor_valor_global)
        return 100.0 * abs(self.mejor_valor_global - self.valor_optimo) / max(1e-12, abs(self.valor_optimo))

    @property
    def n_switches(self) -> int:
        return len(self.log_switches)


def ejecutar_pipeline(
    func               ,  # ContinuousFunction
    tiempo_max         : float = 120.0,
    stag_cfg           : StagnationConfig | None = None,
    pop_injection_mode : str = "mixed",
    verbose            : bool = True,
) -> PipelineResult:

    if stag_cfg is None:
        stag_cfg = StagnationConfig()

    solucion_global   : np.ndarray | None = None
    valor_global      : float = float("inf")
    historial_global  : list[float] = []
    historial_inst_global : list[float] = []
    dtw_deltas_global : list[float] = []
    log_switches      : list[SwitchLog] = []

    epoch_ctr = 0
    t_inicio  = time.time()

    if verbose:
        print("\n" + "=" * 62)
        print("  CONTINUOUS PIPELINE HIBRIDO DTW -- INICIO")
        print(f"  Funcion    : {func.name} (Dim={func.n_dim}, [{func.lb}, {func.ub}])")
        print(f"  Tiempo max : {tiempo_max}s")
        print(f"  Pool       : {POOL_POBLACIONAL}")
        print("=" * 62)

    while (time.time() - t_inicio) < tiempo_max:
        t_mh_inicio = time.time() - t_inicio

        mh = random.choice(POOL_POBLACIONAL)
        tipo = "poblacional"

        if verbose:
            elapsed = time.time() - t_inicio
            print(f"\n  [{elapsed:06.1f}s] > {mh:4s} | global = {valor_global:.6f}")

        resultado = _ejecutar_mh(
            mh_nombre          = mh,
            func               = func,
            solucion_global    = solucion_global,
            stag_cfg           = stag_cfg,
            pop_injection_mode = pop_injection_mode,
            epoch_idx          = epoch_ctr,
            verbose            = verbose,
        )
        epoch_ctr += 1

        if resultado.mejor_valor < valor_global:
            valor_global    = resultado.mejor_valor
            solucion_global = np.array(resultado.mejor_solucion)

        historial_global.extend(resultado.historial)
        historial_inst_global.extend(getattr(resultado, 'historial_inst', []) or [])

        mh_deltas = getattr(resultado, 'dtw_deltas', []) or []
        n_hist    = len(resultado.historial)
        n_deltas  = len(mh_deltas)
        padded    = [float('nan')] * (n_hist - n_deltas) + list(mh_deltas)
        dtw_deltas_global.extend(padded)

        t_mh_fin = time.time() - t_inicio
        n_iters  = len(resultado.historial)

        log_switches.append(SwitchLog(
            mh_nombre   = mh,
            tipo        = tipo,
            mejor_valor = resultado.mejor_valor,
            t_inicio    = t_mh_inicio,
            t_fin       = t_mh_fin,
            n_iters     = n_iters,
            dtw_deltas  = mh_deltas,
        ))

        if verbose:
            dur = t_mh_fin - t_mh_inicio
            print(f"          Duracion: {dur:.1f}s | iters: {n_iters} "
                  f"| mejor MH: {resultado.mejor_valor:.6f} "
                  f"| global: {valor_global:.6f}")

    elapsed_total = time.time() - t_inicio
    if verbose:
        print(f"\n  [{elapsed_total:.1f}s] TIEMPO AGOTADO")
        print(f"  Mejor global  : {valor_global:.6f}")
        print(f"  Total switches: {len(log_switches)}")
        print("=" * 62)

    return PipelineResult(
        mejor_valor_global    = valor_global,
        mejor_solucion_global = solucion_global.tolist() if solucion_global is not None else [],
        historial_global      = historial_global,
        historial_inst_global = historial_inst_global,
        dtw_deltas_global     = dtw_deltas_global,
        log_switches          = log_switches,
        valor_optimo          = func.optimum,
    )


def _ejecutar_mh(
    mh_nombre          : str,
    func               ,
    solucion_global    : np.ndarray | None,
    stag_cfg           : StagnationConfig,
    pop_injection_mode : str,
    epoch_idx          : int,
    verbose            : bool,
):
    if mh_nombre == "GA":
        params = GAParams(
            pop_size=50, generations=300, epochs=1,
            injection_mode=pop_injection_mode,
            use_stagnation=True, stag_cfg=stag_cfg,
        )
        return _ga_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose,
                         sol_inyectada=solucion_global)

    elif mh_nombre == "PSO":
        params = PSOParams(
            pop_size=30, iterations=300, epochs=1,
            injection_mode=pop_injection_mode,
            use_stagnation=True, stag_cfg=stag_cfg,
        )
        return _pso_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose,
                          sol_inyectada=solucion_global)

    elif mh_nombre == "GWO":
        params = GWOParams(
            pop_size=30, iterations=300, epochs=1,
            injection_mode=pop_injection_mode,
            use_stagnation=True, stag_cfg=stag_cfg,
        )
        return _gwo_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose,
                          sol_inyectada=solucion_global)

    elif mh_nombre == "WOA":
        params = WOAParams(
            pop_size=30, iterations=300, epochs=1,
            injection_mode=pop_injection_mode,
            use_stagnation=True, stag_cfg=stag_cfg,
        )
        return _woa_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose,
                          sol_inyectada=solucion_global)

    elif mh_nombre == "EHO":
        params = EHOParams(
            pop_size=30, iterations=300, epochs=1,
            injection_mode=pop_injection_mode,
            use_stagnation=True, stag_cfg=stag_cfg,
        )
        return _eho_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose,
                          sol_inyectada=solucion_global)

    else:
        raise ValueError(f"MH desconocida: '{mh_nombre}'")
