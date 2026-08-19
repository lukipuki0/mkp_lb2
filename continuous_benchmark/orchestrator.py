"""
continuous_benchmark/orchestrator.py
------------------------------------
Orquestador del Pipeline Hibrido de Rotacion de Metaheuristicas
para funciones matematicas continuas (minimizacion).

Contiene SOLO metaheuristicas poblacionales: PSO, GWO, WOA, EHO, ACO, GA, ABC.
El monitor DTW detecta estancamiento y aborta el epoch para rotar a la siguiente MH.

Para HRES2-H2 (que incluye MHs de trayectoria) usar HRES2-H2/orchestrator.py.
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
from continuous_benchmark.mh.aco import ACOParams, ejecutar_epoch as _aco_epoch
from continuous_benchmark.mh.abc import ABCParams, ejecutar_epoch as _abc_epoch


POOL_POBLACIONAL = ["PSO", "GWO", "WOA", "EHO", "ACO"]

COLORES_MH = {
    "PSO": "#2196F3",
    "GWO": "#9C27B0",
    "EHO": "#00BCD4",
    "ACO": "#8D6E63",
    "WOA": "#E040FB",
    "GA" : "#4CAF50",
    "ABC": "#FFC107",
    # Trayectoria (usados en HRES2-H2)
    "ILS": "#E91E63",
    "VNS": "#00E676",
    "TS" : "#795548",
    "SA" : "#FF5722",
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
    dtw_info_global       : list[dict]
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
    max_iters          : int | None = 1000,
    tiempo_max         : float | None = None,
    stag_cfg           : StagnationConfig | None = None,
    pop_injection_mode : str = "mixed",
    verbose            : bool = True,
    on_epoch_callback  = None,
    pool_poblacional   : list[str] | None = None,
    pool_trayectoria   : list[str] | None = None,
    ejecutar_mh_fn     = None,   # funcion personalizada: (mh_nombre, func, sol_global, stag_cfg, mode, epoch, verbose) -> resultado
) -> PipelineResult:

    if max_iters is None and tiempo_max is None:
        max_iters = 1000

    if stag_cfg is None:
        stag_cfg = StagnationConfig()

    if pool_poblacional is None:
        pool_poblacional = list(POOL_POBLACIONAL)

    _mh_fn = ejecutar_mh_fn if ejecutar_mh_fn is not None else _ejecutar_mh

    solucion_global   : np.ndarray | None = None
    valor_global      : float = float("inf")
    historial_global  : list[float] = []
    historial_inst_global : list[float] = []
    dtw_deltas_global : list[float] = []
    dtw_info_global   : list[dict]  = []
    log_switches      : list[SwitchLog] = []

    epoch_ctr   = 0
    fase_actual = "poblacional"
    t_inicio    = time.time()

    if verbose:
        print("\n" + "=" * 62)
        print("  CONTINUOUS PIPELINE HIBRIDO DTW -- INICIO")
        print(f"  Funcion    : {func.name} (Dim={func.n_dim}, [{func.lb}, {func.ub}])")
        lim_str = f"Max Iters: {max_iters}" if max_iters is not None else f"Tiempo max: {tiempo_max}s"
        print(f"  Condicion  : {lim_str}")
        print(f"  Pool Poblacional : {pool_poblacional}")
        print(f"  Pool Trayectoria : {pool_trayectoria if pool_trayectoria else 'Ninguno'}")
        print("=" * 62)

    while True:
        if max_iters is not None and len(historial_global) >= max_iters:
            break
        if max_iters is None and tiempo_max is not None and (time.time() - t_inicio) >= tiempo_max:
            break

        t_mh_inicio = time.time() - t_inicio

        if pool_trayectoria:
            if fase_actual == "poblacional":
                mh = random.choice(pool_poblacional)
                tipo = "poblacional"
                fase_actual = "trayectoria"
            else:
                mh = random.choice(pool_trayectoria)
                tipo = "trayectoria"
                fase_actual = "poblacional"
        else:
            mh = random.choice(pool_poblacional)
            tipo = "poblacional"

        if verbose:
            elapsed = time.time() - t_inicio
            print(f"\n  [{elapsed:06.1f}s] > {mh:4s} | global = {valor_global:.6f}")

        resultado = _mh_fn(
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

        mh_dtw_info = getattr(resultado, 'dtw_info_hist', []) or []
        if len(mh_dtw_info) < n_hist:
            mh_dtw_info = mh_dtw_info + [{}] * (n_hist - len(mh_dtw_info))
        dtw_info_global.extend(mh_dtw_info)

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

        if on_epoch_callback is not None:
            on_epoch_callback(
                epoch         = epoch_ctr,
                mh            = mh,
                tipo          = tipo,
                iters_total   = len(historial_global),
                mejor_valor   = valor_global,
                mejor_solucion= solucion_global,
            )

    elapsed_total = time.time() - t_inicio
    if verbose:
        print(f"\n  [{elapsed_total:.1f}s] EJECUCIÓN FINALIZADA")
        print(f"  Mejor global  : {valor_global:.6f}")
        print(f"  Total switches: {len(log_switches)}")
        print("=" * 62)

    return PipelineResult(
        mejor_valor_global    = valor_global,
        mejor_solucion_global = solucion_global.tolist() if solucion_global is not None else [],
        historial_global      = historial_global,
        historial_inst_global = historial_inst_global,
        dtw_deltas_global     = dtw_deltas_global,
        dtw_info_global       = dtw_info_global,
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
    """Ejecuta una MH poblacional. Para trayectoria, usar HRES2-H2/orchestrator.py."""
    if mh_nombre == "GA":
        params = GAParams(pop_size=50, generations=300, epochs=1,
                          injection_mode=pop_injection_mode, use_stagnation=True, stag_cfg=stag_cfg)
        return _ga_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=solucion_global)

    elif mh_nombre == "PSO":
        params = PSOParams(pop_size=30, iterations=300, epochs=1,
                           injection_mode=pop_injection_mode, use_stagnation=True, stag_cfg=stag_cfg)
        return _pso_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=solucion_global)

    elif mh_nombre == "GWO":
        params = GWOParams(pop_size=30, iterations=300, epochs=1,
                           injection_mode=pop_injection_mode, use_stagnation=True, stag_cfg=stag_cfg)
        return _gwo_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=solucion_global)

    elif mh_nombre == "WOA":
        params = WOAParams(pop_size=30, iterations=300, epochs=1,
                           injection_mode=pop_injection_mode, use_stagnation=True, stag_cfg=stag_cfg)
        return _woa_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=solucion_global)

    elif mh_nombre == "EHO":
        params = EHOParams(pop_size=30, iterations=300, epochs=1,
                           injection_mode=pop_injection_mode, use_stagnation=True, stag_cfg=stag_cfg)
        return _eho_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=solucion_global)

    elif mh_nombre == "ACO":
        params = ACOParams(pop_size=30, iterations=300, epochs=1,
                           injection_mode=pop_injection_mode, use_stagnation=True, stag_cfg=stag_cfg)
        return _aco_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=solucion_global)

    elif mh_nombre == "ABC":
        params = ABCParams(pop_size=30, iterations=300, epochs=1,
                           injection_mode=pop_injection_mode, use_stagnation=True, stag_cfg=stag_cfg)
        return _abc_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=solucion_global)

    else:
        raise ValueError(f"MH desconocida o de trayectoria: '{mh_nombre}'. "
                         f"Las MHs de trayectoria solo estan disponibles en HRES2-H2/orchestrator.py")


def ejecutar_mh_standalone(func, mh_nombre: str, max_iters: int = 1000):
    """Ejecuta una MH poblacional standalone."""
    if mh_nombre == "GA":
        return _ga_epoch(func, GAParams(pop_size=30, generations=max_iters, use_stagnation=False), verbose=False)
    elif mh_nombre == "PSO":
        return _pso_epoch(func, PSOParams(pop_size=30, iterations=max_iters, use_stagnation=False), verbose=False)
    elif mh_nombre == "GWO":
        return _gwo_epoch(func, GWOParams(pop_size=30, iterations=max_iters, use_stagnation=False), verbose=False)
    elif mh_nombre == "WOA":
        return _woa_epoch(func, WOAParams(pop_size=30, iterations=max_iters, use_stagnation=False), verbose=False)
    elif mh_nombre == "EHO":
        return _eho_epoch(func, EHOParams(pop_size=30, iterations=max_iters, use_stagnation=False), verbose=False)
    elif mh_nombre == "ACO":
        return _aco_epoch(func, ACOParams(pop_size=30, iterations=max_iters, use_stagnation=False), verbose=False)
    elif mh_nombre == "ABC":
        return _abc_epoch(func, ABCParams(pop_size=30, iterations=max_iters, use_stagnation=False), verbose=False)
    else:
        raise ValueError(f"MH poblacional no soportada: '{mh_nombre}'. "
                         f"Para trayectoria usar HRES2-H2/orchestrator.py")
