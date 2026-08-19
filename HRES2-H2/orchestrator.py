"""
HRES2-H2/orchestrator.py
------------------------
Orquestador HRES2-H2 que extiende el pipeline continuo con MHs de trayectoria.

Pool Poblacional : PSO, GWO, WOA, EHO, ACO, ABC
Pool Trayectoria : ILS, SA, TS, VNS  (en HRES2-H2/mh/)

Uso:
    import importlib.util, os
    _dir = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("hres2_orch", os.path.join(_dir, "orchestrator.py"))
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ejecutar_pipeline_hres2 = mod.ejecutar_pipeline_hres2
"""

from __future__ import annotations

import sys
import os
import importlib.util

_HRES2_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_HRES2_DIR)

# Asegurar que la raiz del proyecto esta en sys.path
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import numpy as np
from dtw_stagnation import StagnationConfig

# ── Pipeline base (poblacionales) ──────────────────────────────────────────────
from continuous_benchmark.orchestrator import (
    ejecutar_pipeline,
    PipelineResult,
    SwitchLog,
    COLORES_MH,
    _ejecutar_mh as _ejecutar_mh_poblacional,
    ejecutar_mh_standalone,
)


# ── Carga de MHs de trayectoria via importlib (HRES2-H2 usa guion en nombre) ──
def _load_hres2_mh(name: str):
    """Carga un modulo MH de trayectoria desde HRES2-H2/mh/<name>.py"""
    mod_name = f"hres2_mh_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = os.path.join(_HRES2_DIR, "mh", f"{name}.py")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod   # registrar ANTES de exec para que @dataclass funcione
    spec.loader.exec_module(mod)
    return mod

_ils_mod = _load_hres2_mh("ils")
_sa_mod  = _load_hres2_mh("sa")
_ts_mod  = _load_hres2_mh("ts")
_vns_mod = _load_hres2_mh("vns")

ILSParams   = _ils_mod.ILSParams
_ils_epoch  = _ils_mod.ejecutar_epoch
SAParams    = _sa_mod.SAParams
_sa_epoch   = _sa_mod.ejecutar_epoch
TSParams    = _ts_mod.TSParams
_ts_epoch   = _ts_mod.ejecutar_epoch
VNSParams   = _vns_mod.VNSParams
_vns_epoch  = _vns_mod.ejecutar_epoch


POOL_POBLACIONAL_HRES2 = ["PSO", "GWO", "WOA", "EHO", "ACO", "ABC"]
POOL_TRAYECTORIA_HRES2 = ["ILS", "SA", "TS", "VNS"]


def _ejecutar_mh_hres2(
    mh_nombre          : str,
    func               ,
    solucion_global    : np.ndarray | None,
    stag_cfg           : StagnationConfig,
    pop_injection_mode : str,
    epoch_idx          : int,
    verbose            : bool,
):
    """Ejecuta una MH: delega a poblacional o trayectoria HRES2."""
    sol = solucion_global

    if mh_nombre == "ILS":
        params = ILSParams(iterations=300, epochs=1, use_stagnation=True, stag_cfg=stag_cfg)
        return _ils_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=sol)

    elif mh_nombre == "SA":
        params = SAParams(iterations=300, epochs=1, use_stagnation=True, stag_cfg=stag_cfg)
        return _sa_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=sol)

    elif mh_nombre == "TS":
        params = TSParams(iterations=300, epochs=1, use_stagnation=True, stag_cfg=stag_cfg)
        return _ts_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=sol)

    elif mh_nombre == "VNS":
        params = VNSParams(iterations=300, epochs=1, use_stagnation=True, stag_cfg=stag_cfg)
        return _vns_epoch(func, params, epoch_idx=epoch_idx, verbose=verbose, sol_inyectada=sol)

    else:
        return _ejecutar_mh_poblacional(
            mh_nombre=mh_nombre, func=func, solucion_global=solucion_global,
            stag_cfg=stag_cfg, pop_injection_mode=pop_injection_mode,
            epoch_idx=epoch_idx, verbose=verbose,
        )


def ejecutar_pipeline_hres2(
    func,
    max_iters          : int | None = 1000,
    tiempo_max         : float | None = None,
    stag_cfg           : StagnationConfig | None = None,
    verbose            : bool = True,
    on_epoch_callback  = None,
    pool_poblacional   : list[str] | None = None,
    pool_trayectoria   : list[str] | None = None,
) -> PipelineResult:
    """Pipeline HRES2-H2 con MHs poblacionales y de trayectoria."""
    return ejecutar_pipeline(
        func               = func,
        max_iters          = max_iters,
        tiempo_max         = tiempo_max,
        stag_cfg           = stag_cfg,
        verbose            = verbose,
        on_epoch_callback  = on_epoch_callback,
        pool_poblacional   = pool_poblacional or list(POOL_POBLACIONAL_HRES2),
        pool_trayectoria   = pool_trayectoria or list(POOL_TRAYECTORIA_HRES2),
        ejecutar_mh_fn     = _ejecutar_mh_hres2,
    )


def ejecutar_mh_standalone_hres2(func, mh_nombre: str, max_iters: int = 1000):
    """Ejecuta cualquier MH (poblacional o trayectoria) de forma standalone."""
    if mh_nombre == "ILS":
        return _ils_epoch(func, ILSParams(iterations=max_iters, use_stagnation=False), verbose=False)
    elif mh_nombre == "SA":
        return _sa_epoch(func, SAParams(iterations=max_iters, use_stagnation=False), verbose=False)
    elif mh_nombre == "TS":
        return _ts_epoch(func, TSParams(iterations=max_iters, use_stagnation=False), verbose=False)
    elif mh_nombre == "VNS":
        return _vns_epoch(func, VNSParams(iterations=max_iters, use_stagnation=False), verbose=False)
    else:
        return ejecutar_mh_standalone(func, mh_nombre, max_iters)
