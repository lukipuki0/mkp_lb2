"""Single-seed HRES adapter with objective and feasibility kept separate."""

import time
import numpy as np

from .common import effective_monitor, load_hres_module, result_record, seed_rng


def load_case(case):
    model = load_hres_module("wpeb_model.py")
    return model.HRES2Function()


def run_case(case, cfg, seed, budget, max_epoch_iters=None):
    func = load_case(case)
    orchestrator = load_hres_module("orchestrator.py")
    # Construction generates fixed synthetic weather and changes NumPy's RNG.
    seed_rng(seed)
    started = time.perf_counter()
    result = orchestrator.ejecutar_pipeline_hres2(
        func, max_iters=budget, stag_cfg=cfg, verbose=False,
        max_epoch_iters=max_epoch_iters,
    )
    info = func.get_info(np.asarray(result.mejor_solucion_global))
    feasible = bool(info["feasible"])
    return result_record(
        result, metric_name="lcoe_cny_per_kwh",
        metric=float(info["lcoe_cny_per_kwh"]) if feasible else None,
        feasible=feasible, budget=budget, elapsed_seconds=time.perf_counter() - started,
        effective=effective_monitor(
            cfg, orchestrator.POOL_POBLACIONAL_HRES2 + orchestrator.POOL_TRAYECTORIA_HRES2,
        ),
        raw_lcoe_cny_per_kwh=info["lcoe_cny_per_kwh"],
        constraints={"agsr": info["agsr"], "agsr_max": func.config["agsr_max"], "feasible": feasible},
        hres_info=info,
        weather_source="fixed synthetic TMY 8760-hour profile, generator seed 2008",
    )
