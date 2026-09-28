"""Single-seed adapter using the bundled official CEC2022 data only."""

import time
import numpy as np

from continuous_benchmark.funciones_cec2022 import get_test_functions
from continuous_benchmark import orchestrator

from .common import effective_monitor, result_record, seed_rng


def load_case(case):
    function_id = case.get("function_id", case.get("function"))
    dimension = case.get("dimension", 10)
    if isinstance(function_id, bool) or not isinstance(function_id, int) or not 1 <= function_id <= 12:
        raise ValueError("CEC function_id must be an integer from 1 to 12")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension not in (10, 20):
        raise ValueError("CEC dimension must be 10 or 20")
    func = get_test_functions(dimension)[function_id - 1]
    # Force required shift/rotation/shuffle validation before seeding optimization.
    func.func(np.zeros(dimension))
    return func


def run_case(case, cfg, seed, budget, max_epoch_iters=None):
    func = load_case(case)
    seed_rng(seed)
    started = time.perf_counter()
    result = orchestrator.ejecutar_pipeline(
        func, max_iters=budget, stag_cfg=cfg, verbose=False,
        max_epoch_iters=max_epoch_iters,
    )
    solution = np.asarray(result.mejor_solucion_global, dtype=float)
    feasible = solution.shape == (func.n_dim,) and bool(
        np.all(np.isfinite(solution)) and np.all(solution >= func.lb) and np.all(solution <= func.ub)
    )
    return result_record(
        result, metric_name="optimum_error",
        metric=float(result.mejor_valor_global) - func.optimum if feasible else None,
        feasible=feasible, budget=budget, elapsed_seconds=time.perf_counter() - started,
        effective=effective_monitor(cfg, orchestrator.POOL_POBLACIONAL),
        reference_value=func.optimum, reference_source="official CEC2022 function bias",
    )
