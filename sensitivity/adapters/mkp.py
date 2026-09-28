"""Offline single-seed adapter for the original rotational MKP portfolio."""

import math
import time

from hybrid_mkp.mkp_core.data_loader import parsear_instancias
from hybrid_mkp.mkp_core.problem import MKPInstance
from hybrid_mkp import orchestrator

from .common import effective_monitor, local_path, result_record, seed_rng


def load_case(case):
    path = local_path(case.get("path", case.get("file")))
    index = case.get("instance_index", case.get("index", 0))
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("MKP instance_index must be a nonnegative integer")
    instances = parsear_instancias(path.read_text(encoding="utf-8"))
    if index >= len(instances):
        raise ValueError("MKP instance_index is outside the local data file")
    instance = MKPInstance.from_dict(instances[index])
    reference = case.get("reference_value")
    if reference is not None:
        if isinstance(reference, bool) or not isinstance(reference, (int, float)) or (
            not math.isfinite(reference) or reference <= 0
        ):
            raise ValueError("MKP reference_value must be positive and finite")
        if not isinstance(case.get("reference_source"), str) or not case["reference_source"].strip():
            raise ValueError("An explicit MKP reference requires reference_source")
        instance.valor_optimo = float(reference)
    return instance


def run_case(case, cfg, seed, budget, max_epoch_iters=None):
    instance = load_case(case)
    seed_rng(seed)
    started = time.perf_counter()
    result = orchestrator.ejecutar_pipeline(
        instance, max_iters=budget, stag_cfg=cfg, verbose=False,
        max_epoch_iters=max_epoch_iters,
    )
    feasible = instance.es_factible(result.mejor_solucion_global)
    return result_record(
        result, metric_name="gap_pct", metric=result.gap_pct if feasible else None,
        feasible=feasible, budget=budget, elapsed_seconds=time.perf_counter() - started,
        effective=effective_monitor(
            cfg, orchestrator.POOL_POBLACIONAL + orchestrator.POOL_TRAYECTORIA,
            ("GA", "ILS", "WOA", "GWO"),
        ),
        reference_value=instance.valor_optimo if instance.valor_optimo > 0 else None,
        reference_source=case.get("reference_source"),
    )
