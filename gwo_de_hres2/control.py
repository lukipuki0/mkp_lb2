"""
One run of GWO-DE on HRES2 under a gwo_de_dtw profile policy.

The profile controller is gwo_de_dtw.controller.ProfileController, unchanged.
Its monitor maximizes, so it observes -best_objective each generation (the
HRES2 metaheuristics feed the monitor the same way). A profile decided after
generation g applies from generation g+1.

Spec: odd/tasks/gwo-de-hres2.md (S4, S5).
"""

from dataclasses import asdict
import math
import time
from typing import Dict, List, Optional

import numpy as np

from dtw_stagnation import StagnationConfig
from gwo_de_dtw import config as mkp_config
from gwo_de_dtw.controller import ProfileController
from gwo_de_dtw.hybrid import Profile

from . import config
from .search import GWODEContinuous

DECODED_KEYS = ("wind_mw", "pv_mw", "n_el_units", "electrolyzer_mw", "battery_mw",
                "battery_duration_h", "battery_mwh", "total_h2_kg", "electrolyzer_cf")

# G3 is an LB2 parameter; it has no meaning in the continuous search.
EXPLOIT = Profile("exploit", a=mkp_config.EXPLOIT_A, F=mkp_config.EXPLOIT_F,
                  CR=mkp_config.EXPLOIT_CR, G3=0.0)
EXPLORE = Profile("explore", a=mkp_config.EXPLORE_A, F=mkp_config.EXPLORE_F,
                  CR=mkp_config.EXPLORE_CR, G3=0.0)


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def run_search(func, monitor_cfg: StagnationConfig, policy: str,
               pop_size: int = config.POP_SIZE, iterations: int = config.ITERATIONS,
               seed: int = config.SEED) -> Dict:
    """Exactly `iterations` generations; returns one JSON-ready record."""
    controller = ProfileController(monitor_cfg, exploit=EXPLOIT, explore=EXPLORE, policy=policy)
    search = GWODEContinuous(func, np.random.default_rng(seed), pop_size)
    started = time.perf_counter()
    search.initialize()

    best_series: List[float] = []
    profile_series: List[str] = []
    parameter_series: Dict[str, List[float]] = {"a": [], "F": [], "CR": []}
    intensity_series: List[Optional[float]] = []
    step_stats: List[Dict] = []
    changes: List[int] = []
    for generation in range(iterations):
        search.set_profile(controller.profile)
        profile_series.append(controller.profile.name)
        for key in parameter_series:
            parameter_series[key].append(getattr(controller.profile, key))
        intensity_series.append(controller.intensity)
        best = search.step()
        step_stats.append(dict(search.last_step_stats))
        best_series.append(best)
        if controller.observe(-best)["changed"]:
            changes.append(generation)
    elapsed = time.perf_counter() - started

    solution = search.best_solution()
    info = func.get_info(solution)
    decoded = {key: info.get(key) if key == "n_el_units" else _finite(info.get(key))
               for key in DECODED_KEYS}
    return {
        "policy": policy,
        "signal": "ddtw" if monitor_cfg.use_ddtw else "dtw",
        "seed": seed,
        "pop_size": pop_size,
        "iterations": iterations,
        "generations": len(best_series),
        "evaluations": search.evaluations,
        "objective": search.best_objective,
        "objective_repr": repr(search.best_objective),
        "feasible": None if info.get("feasible") is None else bool(info["feasible"]),
        "physical_lcoe": _finite(info.get("lcoe_cny_per_kwh")),
        "physical_lcoh": _finite(info.get("lcoh_cny_per_kg")),
        "physical_agsr": _finite(info.get("agsr")),
        "solution": [float(v) for v in solution],
        "solution_repr": [repr(float(v)) for v in solution],
        "decoded": decoded,
        "monitor_config": asdict(monitor_cfg),
        "best_series": best_series,
        "profile_series": profile_series,
        "parameter_series": parameter_series,
        "intensity_series": intensity_series,
        "changes": changes,
        "step_stats": step_stats,
        "elapsed_seconds": elapsed,
    }
