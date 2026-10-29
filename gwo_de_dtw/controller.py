"""
DTW/DDTW profile controller and run loop for the GWO-DE hybrid.

The controller watches the best-so-far fitness (last W generations) and
switches the joint GWO/DE profile:
    exploit + fire                        -> explore
    explore + recent improvement          -> exploit
        (no_improve_len < plateau_max on a ready observation)
    explore + still stagnant              -> stay in explore

After every profile change the monitor is replaced (fresh window): window,
no_improve_len, trigger_streak and threshold histories are discarded, so no
decision until W new observations generated with the new profile exist.
Global best and population are not reset.

A change decided after generation g applies from generation g+1.

The run stops after `iterations` generations (default config.ITERATIONS).

Spec: odd/tasks/gwo-de-dtw-hybrid.md (S5-S8, S10, S12, S14, S15).
"""

from dataclasses import asdict
from typing import Dict, List

import numpy as np

from . import config
from dtw_stagnation import StagnationConfig, StagnationMonitor

from .hybrid import EXPLOIT, EXPLORE, GWODEHybrid, Profile


class ProfileController:
    def __init__(self, cfg: StagnationConfig,
                 exploit: Profile = EXPLOIT, explore: Profile = EXPLORE):
        self.cfg = cfg
        self.exploit = exploit
        self.explore = explore
        self.profile: Profile = exploit
        self.monitor = StagnationMonitor(cfg=cfg)

    def _next_profile(self, out: Dict) -> Profile:
        if self.profile is self.exploit:
            return self.explore if out["fire"] else self.exploit
        return self.exploit if out["no_improve_len"] < self.cfg.plateau_max else self.explore

    def observe(self, best_fitness: float) -> Dict:
        """Feed one best-so-far observation; returns the monitor output plus the decision."""
        out = dict(self.monitor.update(best_fitness))
        changed = False
        if out["ready"]:
            new = self._next_profile(out)
            if new is not self.profile:
                self.profile = new
                changed = True
                self.monitor = StagnationMonitor(cfg=self.cfg)
        out["changed"] = changed
        out["profile"] = self.profile.name
        return out


def run_hybrid(
    inst: dict,
    monitor_cfg: StagnationConfig,
    pop_size: int = config.POP_SIZE,
    iterations: int = config.ITERATIONS,
    seed: int = config.SEED,
    G1: float = config.G1,
    G2: float = config.G2,
    g3_exploit: float = config.EXPLOIT_G3,
    g3_explore: float = config.EXPLORE_G3,
    binarization: str = config.BINARIZATION,
) -> dict:
    """One run of the hybrid: exactly `iterations` generations.

    g3_exploit and g3_explore build the two profiles for this run (a, F and CR
    stay the config values). The same value in both is a fixed G3.
    """
    exploit = Profile("exploit", a=config.EXPLOIT_A, F=config.EXPLOIT_F,
                      CR=config.EXPLOIT_CR, G3=g3_exploit)
    explore = Profile("explore", a=config.EXPLORE_A, F=config.EXPLORE_F,
                      CR=config.EXPLORE_CR, G3=g3_explore)
    controller = ProfileController(monitor_cfg, exploit=exploit, explore=explore)
    hybrid = GWODEHybrid(inst, np.random.default_rng(seed),
                         pop_size=pop_size, G1=G1, G2=G2,
                         binarization=binarization)
    hybrid.initialize()

    best_series: List[float] = []
    profile_series: List[str] = []
    # G3 is the active profile's LB2 flip probability. Sigmoid does not use it,
    # but it is still recorded so the series follows profile_series.
    parameter_series: Dict[str, List[float]] = {"a": [], "F": [], "CR": [], "G3": []}
    monitor_hist: List[Dict] = []
    step_stats: List[Dict] = []
    changes: List[int] = []

    for generation in range(iterations):
        hybrid.set_profile(controller.profile)
        profile_series.append(controller.profile.name)
        for key in parameter_series:
            parameter_series[key].append(getattr(controller.profile, key))
        best = hybrid.step()
        step_stats.append(dict(hybrid.last_step_stats))
        best_series.append(best)

        out = controller.observe(best)
        monitor_hist.append(out)
        if out["changed"]:
            changes.append(generation)

    solution, fitness = hybrid.get_best()
    optimum = inst.get("optimo", 0.0)
    return {
        "monitor_config": asdict(monitor_cfg),
        "use_ddtw": monitor_cfg.use_ddtw,
        "window": monitor_cfg.window,
        "seed": seed,
        "binarization": hybrid.binarization,
        "G": {"G1": G1, "G2": G2,
              "G3_exploit": g3_exploit, "G3_explore": g3_explore},  # only used by lb2
        "pop_size": pop_size,
        "iterations": iterations,
        "evaluations": hybrid.evaluations,
        "generations": len(best_series),
        "best_fitness": fitness,
        "best_solution": solution,
        "known_optimum": optimum,
        "gap_pct": (100.0 * (optimum - fitness) / optimum) if optimum > 0 else None,
        "best_series": best_series,
        "profile_series": profile_series,
        "parameter_series": parameter_series,
        "changes": changes,
        "monitor": monitor_hist,
        "step_stats": step_stats,
    }
