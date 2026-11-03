"""Profile control. The monitor sees one best-so-far value after each generation."""

from dataclasses import asdict

import numpy as np

from . import config, policies
from .profiles import Profile
from .search import SequentialSearch

# Reuses the repository monitor. Does not import gwo_de_dtw.
from dtw_stagnation import StagnationConfig, StagnationMonitor


class ProfileController:
    def __init__(self, monitor_cfg: StagnationConfig, exploit: Profile, explore: Profile,
                 policy: str = config.POLICY, scale: float = config.SCALE,
                 slope: float = config.K, center: float = config.CENTER,
                 dtw_kirito: bool = config.DTW_KIRITO):
        policies.check_policy(policy)
        self.cfg = monitor_cfg
        self.exploit = exploit
        self.explore = explore
        self.policy = policy
        self.dtw_kirito = dtw_kirito
        self.scale = scale
        self.slope = slope
        self.center = center
        self.profile = exploit
        self.intensity = None if policy in policies.BINARY_POLICIES else 0.0
        self.monitor = StagnationMonitor(cfg=monitor_cfg)

    def _next_binary(self, observation: dict) -> Profile:
        triggered = policies.enters_explore(self.policy, observation)
        if self.dtw_kirito:
            return self.explore if triggered else self.exploit
        if self.profile is self.exploit:
            return self.explore if triggered else self.exploit
        if observation["no_improve_len"] < self.cfg.plateau_max:
            return self.exploit
        return self.explore

    def observe(self, best_fitness: float) -> dict:
        observation = dict(self.monitor.update(best_fitness))
        changed = False
        if observation.get("ready") and self.intensity is None:
            nxt = self._next_binary(observation)
            if nxt is not self.profile:
                self.profile = nxt
                changed = True
                if not self.dtw_kirito:
                    self.monitor = StagnationMonitor(cfg=self.cfg)
        elif observation.get("ready"):
            self.intensity = policies.intensity(
                self.policy, observation, self.scale, self.slope, self.center)
            nxt = policies.blend(self.exploit, self.explore, self.intensity)
            changed = nxt.name != self.profile.name
            self.profile = nxt
        observation["changed"] = changed
        observation["profile"] = self.profile.name
        observation["intensity"] = self.intensity
        return observation


def run_search(inst, monitor_cfg, pop_size=config.POP_SIZE, iterations=config.ITERATIONS,
               seed=config.SEED, G1=config.G1, G2=config.G2,
               g3_exploit=config.EXPLOIT_G3, g3_explore=config.EXPLORE_G3,
               binarization=config.BINARIZATION, policy=config.POLICY,
               dtw_kirito=config.DTW_KIRITO) -> dict:
    """Run ``iterations`` generations. The monitor is updated once after both sweeps."""
    exploit = Profile("exploit", config.EXPLOIT_A, config.EXPLOIT_F,
                      config.EXPLOIT_CR, g3_exploit)
    explore = Profile("explore", config.EXPLORE_A, config.EXPLORE_F,
                      config.EXPLORE_CR, g3_explore)
    controller = ProfileController(monitor_cfg, exploit, explore, policy=policy,
                                   dtw_kirito=dtw_kirito)
    search = SequentialSearch(inst, np.random.default_rng(seed), pop_size=pop_size,
                              G1=G1, G2=G2, binarization=binarization)
    search.initialize()

    best_series = []
    profile_series = []
    intensity_series = []
    monitor_hist = []
    step_stats = []
    changes = []
    for generation in range(iterations):
        search.set_profile(controller.profile)
        profile_series.append(controller.profile.name)
        intensity_series.append(controller.intensity)
        best = float(search.step())
        best_series.append(best)
        step_stats.append(dict(search.last_step_stats))
        observation = controller.observe(best)
        monitor_hist.append(observation)
        if observation["changed"]:
            changes.append(generation)

    optimum = float(inst.get("optimo") or 0.0)
    fitness = float(search.best_fitness)
    return {
        "policy": policy,
        "schedule": config.SCHEDULE,
        "dtw_kirito": bool(dtw_kirito),
        "monitor_config": asdict(monitor_cfg),
        "seed": seed,
        "binarization": search.binarization,
        "pop_size": pop_size,
        "iterations": iterations,
        "evaluations": search.evaluations,
        "generations": len(best_series),
        "best_fitness": fitness,
        "best_solution": None if search.best_solution is None else search.best_solution.copy(),
        "known_optimum": optimum,
        "gap_pct": (100.0 * (optimum - fitness) / optimum) if optimum > 0 else None,
        "best_series": best_series,
        "profile_series": profile_series,
        "intensity_series": intensity_series,
        "changes": changes,
        "monitor": monitor_hist,
        "step_stats": step_stats,
    }
