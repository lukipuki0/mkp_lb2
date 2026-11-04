"""
DTW/DDTW profile controller and run loop for the GWO-DE hybrid.

The controller watches the best-so-far fitness (last W generations) and sets
the joint GWO/DE profile with one of the policies of policies.py. The two
vanilla policies ignore that signal and keep their initial profile.

Binary policies switch between the two fixed profiles:
    exploit + trigger                     -> explore
        (binary_complex: fire; binary_simple: D2_vs_const <= theta_c)
    explore + recent improvement          -> exploit
        (no_improve_len < plateau_max on a ready observation)
    explore + still stagnant              -> stay in explore
After every profile change the monitor is replaced (fresh window): window,
no_improve_len, trigger_streak and threshold histories are discarded, so no
decision until W new observations generated with the new profile exist.

With dtw_kirito=True (config.DTW_KIRITO) binary policies follow the
Kirito rolling control instead: the monitor is never replaced and every ready
observation sets explore if the trigger holds, exploit otherwise.

Continuous policies never replace the monitor. On every ready observation
they compute an intensity in [0, 1] and use the blend of exploit (0) and
explore (1); intensity is 0 before the first ready observation.

Global best and population are never reset. A change decided after
generation g applies from generation g+1.

The run stops after `iterations` generations (default config.ITERATIONS).

Spec: odd/tasks/gwo-de-dtw-hybrid.md (S5-S8, S10, S12, S14, S15, S20).
"""

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from . import config, policies
from dtw_stagnation import StagnationConfig, StagnationMonitor

from .hybrid import EXPLOIT, EXPLORE, GWODEHybrid, Profile


class ProfileController:
    def __init__(self, cfg: StagnationConfig,
                 exploit: Profile = EXPLOIT, explore: Profile = EXPLORE,
                 policy: str = config.POLICY, scale: float = config.SCALE,
                 k: float = config.K, center: float = config.CENTER,
                 dtw_kirito: bool = config.DTW_KIRITO):
        policies.check_policy(policy)
        self.cfg = cfg
        self.exploit = exploit
        self.explore = explore
        self.policy = policy
        self.dtw_kirito = dtw_kirito  # binary policies only
        self.scale, self.k, self.center = scale, k, center
        # Vanilla baselines never leave their mode. The adaptive policies
        # start in exploitation, as before.
        self.profile: Profile = explore if policy == "vanilla_exploration" else exploit
        # Intensity of the active profile; None when the profile is not a blend.
        fixed = policy in policies.BINARY_POLICIES or policy in policies.VANILLA_POLICIES
        self.intensity = None if fixed else 0.0
        self.monitor = StagnationMonitor(cfg=cfg)

    def _next_profile(self, out: Dict) -> Profile:
        if self.dtw_kirito:
            return self.explore if policies.enters_explore(self.policy, out) else self.exploit
        if self.profile is self.exploit:
            return self.explore if policies.enters_explore(self.policy, out) else self.exploit
        return self.exploit if out["no_improve_len"] < self.cfg.plateau_max else self.explore

    def observe(self, best_fitness: float) -> Dict:
        """Feed one best-so-far observation; returns the monitor output plus the decision."""
        out = dict(self.monitor.update(best_fitness))
        changed = False
        if self.policy in policies.VANILLA_POLICIES:
            pass
        elif out["ready"] and self.intensity is None:
            new = self._next_profile(out)
            if new is not self.profile:
                self.profile = new
                changed = True
                if not self.dtw_kirito:
                    self.monitor = StagnationMonitor(cfg=self.cfg)
        elif out["ready"]:
            self.intensity = policies.intensity(self.policy, out, self.scale,
                                                self.k, self.center)
            new = policies.blend(self.exploit, self.explore, self.intensity)
            changed = new.name != self.profile.name
            self.profile = new
        out["changed"] = changed
        out["profile"] = self.profile.name
        out["intensity"] = self.intensity
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
    policy: str = config.POLICY,
    dtw_kirito: bool = config.DTW_KIRITO,
    hybrid_mode: str = config.HYBRID_MODE,
) -> dict:
    """One run of the hybrid: exactly `iterations` generations.

    g3_exploit and g3_explore build the two profiles for this run (a, F and CR
    stay the config values). The same value in both is a fixed G3. `policy` is
    one of policies.POLICIES; SCALE, K and CENTER come from config. dtw_kirito
    selects the binary control (see ProfileController) and is recorded.
    hybrid_mode ("competitive" or "sequential", see hybrid.py) is recorded;
    the monitor observes once per generation, after both operators, and the
    profile is the same for the whole generation in both modes.
    """
    exploit = Profile("exploit", a=config.EXPLOIT_A, F=config.EXPLOIT_F,
                      CR=config.EXPLOIT_CR, G3=g3_exploit)
    explore = Profile("explore", a=config.EXPLORE_A, F=config.EXPLORE_F,
                      CR=config.EXPLORE_CR, G3=g3_explore)
    controller = ProfileController(monitor_cfg, exploit=exploit, explore=explore,
                                   policy=policy, dtw_kirito=dtw_kirito)
    hybrid = GWODEHybrid(inst, np.random.default_rng(seed),
                         pop_size=pop_size, G1=G1, G2=G2,
                         binarization=binarization, hybrid_mode=hybrid_mode)
    hybrid.initialize()

    best_series: List[float] = []
    profile_series: List[str] = []
    # G3 is the active profile's LB2 flip probability. Sigmoid does not use it,
    # but it is still recorded so the series follows profile_series.
    parameter_series: Dict[str, List[float]] = {"a": [], "F": [], "CR": [], "G3": []}
    # Intensity of the profile used at each generation; None for binary policies.
    intensity_series: List[Optional[float]] = []
    monitor_hist: List[Dict] = []
    step_stats: List[Dict] = []
    changes: List[int] = []

    for generation in range(iterations):
        hybrid.set_profile(controller.profile)
        profile_series.append(controller.profile.name)
        for key in parameter_series:
            parameter_series[key].append(getattr(controller.profile, key))
        intensity_series.append(controller.intensity)
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
        "policy": policy,
        "dtw_kirito": dtw_kirito,
        "monitor_config": asdict(monitor_cfg),
        "use_ddtw": monitor_cfg.use_ddtw,
        "window": monitor_cfg.window,
        "seed": seed,
        "binarization": hybrid.binarization,
        "hybrid_mode": hybrid.hybrid_mode,
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
        "intensity_series": intensity_series,
        "changes": changes,
        "monitor": monitor_hist,
        "step_stats": step_stats,
    }
