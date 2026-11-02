"""Exploit/explore profiles and the four stagnation policies of WOA-ABC.

The reaction mirrors ``gwo_de_dtw`` (``config.py``, ``policies.py`` and
``controller.py`` on ``feat/gwo-de-dtw-hybrid``). A profile fixes the WOA
``a``, the ABC step amplitude ``rho`` (phi ~ U[-rho, rho]) and the LB2 base
flip probability ``G3``. Within a profile ``a`` is constant.

Binary policies switch between the two fixed profiles:
    exploit + trigger                     -> explore
        (binary_complex: fire; binary_simple: D2_vs_const <= theta_c)
    explore + recent improvement          -> exploit
        (no_improve_len < plateau_max on a ready observation)
    explore + still stagnant              -> stay in explore
After every profile change the monitor is replaced (fresh window).

With ``dtw_kirito=True`` binary policies follow the Kirito rolling control:
the monitor is never replaced and every ready observation sets explore if the
trigger holds, exploit otherwise.

Continuous policies never replace the monitor. On every ready observation
they compute an intensity in [0, 1] and use the blend of exploit (0) and
explore (1); intensity is 0 before the first ready observation.

The values below are provisional and live only here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from dtw_stagnation import StagnationConfig, StagnationMonitor


# Profiles. Provisional values, to be discussed.
EXPLOIT_A, EXPLOIT_RHO, EXPLOIT_G3 = 0.5, 1.0, 0.5
EXPLORE_A, EXPLORE_RHO, EXPLORE_G3 = 2.0, 2.0, 0.5

# Policy defaults, as gwo_de_dtw/config.py.
POLICY = "binary_complex"
SCALE = 2.0           # continuous_simple: D2 / (theta_c * SCALE)
K = 5.0               # continuous_complex: sigmoid slope
CENTER = 0.5          # continuous_complex: delta / theta_delta at intensity 0.5
DTW_KIRITO = False    # binary policies only

POLICIES = ("binary_simple", "binary_complex", "continuous_simple", "continuous_complex")
BINARY_POLICIES = POLICIES[:2]
CONTINUOUS_POLICIES = POLICIES[2:]
POLICY_ERROR = "policy must be one of " + ", ".join(POLICIES)

EPS = 1e-10


@dataclass(frozen=True)
class Profile:
    """Joint WOA/ABC parameter profile."""

    name: str
    a: float
    rho: float
    G3: float


EXPLOIT = Profile("exploit", a=EXPLOIT_A, rho=EXPLOIT_RHO, G3=EXPLOIT_G3)
EXPLORE = Profile("explore", a=EXPLORE_A, rho=EXPLORE_RHO, G3=EXPLORE_G3)


def check_policy(policy: object) -> str:
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError(POLICY_ERROR)
    return policy


def enters_explore(policy: str, out: dict) -> bool:
    """Exploit -> explore trigger of a binary policy on a ready observation."""

    if policy == "binary_simple":
        return out["D2_vs_const"] <= out["theta_c"]
    return bool(out["fire"])


def _sigmoid(z: float) -> float:
    # Same value as 1 / (1 + exp(-z)) without overflowing for large |z|.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def intensity(policy: str, out: dict, scale: float = SCALE, k: float = K, center: float = CENTER) -> float:
    """Exploration intensity in [0, 1] of a continuous policy on a ready observation."""

    if policy == "continuous_simple":
        ratio = out["D2_vs_const"] / (out["theta_c"] * scale + EPS)
        return 1.0 - min(max(ratio, 0.0), 1.0)
    ratio = out["delta"] / (out["theta_delta"] + EPS)
    return _sigmoid(k * (ratio - center))


def blend(exploit: Profile, explore: Profile, value: float) -> Profile:
    """Linear blend of a, rho and G3; named explore when value > 0.5."""

    def mix(key: str) -> float:
        return (1.0 - value) * getattr(exploit, key) + value * getattr(explore, key)

    name = explore.name if value > 0.5 else exploit.name
    return Profile(name, a=mix("a"), rho=mix("rho"), G3=mix("G3"))


def initial_intensity(policy: str) -> float | None:
    """None for binary policies, 0 for continuous ones."""

    return None if policy in BINARY_POLICIES else 0.0


class ProfileController:
    """Feeds the archived best to the monitor and decides the next profile.

    ``monitor_factory`` builds every monitor, including the replacements made
    after a binary change. It defaults to ``StagnationMonitor(cfg=cfg)``.
    """

    def __init__(
        self,
        cfg: StagnationConfig,
        exploit: Profile = EXPLOIT,
        explore: Profile = EXPLORE,
        policy: str = POLICY,
        scale: float = SCALE,
        k: float = K,
        center: float = CENTER,
        dtw_kirito: bool = DTW_KIRITO,
        monitor_factory: Callable[[], StagnationMonitor] | None = None,
    ) -> None:
        check_policy(policy)
        self.cfg = cfg
        self.exploit = exploit
        self.explore = explore
        self.policy = policy
        self.dtw_kirito = dtw_kirito
        self.scale, self.k, self.center = scale, k, center
        self.profile = exploit
        self.intensity = initial_intensity(policy)
        self._new_monitor = monitor_factory or (lambda: StagnationMonitor(cfg=cfg))
        self.monitor = self._new_monitor()

    def _next_profile(self, out: dict) -> Profile:
        if self.dtw_kirito:
            return self.explore if enters_explore(self.policy, out) else self.exploit
        if self.profile is self.exploit:
            return self.explore if enters_explore(self.policy, out) else self.exploit
        return self.exploit if out["no_improve_len"] < self.cfg.plateau_max else self.explore

    def observe(self, best: float) -> dict:
        """Feed one archived best; returns the monitor output plus the decision."""

        out = dict(self.monitor.update(best))
        changed = False
        if out["ready"] and self.intensity is None:
            new = self._next_profile(out)
            if new is not self.profile:
                self.profile = new
                changed = True
                if not self.dtw_kirito:
                    self.monitor = self._new_monitor()
        elif out["ready"]:
            self.intensity = intensity(self.policy, out, self.scale, self.k, self.center)
            new = blend(self.exploit, self.explore, self.intensity)
            changed = new.name != self.profile.name
            self.profile = new
        out["changed"] = changed
        out["profile"] = self.profile.name
        out["intensity"] = self.intensity
        return out
