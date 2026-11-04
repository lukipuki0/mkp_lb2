"""
The four profile policies, named after the DTW_optimization-main folders,
plus two fixed baselines that never read the monitor decision.

The adaptive policies read the same monitor output
(dtw_stagnation.StagnationMonitor) and use the same exploit/explore profiles:
  - binary_simple      exploit -> explore when D2_vs_const <= theta_c
  - binary_complex     exploit -> explore when the monitor fires
    Both return to exploit with the S8 rule and replace the monitor after
    every change (fresh window); see controller.ProfileController.
  - continuous_simple  intensity = 1 - clip(D2_vs_const / (theta_c * scale + eps), 0, 1)
  - continuous_complex intensity = sigmoid(k * (delta / (theta_delta + eps) - center))
    The active profile is blend(exploit, explore, intensity); the monitor is
    never replaced.
  - vanilla_exploitation  exploit for every generation
  - vanilla_exploration   explore for every generation
    Neither changes mode, whatever the monitor reports.

Spec: odd/tasks/gwo-de-dtw-hybrid.md (S20).
"""

import math
from typing import Dict

from .hybrid import Profile

POLICIES = ("binary_simple", "binary_complex", "continuous_simple", "continuous_complex")
BINARY_POLICIES = POLICIES[:2]
CONTINUOUS_POLICIES = POLICIES[2:]
# Fixed baselines. Not part of POLICIES. run.py and campaign.py iterate
# CAMPAIGN_POLICIES, which appends these two.
VANILLA_POLICIES = ("vanilla_exploitation", "vanilla_exploration")
CAMPAIGN_POLICIES = POLICIES + VANILLA_POLICIES
ACCEPTED_POLICIES = CAMPAIGN_POLICIES

EPS = 1e-10


def check_policy(policy: str) -> None:
    if policy not in ACCEPTED_POLICIES:
        raise ValueError(f"policy must be one of {ACCEPTED_POLICIES}, got {policy!r}")


def enters_explore(policy: str, out: Dict) -> bool:
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


def intensity(policy: str, out: Dict, scale: float, k: float, center: float) -> float:
    """Exploration intensity in [0, 1] of a continuous policy on a ready observation.

    A negative theta_delta is used as is (same formula as DTW_optimization-main).
    """
    if policy == "continuous_simple":
        ratio = out["D2_vs_const"] / (out["theta_c"] * scale + EPS)
        return 1.0 - min(max(ratio, 0.0), 1.0)
    ratio = out["delta"] / (out["theta_delta"] + EPS)
    return _sigmoid(k * (ratio - center))


def blend(exploit: Profile, explore: Profile, value: float) -> Profile:
    """Linear blend of a, F, CR and G3; named explore when value > 0.5."""
    def mix(key: str) -> float:
        return (1.0 - value) * getattr(exploit, key) + value * getattr(explore, key)

    name = explore.name if value > 0.5 else exploit.name
    return Profile(name, a=mix("a"), F=mix("F"), CR=mix("CR"), G3=mix("G3"))
