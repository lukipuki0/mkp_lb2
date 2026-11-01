"""Four profile policies. Binary policies switch; continuous policies blend."""

import math

from .profiles import Profile

POLICIES = ("binary_simple", "binary_complex", "continuous_simple", "continuous_complex")
BINARY_POLICIES = POLICIES[:2]
EPS = 1e-10


def check_policy(policy: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")


def enters_explore(policy: str, observation: dict) -> bool:
    if policy == "binary_simple":
        return observation["D2_vs_const"] <= observation["theta_c"]
    return bool(observation["fire"])


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def intensity(policy: str, observation: dict, scale: float, slope: float, center: float) -> float:
    """Exploration intensity in [0, 1]. A negative theta_delta is used as given."""
    if policy == "continuous_simple":
        ratio = observation["D2_vs_const"] / (observation["theta_c"] * scale + EPS)
        return 1.0 - min(max(ratio, 0.0), 1.0)
    ratio = observation["delta"] / (observation["theta_delta"] + EPS)
    return _sigmoid(slope * (ratio - center))


def blend(exploit: Profile, explore: Profile, value: float) -> Profile:
    def mix(field: str) -> float:
        return (1.0 - value) * getattr(exploit, field) + value * getattr(explore, field)

    name = explore.name if value > 0.5 else exploit.name
    return Profile(name, a=mix("a"), F=mix("F"), CR=mix("CR"), G3=mix("G3"))
