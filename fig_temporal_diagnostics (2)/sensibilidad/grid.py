"""Parameter grid and configuration generation for the OAT campaign."""

from __future__ import annotations

from typing import Any, Dict, List


PAPER_BASE = dict(
    window=200,
    band=2,
    min_slope=2.0,
    p_low=40.0,
    p_high=60.0,
    plateau_max=5,
    patience=3,
)

GRID = {
    # Scope: only the parameters explicitly cited by Reviewer 1 (R1.4):
    # W (window), Sakoe-Chiba band, percentiles (p_low/p_high),
    # tau_pat (patience) and pi_max (plateau_max).
    # min_slope stays fixed at its paper value (see PAPER_BASE).
    "window": [50, 100, 200, 400],
    "band": [1, 2, 4, 8],
    "p_low": [20.0, 30.0, 40.0, 50.0],
    "p_high": [50.0, 60.0, 70.0, 80.0],
    "plateau_max": [3, 5, 8, 12],
    "patience": [1, 2, 3, 5],
}

ENV_MAP = {
    "window": "MKP_WINDOW",
    "band": "MKP_BAND",
    "min_slope": "MKP_MIN_SLOPE",
"p_low": "MKP_P_LOW",
    "p_high": "MKP_P_HIGH",
    "plateau_max": "MKP_PLATEAU_MAX",
    "patience": "MKP_PATIENCE",
}


def validate_oat_configs(configs: List[Dict[str, Any]]) -> None:
    """Validate the invariant shape of the generated OAT configuration list."""
    expected = 1 + sum(
        len([value for value in values if value != PAPER_BASE[param]])
        for param, values in GRID.items()
    )
    if len(configs) != expected:
        raise ValueError(f"Expected {expected} OAT configurations, got {len(configs)}")
    config_ids = [config["config_id"] for config in configs]
    if len(set(config_ids)) != len(config_ids):
        raise ValueError("OAT configuration IDs must be unique")
    if configs[0] != {
        "config_id": "base",
        "param": None,
        "value": None,
        "overrides": {},
    }:
        raise ValueError("The base configuration must be the first OAT entry")


def build_oat_configs() -> List[Dict[str, Any]]:
    """Return the paper base and all one-factor off-base configurations."""
    configs: List[Dict[str, Any]] = [
        {
            "config_id": "base",
            "param": None,
            "value": None,
            "overrides": {},
        }
    ]

    for param, values in GRID.items():
        for value in values:
            if value == PAPER_BASE[param]:
                continue
            configs.append(
                {
                    "config_id": f"{param}={value}",
                    "param": param,
                    "value": value,
                    "overrides": {param: value},
                }
            )

    validate_oat_configs(configs)
    return configs


if __name__ == "__main__":
    # A tiny executable validation keeps the invariant easy to check without
    # starting a campaign.
    configs = build_oat_configs()
    validate_oat_configs(configs)
    print(f"OAT grid OK: {len(configs)} configurations")
