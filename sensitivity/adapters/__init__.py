"""Public single-run adapter API for the original rotational framework."""

import importlib
import math

from dtw_stagnation import StagnationConfig


def run_case(domain: str, case: dict, monitor: dict, seed: int, budget: int,
             max_epoch_iters: int | None = None) -> dict:
    """Run one local case; explicit smoke caps never change legacy defaults."""
    if domain not in ("mkp", "cec", "hres"):
        raise ValueError("domain must be mkp, cec, or hres")
    if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip():
        raise ValueError("case must contain a nonempty id")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in NumPy's 32-bit seed range")
    for name, value in (("budget", budget), ("max_epoch_iters", max_epoch_iters)):
        if value is None and name == "max_epoch_iters":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    cfg = StagnationConfig(**monitor)
    for name in ("window", "band", "plateau_max", "patience"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Monitor {name} must be a positive integer")
    for name in ("min_slope", "improvement_tol"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Monitor {name} must be finite and nonnegative")
    if not 0 <= cfg.p_low < cfg.p_high <= 100:
        raise ValueError("Monitor percentiles must satisfy 0 <= p_low < p_high <= 100")
    if not isinstance(cfg.use_ddtw, bool) or not isinstance(cfg.adapt_thresholds, bool):
        raise ValueError("Monitor switches must be boolean")
    module = importlib.import_module(f"{__name__}.{domain}")
    return module.run_case(case, cfg, seed, budget, max_epoch_iters)
