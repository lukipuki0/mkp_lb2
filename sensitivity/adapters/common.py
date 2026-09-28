"""Shared offline adapter utilities; no experiment is run on import."""

from dataclasses import asdict, replace
import importlib.util
import math
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def local_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError("Case data must be a local repository-relative path")
    supplied = Path(value)
    path = (ROOT / supplied).resolve()
    if supplied.is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError("Case data must remain inside the repository")
    if not path.is_file():
        raise FileNotFoundError(f"Required local case data missing: {path}")
    return path


def load_hres_module(filename: str):
    """Load hyphenated HRES directory under an adapter-specific module name."""
    name = "_sensitivity_hres_" + Path(filename).stem
    if name not in sys.modules:
        path = ROOT / "HRES2-H2" / filename
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def effective_monitor(cfg, solvers, patience_overrides=()):
    return {
        solver: asdict(replace(cfg, patience=max(cfg.patience, 14)))
        if solver in patience_overrides else asdict(cfg)
        for solver in solvers
    }


def result_record(result, *, metric_name, metric, feasible, budget,
                  elapsed_seconds, effective, **details):
    objective = float(result.mejor_valor_global)
    if not math.isfinite(objective):
        raise ValueError("Solver returned a nonfinite objective")
    actual_iters = len(result.historial_global)
    n_epochs = len(result.log_switches)
    return json_safe({
        "objective": objective,
        "metric_name": metric_name,
        "metric": metric,
        "feasible": bool(feasible),
        "actual_iters": actual_iters,
        "n_epochs": n_epochs,
        "n_switches": max(0, n_epochs - 1),
        "budget_overshoot": max(0, actual_iters - budget),
        "elapsed_seconds": elapsed_seconds,
        "effective_monitor": effective,
        "solution": result.mejor_solucion_global,
        **details,
    })
