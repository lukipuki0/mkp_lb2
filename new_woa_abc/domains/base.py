"""Contrato mínimo que consume el motor WOA--ABC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str
    dimension: int
    lower: np.ndarray
    upper: np.ndarray
    objective: Callable[[np.ndarray], float]
    optimum: float | None = None
    decode_fn: Callable[[np.ndarray], dict[str, Any]] | None = None
    metrics_fn: Callable[[np.ndarray], dict[str, Any]] | None = None
    repair_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.shape != (self.dimension,) or upper.shape != (self.dimension,):
            raise ValueError("lower y upper deben tener una entrada por dimensión")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("los límites deben ser finitos")
        if np.any(lower >= upper):
            raise ValueError("cada límite inferior debe ser menor que el superior")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def evaluate(self, position: np.ndarray) -> float:
        value = float(self.objective(self.repair(position)))
        return value if np.isfinite(value) else float("inf")

    def repair(self, position: np.ndarray) -> np.ndarray:
        value = np.asarray(position, dtype=float).copy()
        value = np.clip(value, self.lower, self.upper)
        if self.repair_fn is not None:
            value = np.asarray(self.repair_fn(value), dtype=float)
        return np.clip(value, self.lower, self.upper)

    def decode(self, position: np.ndarray) -> dict[str, Any]:
        return {} if self.decode_fn is None else dict(self.decode_fn(self.repair(position)))

    def metrics(self, position: np.ndarray) -> dict[str, Any]:
        return {} if self.metrics_fn is None else dict(self.metrics_fn(self.repair(position)))
