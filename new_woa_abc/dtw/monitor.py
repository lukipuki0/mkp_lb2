"""Monitor DTW/DDTW local, basado en el repositorio de referencia.

El monitor opera sobre fitness de maximización. El motor WOA--ABC minimiza y
por eso le entrega ``-best_cost``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from new_woa_abc.config import DTWConfig


def dtw_distance(
    source: np.ndarray,
    target: np.ndarray,
    window: int | None = None,
) -> float:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    n, m = len(source), len(target)
    band = max(n, m) if window is None else max(int(window), abs(n - m))
    distances = np.full((n + 1, m + 1), float("inf"))
    distances[0, 0] = 0.0
    for i in range(1, n + 1):
        start = max(1, i - band)
        stop = min(m, i + band)
        for j in range(start, stop + 1):
            cost = abs(source[i - 1] - target[j - 1])
            distances[i, j] = cost + min(
                distances[i - 1, j],
                distances[i, j - 1],
                distances[i - 1, j - 1],
            )
    return float(distances[n, m])


def first_difference(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.diff(values, prepend=values[0])


def ddtw_distance(
    source: np.ndarray,
    target: np.ndarray,
    window: int | None = None,
) -> float:
    return dtw_distance(
        first_difference(source),
        first_difference(target),
        window=window,
    )


@dataclass
class StagnationMonitor:
    """Calcula D1, D2, delta y el fire A4 de tres condiciones."""

    config: DTWConfig
    best_so_far: list[float] = field(default_factory=list)
    no_improve_len: int = 0
    trigger_streak: int = 0
    d1_history: list[float] = field(default_factory=list)
    d2_history: list[float] = field(default_factory=list)
    delta_history: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.best_so_far.clear()
        self.no_improve_len = 0
        self.trigger_streak = 0
        self.d1_history.clear()
        self.d2_history.clear()
        self.delta_history.clear()

    def update(self, new_best: float) -> dict[str, Any]:
        value = float(new_best)
        if self.best_so_far and value <= self.best_so_far[-1]:
            self.no_improve_len += 1
            value = self.best_so_far[-1]
        else:
            self.no_improve_len = 0
        self.best_so_far.append(value)

        size = len(self.best_so_far)
        window = self.config.window
        if size < window:
            return {
                "ready": False,
                "fire": False,
                "no_improve_len": self.no_improve_len,
                "trigger_streak": self.trigger_streak,
                "n": size,
            }

        observed = np.asarray(self.best_so_far[-window:], dtype=float)
        start = float(observed[0])
        slope = self.config.min_slope
        if slope == 0.0:
            observed_range = max(1.0, abs(float(observed[-1] - observed[0])))
            slope = 0.01 * observed_range / window

        ramp = start + slope * np.arange(window, dtype=float)
        constant = np.full(window, start, dtype=float)
        distance = ddtw_distance if self.config.use_ddtw else dtw_distance
        d1 = distance(observed, ramp, self.config.effective_band)
        d2 = distance(observed, constant, self.config.effective_band)
        delta = d1 - d2

        self.d1_history.append(d1)
        self.d2_history.append(d2)
        self.delta_history.append(delta)

        if self.config.adapt_thresholds and len(self.d1_history) >= 10:
            theta_c = float(np.percentile(self.d2_history, self.config.p_low))
            theta_r = float(np.percentile(self.d1_history, self.config.p_high))
            theta_delta = float(np.percentile(self.delta_history, self.config.p_high))
        else:
            theta_c = 0.1 * window
            theta_r = 0.5 * window
            theta_delta = 0.3 * window

        cond_plateau = self.no_improve_len >= self.config.plateau_max
        cond_constant = d2 <= theta_c
        cond_ramp = d1 >= theta_r or delta >= theta_delta
        if cond_plateau and cond_constant and cond_ramp:
            self.trigger_streak += 1
        else:
            self.trigger_streak = 0

        return {
            "ready": True,
            "fire": self.trigger_streak >= self.config.patience,
            "D1_vs_ramp": float(d1),
            "D2_vs_const": float(d2),
            "delta": float(delta),
            "theta_c": theta_c,
            "theta_r": theta_r,
            "theta_delta": theta_delta,
            "cond_plateau": bool(cond_plateau),
            "cond_constant": bool(cond_constant),
            "cond_ramp": bool(cond_ramp),
            "no_improve_len": self.no_improve_len,
            "trigger_streak": self.trigger_streak,
            "n": size,
        }


__all__ = ["StagnationMonitor", "ddtw_distance", "dtw_distance"]
