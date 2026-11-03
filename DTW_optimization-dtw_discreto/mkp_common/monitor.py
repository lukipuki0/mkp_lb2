"""
Monitor de estancamiento basado en DTW.
Implementación fiel al notebook (celda 13).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def dtw_distance(s: np.ndarray, t: np.ndarray, window: Optional[int] = None) -> float:
    """Distancia DTW con banda Sakoe-Chiba."""
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    n, m = len(s), len(t)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        si = s[i - 1]
        for j in range(j_start, j_end + 1):
            cost = abs(si - t[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def first_diff(x: np.ndarray) -> np.ndarray:
    """Primera diferencia de una secuencia (para DDTW)."""
    x = np.asarray(x, dtype=float)
    return np.diff(x, prepend=x[0])


def ddtw_distance(s: np.ndarray, t: np.ndarray, window: Optional[int] = None) -> float:
    """Derivative DTW: DTW sobre primeras diferencias."""
    return dtw_distance(first_diff(s), first_diff(t), window=window)


def ramp_baseline(start_value: float, s_min: float, length: int) -> np.ndarray:
    """Genera una rampa lineal ideal."""
    i = np.arange(length, dtype=float)
    return start_value + s_min * i


def constant_baseline(start_value: float, length: int) -> np.ndarray:
    """Genera una linea constante (meseta ideal)."""
    return np.full(length, float(start_value))


def moving_percentile(buffer: List[float], p: float) -> float:
    """Percentil movil sobre un buffer historico."""
    arr = np.asarray(buffer, dtype=float)
    return float(np.percentile(arr, p)) if arr.size > 0 else float("nan")


@dataclass
class StagnationConfig:
    """Configuración del monitor de estancamiento (valores del notebook)."""

    window: int = 30
    band: int = 0
    min_slope: float = 0.0
    plateau_max: int = 15
    patience: int = 3
    use_ddtw: bool = False
    adapt_thresholds: bool = True
    p_low: float = 30.0
    p_high: float = 70.0

    def __post_init__(self):
        if self.band <= 0:
            self.band = max(1, int(0.1 * self.window))

    def to_dict(self) -> dict:
        """Retorna todos los parámetros configurados del monitor para persistencia."""
        return {
            "dtw_window": int(self.window),
            "dtw_band": int(self.band),
            "dtw_min_slope": float(self.min_slope),
            "dtw_plateau_max": int(self.plateau_max),
            "dtw_patience": int(self.patience),
            "dtw_use_ddtw": bool(self.use_ddtw),
            "dtw_adapt_thresholds": bool(self.adapt_thresholds),
            "dtw_p_low": float(self.p_low),
            "dtw_p_high": float(self.p_high),
        }


@dataclass
class StagnationMonitor:
    """
    Monitor de estancamiento basado en DTW.

    Lógica de decisión (3 condiciones simultáneas):
        1. cond_plateau:  sin mejora durante >= plateau_max iteraciones
        2. cond_constant: D2 <= theta_c  (cerca de la meseta)
        3. cond_ramp:     D1 >= theta_r OR delta >= theta_delta  (lejos de la rampa)

    fire = True cuando trigger_streak >= patience.
    """

    cfg: StagnationConfig
    best_so_far: List[float] = field(default_factory=list)
    no_improve_len: int = 0
    trigger_streak: int = 0
    d1_hist: List[float] = field(default_factory=list)
    d2_hist: List[float] = field(default_factory=list)
    delta_hist: List[float] = field(default_factory=list)

    def update(self, new_best: float) -> Dict:
        """Actualiza el monitor con el mejor fitness de la iteración actual."""
        if isinstance(new_best, (list, tuple, np.ndarray)):
            arr = np.asarray(new_best, dtype=float)
            new_best = float(np.max(arr))
        else:
            new_best = float(new_best)

        if self.best_so_far and isinstance(
            self.best_so_far[-1], (list, tuple, np.ndarray)
        ):
            self.best_so_far = [
                float(np.max(np.asarray(v, dtype=float))) for v in self.best_so_far
            ]

        if self.best_so_far and new_best <= self.best_so_far[-1]:
            self.no_improve_len += 1
            new_best = self.best_so_far[-1]
        else:
            self.no_improve_len = 0
        self.best_so_far.append(float(new_best))

        n = len(self.best_so_far)
        W = self.cfg.window
        if n < W:
            # Keep the data window at W points, but expose finite placeholders
            # so strategy-specific decisions can be called during warm-up.
            theta_c = 0.1 * W
            theta_r = 0.5 * W
            theta_delta = 0.3 * W
            return {
                "ready": False,
                "fire": False,
                "D1_vs_ramp": 0.0,
                "D2_vs_const": 0.0,
                "delta": 0.0,
                "theta_c": float(theta_c),
                "theta_r": float(theta_r),
                "theta_delta": float(theta_delta),
                "no_improve_len": self.no_improve_len, "n": n,
                "trigger_streak": 0,
            }

        X = np.array(self.best_so_far[-W:], dtype=float)
        start = X[0]

        s_min = self.cfg.min_slope
        if s_min == 0.0:
            rng = max(1.0, abs(X[-1] - X[0]))
            s_min = 0.01 * rng / W

        r = ramp_baseline(start, s_min, W)
        c = constant_baseline(start, W)

        if self.cfg.use_ddtw:
            D1 = ddtw_distance(X, r, window=self.cfg.band)
            D2 = ddtw_distance(X, c, window=self.cfg.band)
        else:
            D1 = dtw_distance(X, r, window=self.cfg.band)
            D2 = dtw_distance(X, c, window=self.cfg.band)

        delta = D1 - D2
        self.d1_hist.append(D1)
        self.d2_hist.append(D2)
        self.delta_hist.append(delta)

        if self.cfg.adapt_thresholds and len(self.d1_hist) >= 10:
            theta_c = moving_percentile(self.d2_hist, self.cfg.p_low)
            theta_r = moving_percentile(self.d1_hist, self.cfg.p_high)
            theta_delta = moving_percentile(self.delta_hist, self.cfg.p_high)
        else:
            theta_c = 0.1 * W
            theta_r = 0.5 * W
            theta_delta = 0.3 * W

        cond_plateau = self.no_improve_len >= self.cfg.plateau_max
        cond_constant = D2 <= theta_c
        cond_ramp = (D1 >= theta_r) or (delta >= theta_delta)

        if cond_plateau and cond_constant and cond_ramp:
            self.trigger_streak += 1
        else:
            self.trigger_streak = 0

        fire = self.trigger_streak >= self.cfg.patience

        return {
            "ready": True,
            "fire": bool(fire),
            "D1_vs_ramp": float(D1),
            "D2_vs_const": float(D2),
            "delta": float(delta),
            "theta_c": float(theta_c),
            "theta_r": float(theta_r),
            "theta_delta": float(theta_delta),
            "no_improve_len": int(self.no_improve_len),
            "trigger_streak": int(self.trigger_streak),
            "n": int(n),
        }
