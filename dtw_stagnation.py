"""
stagnation.py
─────────────
Monitor de estancamiento basado en DTW (Dynamic Time Warping) para SA-MKP.

Extraído del notebook Advance_of_LB2_para_MKP y adaptado como módulo
independiente.  Detecta si la curva de mejor-valor ha dejado de progresar
comparándola con una rampa ascendente (progresando) y una línea constante
(estancado) usando distancia DTW o Derivada-DTW (DDTW).

Clases públicas
───────────────
  StagnationConfig   – hiperparámetros del monitor.
  StagnationMonitor  – estado acumulado + método update().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ── DTW puro ─────────────────────────────────────────────────────────────────

def dtw_distance(
    s: np.ndarray,
    t: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """Distancia DTW entre las series *s* y *t* con banda de Sakoe-Chiba."""
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
        j_end   = min(m, i + window)
        si = s[i - 1]
        for j in range(j_start, j_end + 1):
            cost = abs(si - t[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(D[n, m])


def _first_diff(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.diff(x, prepend=x[0])


def ddtw_distance(
    s: np.ndarray,
    t: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """DTW sobre la primera derivada de las series (DDTW)."""
    return dtw_distance(_first_diff(s), _first_diff(t), window=window)


# ── Baselines ─────────────────────────────────────────────────────────────────

def ramp_baseline(start_value: float, s_min: float, length: int) -> np.ndarray:
    """Serie que crece linealmente desde *start_value* con pendiente *s_min*."""
    i = np.arange(length, dtype=float)
    return start_value + s_min * i


def constant_baseline(start_value: float, length: int) -> np.ndarray:
    """Serie constante igual a *start_value*."""
    return np.full(length, float(start_value))


def moving_percentile(buffer: List[float], p: float) -> float:
    arr = np.asarray(buffer, dtype=float)
    return float(np.percentile(arr, p)) if arr.size > 0 else float("nan")


# ── Configuración ─────────────────────────────────────────────────────────────

@dataclass
class StagnationConfig:
    """Hiperparámetros del monitor de estancamiento.

    Atributos
    ---------
    window : int
        Tamaño de la ventana de historial analizada.
    band : int
        Banda Sakoe-Chiba para DTW (0 → 10% de *window*).
    min_slope : float
        Pendiente mínima de la rampa (0 → auto 1% del rango).
    plateau_max : int
        Iteraciones sin mejora para considerar meseta.
    patience : int
        Epochs consecutivos de trigger antes de disparar.
    use_ddtw : bool
        Usar DDTW (derivada) en lugar de DTW estándar.
    adapt_thresholds : bool
        Ajustar umbrales usando percentiles del historial.
    p_low / p_high : float
        Percentiles para los umbrales adaptativos.
    """

    window:           int   = 30
    band:             int   = 0        # 0 → auto
    min_slope:        float = 0.0
    plateau_max:      int   = 15
    patience:         int   = 3
    use_ddtw:         bool  = False
    adapt_thresholds: bool  = True
    p_low:            float = 30.0
    p_high:           float = 70.0

    def __post_init__(self) -> None:
        if self.band <= 0:
            self.band = max(1, int(0.1 * self.window))


# ── Monitor ───────────────────────────────────────────────────────────────────

@dataclass
class StagnationMonitor:
    """Monitor de estancamiento acumulativo.

    Uso
    ---
    >>> cfg = StagnationConfig(window=30, plateau_max=15, patience=3)
    >>> monitor = StagnationMonitor(cfg)
    >>> status = monitor.update(mejor_valor_actual)
    >>> if status["fire"]:
    ...     # reiniciar solución / perturbar temperatura
    """

    cfg:            StagnationConfig
    best_so_far:    List[float] = field(default_factory=list)
    no_improve_len: int         = 0
    trigger_streak: int         = 0
    d1_hist:        List[float] = field(default_factory=list)
    d2_hist:        List[float] = field(default_factory=list)
    delta_hist:     List[float] = field(default_factory=list)

    def reset(self) -> None:
        """Reinicia el estado del monitor (útil al inicio de cada epoch)."""
        self.best_so_far    = []
        self.no_improve_len = 0
        self.trigger_streak = 0
        self.d1_hist        = []
        self.d2_hist        = []
        self.delta_hist     = []

    def update(self, new_best: float) -> dict:
        """Actualiza el historial y devuelve el estado del detector.

        Parameters
        ----------
        new_best : float
            Mejor valor encontrado en la iteración actual.

        Returns
        -------
        dict con claves:
            ready           – True cuando hay suficientes datos (>= window).
            fire            – True si se detecta estancamiento sostenido.
            no_improve_len  – iteraciones consecutivas sin mejora.
            trigger_streak  – epochs consecutivos de condición cumplida.
            D1_vs_ramp      – distancia DTW a la rampa (solo si ready).
            D2_vs_const     – distancia DTW a la constante (solo si ready).
            delta           – D1 - D2 (solo si ready).
        """
        # ── Sanitizar entrada ─────────────────────────────────────────────
        if isinstance(new_best, (list, tuple, np.ndarray)):
            new_best = float(np.max(np.asarray(new_best, dtype=float)))
        else:
            new_best = float(new_best)

        # Reparar historial previo si algo se coló como array
        if self.best_so_far and isinstance(
            self.best_so_far[-1], (list, tuple, np.ndarray)
        ):
            self.best_so_far = [
                float(np.max(np.asarray(v, dtype=float)))
                for v in self.best_so_far
            ]

        # ── Actualizar historial ──────────────────────────────────────────
        if self.best_so_far and new_best <= self.best_so_far[-1]:
            self.no_improve_len += 1
            new_best = self.best_so_far[-1]
        else:
            self.no_improve_len = 0

        self.best_so_far.append(float(new_best))

        n = len(self.best_so_far)
        W = self.cfg.window

        if n < W:
            return {
                "ready": False,
                "fire": False,
                "no_improve_len": self.no_improve_len,
                "n": n,
            }

        # ── Calcular distancias DTW ───────────────────────────────────────
        X     = np.array(self.best_so_far[-W:], dtype=float)
        start = X[0]

        s_min = self.cfg.min_slope
        if s_min == 0.0:
            rng   = max(1.0, abs(X[-1] - X[0]))
            s_min = 0.01 * rng / W

        r = ramp_baseline(start, s_min, W)
        c = constant_baseline(start, W)

        _dist = ddtw_distance if self.cfg.use_ddtw else dtw_distance

        D1    = _dist(X, r, window=self.cfg.band)
        D2    = _dist(X, c, window=self.cfg.band)
        delta = D1 - D2

        self.d1_hist.append(D1)
        self.d2_hist.append(D2)
        self.delta_hist.append(delta)

        # ── Umbrales adaptativos ──────────────────────────────────────────
        if self.cfg.adapt_thresholds and len(self.d1_hist) >= 10:
            theta_c     = moving_percentile(self.d2_hist,    self.cfg.p_low)
            theta_r     = moving_percentile(self.d1_hist,    self.cfg.p_high)
            theta_delta = moving_percentile(self.delta_hist, self.cfg.p_high)
        else:
            theta_c     = 0.1 * W
            theta_r     = 0.5 * W
            theta_delta = 0.3 * W

        # ── Condiciones de estancamiento ──────────────────────────────────
        cond_plateau  = self.no_improve_len >= self.cfg.plateau_max
        cond_constant = D2 <= theta_c
        cond_ramp     = (D1 >= theta_r) or (delta >= theta_delta)

        if cond_plateau and cond_constant and cond_ramp:
            self.trigger_streak += 1
        else:
            self.trigger_streak = 0

        fire = self.trigger_streak >= self.cfg.patience

        return {
            "ready":          True,
            "fire":           bool(fire),
            "D1_vs_ramp":     float(D1),
            "D2_vs_const":    float(D2),
            "delta":          float(delta),
            "theta_c":        float(theta_c),
            "theta_r":        float(theta_r),
            "theta_delta":    float(theta_delta),
            "no_improve_len": int(self.no_improve_len),
            "trigger_streak": int(self.trigger_streak),
            "n":              int(n),
        }
