"""
lb2/binarization.py
-------------------
Framework de binarización LB2 reutilizable para convertir posiciones
continuas de cualquier metaheurística (PSO, GWO, etc.) en decisiones
binarias (0/1) para el MKP.

Las fórmulas L1/L2 son **idénticas** al notebook de referencia:
    kirito/LB2_MKP.ipynb  (celdas "Parámetros de binarización" y "Original")

Fórmulas clave del notebook:
    L1 = -G1 * v / (V_max - G2) + G3
    L2 =  G1 * v / (V_max - G2) + G3
    prob = np.clip(prob, 0, 1)
    if rand < prob: bit = 1 - bit   # flip

Transición lineal de G (notebook):
    G = G_f + (G_i - G_f) / (1 - T_max) * (t - T_max)
"""

from __future__ import annotations

import random
import numpy as np

from mkp_core.problem import MKPInstance
from mkp_core.repair  import reparar_solucion


# ── Funciones de transferencia L1 / L2 (vectorizadas) ────────────────────────

def _calcular_L1(
    velocidades: np.ndarray,
    v_max: float,
    G1: float,
    G2: float,
    G3: float,
) -> np.ndarray:
    """Función de transferencia L1 (conservadora, pendiente negativa).

    Fórmula del notebook:
        L1 = -G1 * v / (V_max - G2) + G3
        L1 = np.clip(L1, 0, 1)
    """
    denom = v_max - G2
    if abs(denom) < 1e-12:
        denom = 1e-12
    prob = -G1 * velocidades / denom + G3
    return np.clip(prob, 0.0, 1.0)


def _calcular_L2(
    velocidades: np.ndarray,
    v_max: float,
    G1: float,
    G2: float,
    G3: float,
) -> np.ndarray:
    """Función de transferencia L2 (agresiva, pendiente positiva).

    Fórmula del notebook:
        L2 = G1 * v / (V_max - G2) + G3
        L2 = np.clip(L2, 0, 1)
    """
    denom = v_max - G2
    if abs(denom) < 1e-12:
        denom = 1e-12
    prob = G1 * velocidades / denom + G3
    return np.clip(prob, 0.0, 1.0)


# ── Binarización completa ────────────────────────────────────────────────────

def binarizar_posicion(
    velocidades: np.ndarray,
    sol_actual: list[int],
    inst: MKPInstance,
    G1: float,
    G2: float,
    G3: float,
    v_max: float,
) -> tuple[list[int], float]:
    """Aplica binarización LB2 sobre un vector de velocidades.

    Genera DOS candidatas usando L1 y L2 respectivamente.
    Cada candidata se obtiene haciendo 'flip' probabilístico sobre
    la solución actual (NO se descarta la solución, a diferencia del
    BPSO clásico). Ambas se reparan y se devuelve la mejor.

    Lógica idéntica al notebook kirito/LB2_MKP.ipynb (celda "Original").

    Parameters
    ----------
    velocidades : np.ndarray shape (n,)
        Vector de velocidades continuas (posiciones en el GWO).
    sol_actual : list[int]
        Solución binaria actual.
    inst : MKPInstance
        Instancia del problema.
    G1, G2, G3 : float
        Parámetros de control LB2.
    v_max : float
        Velocidad máxima para normalización.

    Returns
    -------
    tuple[list[int], float]
        Mejor solución binaria factible y su valor.
    """
    n = inst.n

    # Calcular probabilidades de flip (vectorizado)
    L1 = _calcular_L1(velocidades, v_max, G1, G2, G3)
    L2 = _calcular_L2(velocidades, v_max, G1, G2, G3)

    # Candidata 1: flips por L1
    x1 = list(sol_actual)  # copia
    for i in range(n):
        if np.random.rand() < L1[i]:
            x1[i] = 1 - x1[i]

    # Candidata 2: flips por L2
    x2 = list(sol_actual)  # copia
    for i in range(n):
        if np.random.rand() < L2[i]:
            x2[i] = 1 - x2[i]

    # Reparar ambas
    x1, val_1 = reparar_solucion(x1, inst)
    x2, val_2 = reparar_solucion(x2, inst)

    # Selección greedy: devolver la mejor
    if val_1 > val_2:
        x2 = x1
        val_2 = val_1

    return x2, val_2


# ── Transición de parámetros G ───────────────────────────────────────────────

def interpolar_G(
    t: int,
    t_max: int,
    G_initial: float,
    G_final: float,
) -> float:
    """Interpolación lineal de un parámetro G.

    Fórmula del notebook (kirito/LB2_MKP.ipynb):
        G = G_f + (G_i - G_f) / (1 - T_max) * (t - T_max)

    Esto es una interpolación lineal de G_i a G_f conforme t va de 0 a T_max-1.
    """
    if t_max <= 1:
        return G_final
    return G_final + (G_initial - G_final) / (1 - t_max) * (t - t_max)


def adaptar_G_por_dtw(
    delta: float,
    theta_delta: float,
    G1_i: float, G1_f: float,
    G2_i: float, G2_f: float,
    G3_i: float, G3_f: float,
) -> tuple[float, float, float, str]:
    """Modula G1, G2, G3 según el Delta del DTW (4 estados).

    Referencia: kirito/CONTEXT_LB2_MKP.md

    Estados:
      1. Explorar mucho  (delta >  theta_delta)  → G exploratorios
      2. Explorar poco    (0 <= delta <= theta)   → G moderados
      3. Explotar poco    (-theta <= delta < 0)   → G equilibrados
      4. Explotar mucho   (delta < -theta_delta)  → G explotadores

    Returns
    -------
    tuple[float, float, float, str]
        (G1, G2, G3, nombre_del_estado)
    """
    if delta > theta_delta:
        # Estado 1: Estancamiento severo -> explorar mucho
        G1 = G1_i
        G2 = G2_i
        G3 = G3_i
        estado = "Explorar mucho"
    elif 0 <= delta <= theta_delta:
        # Estado 2: Estancamiento leve -> explorar poco
        alpha = 0.35
        G1 = G1_i + alpha * (G1_f - G1_i)
        G2 = G2_i + alpha * (G2_f - G2_i)
        G3 = G3_i + alpha * (G3_f - G3_i)
        estado = "Explorar poco"
    elif -theta_delta <= delta < 0:
        # Estado 3: Mejora constante -> explotar poco (equilibrio)
        alpha = 0.65
        G1 = G1_i + alpha * (G1_f - G1_i)
        G2 = G2_i + alpha * (G2_f - G2_i)
        G3 = G3_i + alpha * (G3_f - G3_i)
        estado = "Explotar poco"
    else:
        # Estado 4: Mejora repentina fuerte -> explotar mucho
        G1 = G1_f
        G2 = G2_f
        G3 = G3_f
        estado = "Explotar mucho"

    return G1, G2, G3, estado
