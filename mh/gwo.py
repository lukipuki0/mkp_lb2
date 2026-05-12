"""
mh/gwo.py
---------
Grey Wolf Optimizer (GWO) para el MKP con binarización LB2.

Versión limpia para el pipeline híbrido: solo usa estrategia "abort"
cuando el monitor DTW detecta estancamiento.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from mkp_core.problem   import MKPInstance
from mkp_core.repair    import reparar_solucion
from dtw_stagnation     import StagnationConfig, StagnationMonitor
from lb2 import binarizar_posicion, interpolar_G


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class GWOParams:
    """Hiperparámetros del GWO."""
    pop_size    : int   = 30
    iterations  : int   = 300
    epochs      : int   = 10
    v_max       : float = 6.0
    # LB2 params
    G1_i : float = 0.5;  G1_f : float = 1.0
    G2_i : float = 0.5;  G2_f : float = 7.2
    G3_i : float = 0.5;  G3_f : float = 0.0
    # Inyección de solución (pipeline híbrido)
    injection_mode : str  = "random"    # "random" | "mutated" | "mixed"
    # Stagnation
    use_stagnation : bool = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class GWOEpochResult:
    """Resultado de un epoch del GWO."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    mejor_solucion   : list[int]  = field(default_factory=list)


@dataclass
class GWOResult:
    """Resultado completo del GWO (todos los epochs)."""
    epochs             : list[GWOEpochResult]
    mejor_valor_global : float
    mejor_sol_global   : list[int]
    valor_optimo       : float

    @property
    def gap_pct(self) -> float | None:
        if self.valor_optimo == 0:
            return None
        return 100.0 * (self.valor_optimo - self.mejor_valor_global) / self.valor_optimo

    @property
    def valores_por_epoch(self) -> list[float]:
        return [ep.mejor_valor for ep in self.epochs]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inicializar_manada(
    inst: MKPInstance,
    pop_size: int,
    v_max: float,
) -> tuple[np.ndarray, list[list[int]], list[float]]:
    """Genera la manada inicial: posiciones continuas + soluciones binarias factibles."""
    n = inst.n
    posiciones = np.random.uniform(-v_max, v_max, size=(pop_size, n))
    poblacion_bin = []
    fitnesses     = []

    for i in range(pop_size):
        sol = [random.randint(0, 1) for _ in range(n)]
        sol, val = reparar_solucion(sol, inst)
        poblacion_bin.append(sol)
        fitnesses.append(val)

    return posiciones, poblacion_bin, fitnesses


def _mutar_solucion(sol: list[int], inst: MKPInstance, n_flips: int = 0) -> tuple[list[int], float]:
    """Crea una copia de `sol` con bits invertidos al azar y la repara."""
    copia = list(sol)
    n = len(copia)
    if n_flips <= 0:
        n_flips = random.randint(1, 3)
    indices = random.sample(range(n), min(n_flips, n))
    for idx in indices:
        copia[idx] = 1 - copia[idx]
    copia, val = reparar_solucion(copia, inst)
    return copia, val


def _obtener_jerarquia(fitnesses: list[float]) -> tuple[int, int, int]:
    """Devuelve los índices de Alpha, Beta y Delta (los 3 mejores lobos)."""
    indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    return indices[0], indices[1], indices[2]


# ── Epoch individual ─────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst      : MKPInstance,
    params    : GWOParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: list[int] | None = None,
) -> GWOEpochResult:
    """Ejecuta un epoch completo del GWO con detección de estancamiento (abort)."""

    pop_size = params.pop_size
    n = inst.n

    # Inicializar manada
    posiciones, poblacion_bin, fitnesses = _inicializar_manada(
        inst, pop_size, params.v_max,
    )

    # Inyectar solución del orquestador según el modo de inyección
    if sol_inyectada is not None:
        sol_rep = list(sol_inyectada)
        sol_rep, val_rep = reparar_solucion(sol_rep, inst)
        mode = params.injection_mode

        if mode == "random":
            peor_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            poblacion_bin[peor_idx] = sol_rep
            fitnesses[peor_idx] = val_rep

        elif mode == "mutated":
            poblacion_bin[0] = sol_rep
            fitnesses[0] = val_rep
            for i in range(1, pop_size):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion_bin[i] = msol
                fitnesses[i] = mval

        elif mode == "mixed":
            poblacion_bin[0] = sol_rep
            fitnesses[0] = val_rep
            n_mutados = pop_size // 2
            for i in range(1, n_mutados):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion_bin[i] = msol
                fitnesses[i] = mval

    alpha_idx, beta_idx, delta_idx = _obtener_jerarquia(fitnesses)
    mejor_val = fitnesses[alpha_idx]
    mejor_sol = poblacion_bin[alpha_idx].copy()

    historial  = []
    stag_fires = 0

    # Estado dinámico de los parámetros G
    G1 = params.G1_i
    G2 = params.G2_i
    G3 = params.G3_i

    # Inicializar monitor
    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        # Coeficiente 'a' del GWO: decrece linealmente de 2 a 0
        a = 2.0 - it * (2.0 / params.iterations)

        # Posiciones de los líderes
        X_alpha = posiciones[alpha_idx].copy()
        X_beta  = posiciones[beta_idx].copy()
        X_delta = posiciones[delta_idx].copy()

        # Actualizar cada lobo
        for i in range(pop_size):
            if i in (alpha_idx, beta_idx, delta_idx):
                continue

            X_i = posiciones[i]

            # Coeficientes aleatorios para cada líder
            r1_a, r2_a = np.random.random(n), np.random.random(n)
            r1_b, r2_b = np.random.random(n), np.random.random(n)
            r1_d, r2_d = np.random.random(n), np.random.random(n)

            A1 = 2.0 * a * r1_a - a
            C1 = 2.0 * r2_a
            A2 = 2.0 * a * r1_b - a
            C2 = 2.0 * r2_b
            A3 = 2.0 * a * r1_d - a
            C3 = 2.0 * r2_d

            # Posiciones guiadas por los 3 líderes
            D_alpha = np.abs(C1 * X_alpha - X_i)
            D_beta  = np.abs(C2 * X_beta  - X_i)
            D_delta = np.abs(C3 * X_delta - X_i)

            X1 = X_alpha - A1 * D_alpha
            X2 = X_beta  - A2 * D_beta
            X3 = X_delta - A3 * D_delta

            # Nueva posición continua (promedio de las 3 guías)
            X_new = (X1 + X2 + X3) / 3.0
            X_new = np.clip(X_new, -params.v_max, params.v_max)
            posiciones[i] = X_new

            # Binarización LB2
            nueva_sol, nueva_val = binarizar_posicion(
                X_new, poblacion_bin[i], inst,
                G1, G2, G3, params.v_max,
            )

            # Actualizar solo si mejora (selección greedy)
            if nueva_val >= fitnesses[i]:
                poblacion_bin[i] = nueva_sol
                fitnesses[i]     = nueva_val

        # Recalcular jerarquía
        alpha_idx, beta_idx, delta_idx = _obtener_jerarquia(fitnesses)

        # Actualizar mejor global
        if fitnesses[alpha_idx] > mejor_val:
            mejor_val = fitnesses[alpha_idx]
            mejor_sol = poblacion_bin[alpha_idx].copy()

        historial.append(mejor_val)

        # ── Stagnation check ──────────────────────────────────────────────
        if monitor is not None:
            status = monitor.update(mejor_val)

            if verbose and status.get("ready"):
                d1  = status.get("D1_vs_ramp", 0.0)
                d2  = status.get("D2_vs_const", 0.0)
                dlt = status.get("delta", 0.0)
                td  = status.get("theta_delta", 0.0)
                tc  = status.get("theta_c", 0.0)
                tr  = status.get("theta_r", 0.0)
                ni  = status.get("no_improve_len", 0)
                fr  = status.get("fire", False)
                ns  = status.get("n", 0)
                print(
                    f"iter={it:03d}",
                    f"n={ns}",
                    f"D1={d1:.3f}",
                    f"D2={d2:.3f}",
                    f"Delta={dlt:.3f}",
                    f"theta={td}",
                    f"theta_c={tc:.2f}",
                    f"theta_r={tr:.2f}",
                    f"no_improve={ni}",
                    f"fire={fr}",
                    f"best={mejor_val:.1f}",
                )

            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break
        else:
            # Sin monitor: transición lineal ciega de G
            G1 = interpolar_G(it, params.iterations, params.G1_i, params.G1_f)
            G2 = interpolar_G(it, params.iterations, params.G2_i, params.G2_f)
            G3 = interpolar_G(it, params.iterations, params.G3_i, params.G3_f)

    return GWOEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = params.iterations,
        stagnation_fires = stag_fires,
        historial        = historial,
        mejor_solucion   = mejor_sol,
    )


# ── Ejecución multi-epoch ────────────────────────────────────────────────────

def ejecutar_gwo(
    inst: MKPInstance,
    params: GWOParams,
    verbose: bool = True,
) -> GWOResult:
    """Ejecuta el GWO completo (todos los epochs) y retorna el GWOResult."""
    epochs_result    = []
    mejor_val_global = -float("inf")
    mejor_sol_global: list[int] = []

    for e in range(params.epochs):
        epoch_res = ejecutar_epoch(inst, params, epoch_idx=e, verbose=verbose)
        epochs_result.append(epoch_res)

        if epoch_res.mejor_valor > mejor_val_global:
            mejor_val_global = epoch_res.mejor_valor
            mejor_sol_global = epoch_res.mejor_solucion.copy()

    return GWOResult(
        epochs             = epochs_result,
        mejor_valor_global = mejor_val_global,
        mejor_sol_global   = mejor_sol_global,
        valor_optimo       = inst.valor_optimo,
    )

