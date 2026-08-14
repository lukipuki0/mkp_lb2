"""
mezclas_mh/woa_abc/variante_d_dtw.py
------------------------------------
Variante D: DTW-Adaptive WOA-ABC (Control Interno por DTW).

En esta variante, el monitor DTW actúa como un controlador adaptativo interno:
  1. Conmutación DTW-Driven: Al detectarse desaceleración o meseta (delta DTW < theta_delta),
     el algoritmo conmuta de WOA (Exploración) a ABC (Explotación).
  2. Momentum Adaptativo por DTW: El factor de momentum beta se amplifica cuando el delta DTW decrece,
     impulsando la salida de pozos de atracción locales.
  3. DTW Scout Rescue: Ante alertas críticas de estancamiento (fire == True), se reinicializa
     el 30% peor de la población manteniendo al líder.

Soporta:
  - Problema Discreto MKP (con binarización LB2 y reparación).
  - Benchmark Continuo CEC2022.
"""

from __future__ import annotations

import os
import sys
import math
import random
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mkp_core.problem import MKPInstance
from mkp_core.repair import reparar_solucion
from lb2 import binarizar_posicion, interpolar_G
from dtw_stagnation import StagnationConfig, StagnationMonitor
from continuous_benchmark.funciones_cec2022 import ContinuousFunction, get_test_functions


# ── Parametros y Resultados ───────────────────────────────────────────────────

@dataclass
class DTWWOAABCParams:
    """Hiperparámetros para la Variante D (DTW-Adaptive)."""
    pop_size       : int   = 30
    iterations     : int   = 100
    epochs         : int   = 1
    v_max          : float = 6.0
    # LB2 params
    G1_i: float = 0.5;  G1_f: float = 1.0
    G2_i: float = 0.5;  G2_f: float = 7.2
    G3_i: float = 0.5;  G3_f: float = 0.0
    # Momentum adaptativo
    beta_min       : float = 0.1
    beta_max       : float = 0.9
    step_init      : float = 0.2
    step_decay     : float = 0.02
    # Inyección
    injection_mode : str   = "mixed"
    # DTW Stagnation Monitor
    stag_cfg       : StagnationConfig = field(default_factory=lambda: StagnationConfig(
        window=10, patience=3, min_slope=0.01, plateau_max=5
    ))


@dataclass
class DTWWOAABCEpochResult:
    """Resultado de la Variante D por epoch."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[int] | np.ndarray = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)
    fases            : list[str]   = field(default_factory=list)


# ── Helpers MKP ───────────────────────────────────────────────────────────────

def _inicializar_poblacion_mkp(inst: MKPInstance, pop_size: int, v_max: float):
    n = inst.n
    posiciones = np.random.uniform(-v_max, v_max, size=(pop_size, n))
    poblacion_bin = []
    fitnesses = []
    for _ in range(pop_size):
        sol = [random.randint(0, 1) for _ in range(n)]
        sol, val = reparar_solucion(sol, inst)
        poblacion_bin.append(sol)
        fitnesses.append(val)
    trials = np.zeros(pop_size, dtype=int)
    momentum = np.zeros((pop_size, n))
    return posiciones, poblacion_bin, fitnesses, trials, momentum


def _dtw_scout_rescue_mkp(
    posiciones: np.ndarray,
    poblacion_bin: list[list[int]],
    fitnesses: list[float],
    inst: MKPInstance,
    v_max: float,
    ratio: float = 0.3,
):
    """Reinicializa el peor ratio% de soluciones manteniendo al líder intacto."""
    pop_size, n = posiciones.shape
    num_rescue = max(1, int(pop_size * ratio))
    peores_indices = np.argsort(fitnesses)[:num_rescue]
    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])

    for idx in peores_indices:
        if idx == best_idx:
            continue
        # Inversión por oposición combinada con perturbación gaussiana
        posiciones[idx] = -posiciones[best_idx] + np.random.normal(0, 1.0, size=n)
        posiciones[idx] = np.clip(posiciones[idx], -v_max, v_max)
        sol_opp = [1 - b for b in poblacion_bin[best_idx]]
        sol_rep, val_rep = reparar_solucion(sol_opp, inst)
        poblacion_bin[idx] = sol_rep
        fitnesses[idx] = val_rep


# ── Executor MKP (Variante D) ────────────────────────────────────────────────

def ejecutar_epoch(
    inst: MKPInstance,
    params: DTWWOAABCParams,
    epoch_idx: int = 0,
    verbose: bool = True,
    sol_inyectada: list[int] | None = None,
) -> DTWWOAABCEpochResult:
    """Ejecuta la Variante D en el dominio discreto MKP."""
    n = inst.n
    pop_size = params.pop_size
    limit = int(pop_size * n / 2)

    posiciones, poblacion_bin, fitnesses, trials, momentum = _inicializar_poblacion_mkp(
        inst, pop_size, params.v_max
    )

    if sol_inyectada is not None:
        sol_rep, val_rep = reparar_solucion(list(sol_inyectada), inst)
        poblacion_bin[0] = sol_rep
        fitnesses[0] = val_rep

    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    mejor_val = fitnesses[best_idx]
    mejor_sol = poblacion_bin[best_idx].copy()

    historial = []
    historial_inst = []
    dtw_deltas = []
    fases = []
    stag_fires = 0

    monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        # 1. Actualizar Monitor DTW y determinar estado
        status = monitor.update(mejor_val)
        delta_dtw = status.get("delta", 0.0) if status.get("ready") else 0.0
        dtw_deltas.append(delta_dtw)

        # Determinar Fase según DTW
        fase_actual = "WOA"
        if status.get("ready"):
            td = status.get("theta_delta", 0.0)
            if status.get("fire") or delta_dtw <= td:
                fase_actual = "ABC"
        fases.append(fase_actual)

        # Rescate DTW si dispara alerta crítica
        if status.get("fire"):
            stag_fires += 1
            _dtw_scout_rescue_mkp(posiciones, poblacion_bin, fitnesses, inst, params.v_max)

        # Ajuste dinámico de Momentum beta según Delta DTW
        if delta_dtw < 0:
            beta = params.beta_max
        else:
            beta = params.beta_min + (params.beta_max - params.beta_min) * math.exp(-0.5 * delta_dtw)
        beta = float(np.clip(beta, params.beta_min, params.beta_max))

        G1 = interpolar_G(it, params.iterations, params.G1_i, params.G1_f)
        G2 = interpolar_G(it, params.iterations, params.G2_i, params.G2_f)
        G3 = interpolar_G(it, params.iterations, params.G3_i, params.G3_f)

        step_size = params.step_init * math.exp(-params.step_decay * it)
        a_woa = 2.0 - it * (2.0 / params.iterations)

        best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        X_best = posiciones[best_idx].copy()

        # 2. Ejecutar Paso según Fase
        if fase_actual == "WOA":
            for i in range(pop_size):
                r1, r2 = random.random(), random.random()
                A = 2.0 * a_woa * r1 - a_woa
                C = 2.0 * r2
                p = random.random()
                b_param, l = 1.0, random.uniform(-1.0, 1.0)

                if p < 0.5:
                    if abs(A) < 1.0:
                        D = abs(C * X_best - posiciones[i])
                        X_new = X_best - A * D
                    else:
                        rand_idx = random.randint(0, pop_size - 1)
                        D = abs(C * posiciones[rand_idx] - posiciones[i])
                        X_new = posiciones[rand_idx] - A * D
                else:
                    D_prime = abs(X_best - posiciones[i])
                    X_new = D_prime * math.exp(b_param * l) * math.cos(2.0 * math.pi * l) + X_best

                X_new = np.clip(X_new, -params.v_max, params.v_max)
                sol_prop, val_prop = binarizar_posicion(X_new, poblacion_bin[i], inst, G1, G2, G3, params.v_max)
                if val_prop >= fitnesses[i]:
                    posiciones[i] = X_new
                    poblacion_bin[i] = sol_prop
                    fitnesses[i] = val_prop

        else:  # Fase ABC con Momentum
            for i in range(pop_size):
                k = random.choice([idx for idx in range(pop_size) if idx != i])
                phi = random.uniform(-1.0, 1.0)
                momentum[i] = beta * momentum[i] + (1.0 - beta) * phi * (posiciones[i] - posiciones[k])
                v_i = posiciones[i] + step_size * momentum[i]
                v_i = np.clip(v_i, -params.v_max, params.v_max)

                sol_prop, val_prop = binarizar_posicion(v_i, poblacion_bin[i], inst, G1, G2, G3, params.v_max)
                if val_prop >= fitnesses[i]:
                    posiciones[i] = v_i
                    poblacion_bin[i] = sol_prop
                    fitnesses[i] = val_prop
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Scout bees en ABC
            for i in range(pop_size):
                if trials[i] >= limit:
                    posiciones[i] = np.random.uniform(-params.v_max, params.v_max, size=n)
                    rand_sol = [random.randint(0, 1) for _ in range(n)]
                    sol_rep, val_rep = reparar_solucion(rand_sol, inst)
                    poblacion_bin[i] = sol_rep
                    fitnesses[i] = val_rep
                    trials[i] = 0

        # Actualizar mejor global
        best_idx_iter = max(range(pop_size), key=lambda idx: fitnesses[idx])
        fit_iter_best = fitnesses[best_idx_iter]
        if fit_iter_best > mejor_val:
            mejor_val = fit_iter_best
            mejor_sol = poblacion_bin[best_idx_iter].copy()

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)

        if verbose:
            print(f"  [Variante D MKP] Iter {it+1:3d}/{params.iterations} | Fase: {fase_actual:<3} | Beta: {beta:.2f} | Delta: {delta_dtw:6.1f} | Mejor: {mejor_val:10.1f}", flush=True)

    return DTWWOAABCEpochResult(
        epoch_idx=epoch_idx,
        mejor_valor=mejor_val,
        iteraciones=len(historial),
        stagnation_fires=stag_fires,
        historial=historial,
        historial_inst=historial_inst,
        mejor_solucion=mejor_sol,
        dtw_deltas=dtw_deltas,
        fases=fases,
    )


# ── Executor Continuo CEC2022 (Variante D) ───────────────────────────────────

def ejecutar_epoch_continuo(
    func: ContinuousFunction,
    params: DTWWOAABCParams,
    epoch_idx: int = 0,
    verbose: bool = True,
) -> DTWWOAABCEpochResult:
    """Ejecuta la Variante D en el benchmark continuo CEC2022 (minimización)."""
    dim = func.n_dim
    lb, ub = func.lb, func.ub
    pop_size = params.pop_size

    posiciones = np.random.uniform(lb, ub, size=(pop_size, dim))
    fitnesses = np.array([func.func(p) for p in posiciones])
    momentum = np.zeros((pop_size, dim))

    best_idx = np.argmin(fitnesses)
    mejor_val = fitnesses[best_idx]
    mejor_sol = posiciones[best_idx].copy()

    historial = []
    historial_inst = []
    dtw_deltas = []
    fases = []
    stag_fires = 0

    monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        # 1. En minimización usamos -fitness para el monitor
        status = monitor.update(-mejor_val)
        delta_dtw = status.get("delta", 0.0) if status.get("ready") else 0.0
        dtw_deltas.append(delta_dtw)

        fase_actual = "WOA"
        if status.get("ready"):
            td = status.get("theta_delta", 0.0)
            if status.get("fire") or delta_dtw <= td:
                fase_actual = "ABC"
        fases.append(fase_actual)

        # Rescate DTW continuo
        if status.get("fire"):
            stag_fires += 1
            num_rescue = max(1, int(pop_size * 0.3))
            peores_indices = np.argsort(fitnesses)[-num_rescue:]
            for idx in peores_indices:
                if idx == best_idx:
                    continue
                posiciones[idx] = np.random.uniform(lb, ub, size=dim)
                fitnesses[idx] = func.func(posiciones[idx])

        if delta_dtw < 0:
            beta = params.beta_max
        else:
            beta = params.beta_min + (params.beta_max - params.beta_min) * math.exp(-0.5 * delta_dtw)
        beta = float(np.clip(beta, params.beta_min, params.beta_max))

        step_size = params.step_init * math.exp(-params.step_decay * it)
        a_woa = 2.0 - it * (2.0 / params.iterations)
        best_idx = np.argmin(fitnesses)
        X_best = posiciones[best_idx].copy()

        if fase_actual == "WOA":
            for i in range(pop_size):
                r1, r2 = random.random(), random.random()
                A = 2.0 * a_woa * r1 - a_woa
                C = 2.0 * r2
                p = random.random()
                b_param, l = 1.0, random.uniform(-1.0, 1.0)

                if p < 0.5:
                    if abs(A) < 1.0:
                        D = abs(C * X_best - posiciones[i])
                        X_new = X_best - A * D
                    else:
                        rand_idx = random.randint(0, pop_size - 1)
                        D = abs(C * posiciones[rand_idx] - posiciones[i])
                        X_new = posiciones[rand_idx] - A * D
                else:
                    D_prime = abs(X_best - posiciones[i])
                    X_new = D_prime * math.exp(b_param * l) * math.cos(2.0 * math.pi * l) + X_best

                X_new = np.clip(X_new, lb, ub)
                val_new = func.func(X_new)
                if val_new <= fitnesses[i]:
                    posiciones[i] = X_new
                    fitnesses[i] = val_new

        else:  # ABC continuo
            for i in range(pop_size):
                k = random.choice([idx for idx in range(pop_size) if idx != i])
                phi = random.uniform(-1.0, 1.0)
                momentum[i] = beta * momentum[i] + (1.0 - beta) * phi * (posiciones[i] - posiciones[k])
                v_i = posiciones[i] + step_size * momentum[i]
                v_i = np.clip(v_i, lb, ub)
                val_new = func.func(v_i)
                if val_new <= fitnesses[i]:
                    posiciones[i] = v_i
                    fitnesses[i] = val_new

        best_idx_iter = np.argmin(fitnesses)
        fit_iter_best = fitnesses[best_idx_iter]

        if fit_iter_best < mejor_val:
            mejor_val = fit_iter_best
            mejor_sol = posiciones[best_idx_iter].copy()

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)

        if verbose:
            print(f"  [Variante D Continuo] Iter {it+1:3d}/{params.iterations} | Fase: {fase_actual:<3} | Beta: {beta:.2f} | Delta: {delta_dtw:6.1f} | Mejor: {mejor_val:10.4f}", flush=True)

    return DTWWOAABCEpochResult(
        epoch_idx=epoch_idx,
        mejor_valor=mejor_val,
        iteraciones=len(historial),
        stagnation_fires=stag_fires,
        historial=historial,
        historial_inst=historial_inst,
        mejor_solucion=mejor_sol,
        dtw_deltas=dtw_deltas,
        fases=fases,
    )


# ── Pruebas Unitarias ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from mkp_core.data_loader import cargar_instancias

    print("=== PRUEBA UNITARIA: VARIANTE D (DTW-ADAPTIVE WOA-ABC) ===")
    p = DTWWOAABCParams(pop_size=20, iterations=30)

    # 1. MKP
    insts = cargar_instancias("instancias/mknapcb1.txt")
    inst = MKPInstance.from_dict(insts[0])
    res_mkp = ejecutar_epoch(inst, p, verbose=True)
    print(f"\nResult MKP -> Mejor Valor: {res_mkp.mejor_valor} (Opt: {inst.valor_optimo})")

    # 2. Continuo
    funcs = get_test_functions(n_dim=10)
    res_cont = ejecutar_epoch_continuo(funcs[0], p, verbose=True)
    print(f"\nResult CEC2022 {funcs[0].name} -> Mejor Valor: {res_cont.mejor_valor:.4f} (Opt: {funcs[0].optimum})")
