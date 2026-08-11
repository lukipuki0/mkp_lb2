"""
mezclas_mh/woa_abc/variante_c.py
--------------------------------
Variante C: Switching basado en la Diversidad Poblacional.

Ecuaciones Clave:
  X_mean(t) = (1/Npop) * sum(X_i(t))
  Diversidad(t) = (1/Npop) * sum(||X_i(t) - X_mean(t)||)
  Diversidad_norm(t) = Diversidad(t) / Diversidad(0)
  umbral(t) = umbral_init * (1 - t/MaxIt) + umbral_final * (t/MaxIt)

Decisión de fase:
  Si Diversidad_norm(t) <= umbral(t): Ejecutar WOA (romper agrupamiento).
  Si Diversidad_norm(t) > umbral(t):  Ejecutar ABC (refinamiento por explotación).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mkp_core.problem import MKPInstance
from mkp_core.repair import reparar_solucion
from dtw_stagnation import StagnationConfig, StagnationMonitor
from lb2 import binarizar_posicion, interpolar_G


@dataclass
class VariantCParams:
    pop_size : int = 30
    iterations : int = 300
    epochs : int = 1
    v_max : float = 6.0
    limit : int | None = None
    b_spiral : float = 1.0
    umbral_init : float = 0.5    # Umbral de diversidad al inicio
    umbral_final : float = 0.1   # Umbral de diversidad al final
    # LB2 params
    G1_i : float = 0.5; G1_f : float = 1.0
    G2_i : float = 0.5; G2_f : float = 7.2
    G3_i : float = 0.5; G3_f : float = 0.0
    injection_mode : str = "mixed"
    use_stagnation : bool = True
    stag_cfg : StagnationConfig | None = None


@dataclass
class VariantCEpochResult:
    epoch_idx : int
    mejor_valor : float
    iteraciones : int
    stagnation_fires : int
    historial : list[float] = field(default_factory=list)
    historial_inst : list[float] = field(default_factory=list)
    historial_diversidad : list[float] = field(default_factory=list)
    mejor_solucion : list = field(default_factory=list)
    dtw_deltas : list[float] = field(default_factory=list)


def _calcular_diversidad(posiciones: np.ndarray) -> float:
    centroid = np.mean(posiciones, axis=0)
    dists = np.linalg.norm(posiciones - centroid, axis=1)
    return float(np.mean(dists))


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOMINIO DISCRETO MKP
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst : MKPInstance,
    params : VariantCParams,
    epoch_idx : int = 0,
    verbose : bool = True,
    sol_inyectada : list[int] | None = None,
) -> VariantCEpochResult:

    n = inst.n
    pop_size = params.pop_size
    limit = params.limit if params.limit is not None else int(pop_size * n / 2)

    posiciones = np.random.uniform(-params.v_max, params.v_max, size=(pop_size, n))
    poblacion_bin = []
    fitnesses = []

    for i in range(pop_size):
        sol = [random.randint(0, 1) for _ in range(n)]
        sol, val = reparar_solucion(sol, inst)
        poblacion_bin.append(sol)
        fitnesses.append(val)

    trials = np.zeros(pop_size, dtype=int)

    if sol_inyectada is not None:
        sol_rep = list(sol_inyectada)
        sol_rep, val_rep = reparar_solucion(sol_rep, inst)
        peor_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        poblacion_bin[peor_idx] = sol_rep
        fitnesses[peor_idx] = val_rep
        trials[peor_idx] = 0

    div_0 = max(1e-12, _calcular_diversidad(posiciones))

    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    mejor_val = fitnesses[best_idx]
    mejor_sol = poblacion_bin[best_idx].copy()
    best_pos = posiciones[best_idx].copy()

    historial = []
    historial_inst = []
    historial_div = []
    dtw_deltas = []
    stag_fires = 0

    G1, G2, G3 = params.G1_i, params.G2_i, params.G3_i

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        div_t = _calcular_diversidad(posiciones)
        div_norm = div_t / div_0
        historial_div.append(div_norm)

        umbral_t = params.umbral_init * (1.0 - it / max(1, params.iterations)) + params.umbral_final * (it / max(1, params.iterations))
        a = 2.0 - it * (2.0 / params.iterations)

        if div_norm <= umbral_t:
            # ══ FASE WOA (Población agrupada -> Explorar) ══
            for i in range(pop_size):
                r1, r2 = np.random.random(n), np.random.random(n)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = random.random()
                l = random.uniform(-1.0, 1.0)
                norm_A = np.linalg.norm(A) / np.sqrt(n)

                if p < 0.5 and norm_A >= 1.0:
                    X_rand = posiciones[random.randint(0, pop_size - 1)]
                    D = np.abs(C * X_rand - posiciones[i])
                    X_new = X_rand - A * D
                elif p < 0.5 and norm_A < 1.0:
                    D = np.abs(C * best_pos - posiciones[i])
                    X_new = best_pos - A * D
                else:
                    D_prime = np.abs(best_pos - posiciones[i])
                    X_new = D_prime * np.exp(params.b_spiral * l) * np.cos(2.0 * np.pi * l) + best_pos

                X_new = np.clip(X_new, -params.v_max, params.v_max)
                sol_prop, val_prop = binarizar_posicion(
                    X_new, poblacion_bin[i], inst, G1, G2, G3, params.v_max
                )

                if val_prop >= fitnesses[i]:
                    posiciones[i] = X_new
                    poblacion_bin[i] = sol_prop
                    fitnesses[i] = val_prop
                    trials[i] = 0
                else:
                    trials[i] += 1
        else:
            # ══ FASE ABC (Población dispersa -> Explotar) ══
            for i in range(pop_size):
                k = random.choice([idx for idx in range(pop_size) if idx != i])
                j = random.randint(0, n - 1)
                phi = random.uniform(-1.0, 1.0)

                X_new = posiciones[i].copy()
                X_new[j] = X_new[j] + (X_new[j] - posiciones[k][j]) * phi
                X_new = np.clip(X_new, -params.v_max, params.v_max)

                sol_prop, val_prop = binarizar_posicion(
                    X_new, poblacion_bin[i], inst, G1, G2, G3, params.v_max
                )

                if val_prop >= fitnesses[i]:
                    posiciones[i] = X_new
                    poblacion_bin[i] = sol_prop
                    fitnesses[i] = val_prop
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker bee
            max_fit = max(fitnesses)
            probs = np.array([0.9 * (fit / max_fit) + 0.1 for fit in fitnesses]) if max_fit > 0 else np.full(pop_size, 1.0 / pop_size)
            probs = probs / np.sum(probs)

            for _ in range(pop_size):
                i = np.random.choice(pop_size, p=probs)
                k = random.choice([idx for idx in range(pop_size) if idx != i])
                j = random.randint(0, n - 1)
                phi = random.uniform(-1.0, 1.0)

                X_new = posiciones[i].copy()
                X_new[j] = X_new[j] + (X_new[j] - posiciones[k][j]) * phi
                X_new = np.clip(X_new, -params.v_max, params.v_max)

                sol_prop, val_prop = binarizar_posicion(
                    X_new, poblacion_bin[i], inst, G1, G2, G3, params.v_max
                )

                if val_prop >= fitnesses[i]:
                    posiciones[i] = X_new
                    poblacion_bin[i] = sol_prop
                    fitnesses[i] = val_prop
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Scout bee
            for i in range(pop_size):
                if trials[i] >= limit:
                    posiciones[i] = np.random.uniform(-params.v_max, params.v_max, size=n)
                    sol_rep, val_rep = reparar_solucion([random.randint(0, 1) for _ in range(n)], inst)
                    poblacion_bin[i] = sol_rep
                    fitnesses[i] = val_rep
                    trials[i] = 0

        iter_best_idx = max(range(pop_size), key=lambda idx: fitnesses[idx])
        fit_iter_best = fitnesses[iter_best_idx]

        if fit_iter_best > mejor_val:
            mejor_val = fit_iter_best
            mejor_sol = poblacion_bin[iter_best_idx].copy()
            best_pos = posiciones[iter_best_idx].copy()

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)

        if verbose:
            fase_str = "WOA" if div_norm <= umbral_t else "ABC"
            print(f"  [Var C MKP] Iter {it+1:3d}/{params.iterations} | Fase: {fase_str:<3} | DivNorm: {div_norm:.3f} | Mejor: {mejor_val:10.1f} | IterBest: {fit_iter_best:10.1f}")

        if monitor is not None:
            status = monitor.update(mejor_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [WOA-ABC-C Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return VariantCEpochResult(
        epoch_idx = epoch_idx,
        mejor_valor = mejor_val,
        iteraciones = len(historial),
        stagnation_fires = stag_fires,
        historial = historial,
        historial_inst = historial_inst,
        historial_diversidad = historial_div,
        mejor_solucion = mejor_sol,
        dtw_deltas = dtw_deltas,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOMINIO CONTINUO (CEC2022 / Minimización)
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_epoch_continuo(
    func,
    params : VariantCParams,
    epoch_idx : int = 0,
    verbose : bool = True,
    sol_inyectada : np.ndarray | None = None,
) -> VariantCEpochResult:

    n_dim = func.n_dim
    lb, ub = func.lb, func.ub
    pop_size = params.pop_size
    limit = params.limit if params.limit is not None else int(pop_size * n_dim / 2)

    posiciones = np.random.uniform(lb, ub, size=(pop_size, n_dim))
    obj_vals = np.array([func.func(p) for p in posiciones])
    trials = np.zeros(pop_size, dtype=int)

    if sol_inyectada is not None:
        sol_rep = np.clip(sol_inyectada, lb, ub)
        peor_idx = np.argmax(obj_vals)
        posiciones[peor_idx] = sol_rep
        obj_vals[peor_idx] = func.func(sol_rep)
        trials[peor_idx] = 0

    div_0 = max(1e-12, _calcular_diversidad(posiciones))

    gbest_idx = np.argmin(obj_vals)
    gbest_val = float(obj_vals[gbest_idx])
    gbest_pos = posiciones[gbest_idx].copy()

    historial = []
    historial_inst = []
    historial_div = []
    dtw_deltas = []
    stag_fires = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        div_t = _calcular_diversidad(posiciones)
        div_norm = div_t / div_0
        historial_div.append(div_norm)

        umbral_t = params.umbral_init * (1.0 - it / max(1, params.iterations)) + params.umbral_final * (it / max(1, params.iterations))
        a = 2.0 - it * (2.0 / params.iterations)

        if div_norm <= umbral_t:
            for i in range(pop_size):
                r1, r2 = np.random.random(n_dim), np.random.random(n_dim)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = random.random()
                l = random.uniform(-1.0, 1.0)

                if p < 0.5 and np.linalg.norm(A) / np.sqrt(n_dim) >= 1.0:
                    X_rand = posiciones[random.randint(0, pop_size - 1)]
                    D = np.abs(C * X_rand - posiciones[i])
                    X_new = X_rand - A * D
                elif p < 0.5 and np.linalg.norm(A) / np.sqrt(n_dim) < 1.0:
                    D = np.abs(C * gbest_pos - posiciones[i])
                    X_new = gbest_pos - A * D
                else:
                    D_prime = np.abs(gbest_pos - posiciones[i])
                    X_new = D_prime * np.exp(params.b_spiral * l) * np.cos(2.0 * np.pi * l) + gbest_pos

                X_new = np.clip(X_new, lb, ub)
                f_v = func.func(X_new)

                if f_v <= obj_vals[i]:
                    posiciones[i] = X_new
                    obj_vals[i] = f_v
                    trials[i] = 0
                else:
                    trials[i] += 1
        else:
            for i in range(pop_size):
                k = random.choice([idx for idx in range(pop_size) if idx != i])
                j = random.randint(0, n_dim - 1)
                phi = random.uniform(-1.0, 1.0)

                X_new = posiciones[i].copy()
                X_new[j] = X_new[j] + (X_new[j] - posiciones[k][j]) * phi
                X_new = np.clip(X_new, lb, ub)
                f_v = func.func(X_new)

                if f_v <= obj_vals[i]:
                    posiciones[i] = X_new
                    obj_vals[i] = f_v
                    trials[i] = 0
                else:
                    trials[i] += 1

            fits = np.array([1.0 / (1.0 + v) if v >= 0 else 1.0 + abs(v) for v in obj_vals])
            max_fit = np.max(fits)
            probs = 0.9 * (fits / max_fit) + 0.1 if max_fit > 0 else np.full(pop_size, 1.0 / pop_size)
            probs = probs / np.sum(probs)

            for _ in range(pop_size):
                i = np.random.choice(pop_size, p=probs)
                k = random.choice([idx for idx in range(pop_size) if idx != i])
                j = random.randint(0, n_dim - 1)
                phi = random.uniform(-1.0, 1.0)

                X_new = posiciones[i].copy()
                X_new[j] = X_new[j] + (X_new[j] - posiciones[k][j]) * phi
                X_new = np.clip(X_new, lb, ub)
                f_v = func.func(X_new)

                if f_v <= obj_vals[i]:
                    posiciones[i] = X_new
                    obj_vals[i] = f_v
                    trials[i] = 0
                else:
                    trials[i] += 1

            for i in range(pop_size):
                if trials[i] >= limit:
                    posiciones[i] = np.random.uniform(lb, ub, size=n_dim)
                    obj_vals[i] = func.func(posiciones[i])
                    trials[i] = 0

        iter_best_idx = np.argmin(obj_vals)
        fit_iter_best = float(obj_vals[iter_best_idx])

        if fit_iter_best < gbest_val:
            gbest_val = fit_iter_best
            gbest_pos = posiciones[iter_best_idx].copy()

        historial.append(gbest_val)
        historial_inst.append(fit_iter_best)

        if verbose:
            fase_str = "WOA" if div_norm <= umbral_t else "ABC"
            print(f"  [Var C CEC2022] Iter {it+1:3d}/{params.iterations} | Fase: {fase_str:<3} | DivNorm: {div_norm:.3f} | Mejor(Min): {gbest_val:12.4f} | IterBest: {fit_iter_best:12.4f}")

        if monitor is not None:
            status = monitor.update(-gbest_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [WOA-ABC-C Continuous Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return VariantCEpochResult(
        epoch_idx = epoch_idx,
        mejor_valor = gbest_val,
        iteraciones = len(historial),
        stagnation_fires = stag_fires,
        historial = historial,
        historial_inst = historial_inst,
        historial_diversidad = historial_div,
        mejor_solucion = gbest_pos.tolist(),
        dtw_deltas = dtw_deltas,
    )


if __name__ == "__main__":
    from mkp_core.data_loader import cargar_instancias
    from continuous_benchmark.funciones_cec2022 import get_test_functions

    print("=== Demo WOA-ABC (Variante C) ===")
    inst = MKPInstance.from_dict(cargar_instancias('instancias/mknapcb1.txt')[0])
    res_mkp = ejecutar_epoch(inst, VariantCParams(iterations=30), verbose=True)
    print(f"[MKP] Mejor Valor: {res_mkp.mejor_valor:.1f} | Iteraciones: {res_mkp.iteraciones}")

    func = get_test_functions(10)[0]
    res_cont = ejecutar_epoch_continuo(func, VariantCParams(iterations=30), verbose=True)
    print(f"[CEC2022] Mejor Valor: {res_cont.mejor_valor:.4f} | Iteraciones: {res_cont.iteraciones}")

