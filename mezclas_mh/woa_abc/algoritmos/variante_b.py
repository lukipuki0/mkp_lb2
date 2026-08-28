"""
mezclas_mh/woa_abc/variante_b.py
--------------------------------
Variante B: Momentum-Guided (memoria histórica) con paso adaptativo.

Ecuaciones Clave:
  Momentum(t) = BestPos(t-1) - BestPos(t-2)
  M(t) = beta * Momentum(t)
  step(t) = step_init * (step_final / step_init) ** (t / MaxIt)
  
WOA update + step(t) * M(t)
ABC update: x_new[j] = x[j] + (x[j] - x_k[j])*phi + step(t) * beta * Momentum[j](t)
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
class VariantBParams:
    pop_size : int = 30
    iterations : int = 300
    epochs : int = 1
    v_max : float = 6.0
    limit : int | None = None
    b_spiral : float = 1.0
    beta : float = 0.5            # Factor de momentum
    step_init : float = 1.0       # Paso inicial
    step_final : float = 0.01     # Paso final
    GP : float = 0.5              # Probabilidad de fase WOA vs ABC
    # LB2 params
    G1_i : float = 0.5; G1_f : float = 1.0
    G2_i : float = 0.5; G2_f : float = 7.2
    G3_i : float = 0.5; G3_f : float = 0.0
    injection_mode : str = "mixed"
    use_stagnation : bool = True
    stag_cfg : StagnationConfig | None = None


@dataclass
class VariantBEpochResult:
    epoch_idx : int
    mejor_valor : float
    iteraciones : int
    stagnation_fires : int
    historial : list[float] = field(default_factory=list)
    historial_inst : list[float] = field(default_factory=list)
    mejor_solucion : list = field(default_factory=list)
    dtw_deltas : list[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOMINIO DISCRETO MKP
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst : MKPInstance,
    params : VariantBParams,
    epoch_idx : int = 0,
    verbose : bool = True,
    sol_inyectada : list[int] | None = None,
) -> VariantBEpochResult:

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

    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    mejor_val = fitnesses[best_idx]
    mejor_sol = poblacion_bin[best_idx].copy()
    best_pos = posiciones[best_idx].copy()
    best_pos_prev = best_pos.copy()

    historial = []
    historial_inst = []
    dtw_deltas = []
    stag_fires = 0

    G1, G2, G3 = params.G1_i, params.G2_i, params.G3_i

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        step_t = params.step_init * ((params.step_final / max(1e-12, params.step_init)) ** (it / max(1, params.iterations)))
        momentum_vec = best_pos - best_pos_prev
        M_t = params.beta * momentum_vec
        best_pos_prev = best_pos.copy()

        a = 2.0 - it * (2.0 / params.iterations)

        # ══ FASE WOA CON MOMENTUM ══
        for i in range(pop_size):
            if random.random() < params.GP:
                r1, r2 = np.random.random(n), np.random.random(n)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = random.random()
                l = random.uniform(-1.0, 1.0)

                if p < 0.5 and np.linalg.norm(A) / np.sqrt(n) >= 1.0:
                    X_rand = posiciones[random.randint(0, pop_size - 1)]
                    D = np.abs(C * X_rand - posiciones[i])
                    X_new = X_rand - A * D + step_t * M_t
                elif p < 0.5 and np.linalg.norm(A) / np.sqrt(n) < 1.0:
                    D = np.abs(C * best_pos - posiciones[i])
                    X_new = best_pos - A * D + step_t * M_t
                else:
                    D_prime = np.abs(best_pos - posiciones[i])
                    X_new = D_prime * np.exp(params.b_spiral * l) * np.cos(2.0 * np.pi * l) + best_pos + step_t * M_t

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

        # ══ FASE ABC CON MOMENTUM (Employed Bee) ══
        for i in range(pop_size):
            k = random.choice([idx for idx in range(pop_size) if idx != i])
            j = random.randint(0, n - 1)
            phi = random.uniform(-1.0, 1.0)

            X_new = posiciones[i].copy()
            X_new[j] = X_new[j] + (X_new[j] - posiciones[k][j]) * phi + step_t * params.beta * momentum_vec[j]
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

        # Onlooker Bee con momentum
        max_fit = max(fitnesses)
        probs = np.array([0.9 * (fit / max_fit) + 0.1 for fit in fitnesses]) if max_fit > 0 else np.full(pop_size, 1.0 / pop_size)
        probs = probs / np.sum(probs)

        for _ in range(pop_size):
            i = np.random.choice(pop_size, p=probs)
            k = random.choice([idx for idx in range(pop_size) if idx != i])
            j = random.randint(0, n - 1)
            phi = random.uniform(-1.0, 1.0)

            X_new = posiciones[i].copy()
            X_new[j] = X_new[j] + (X_new[j] - posiciones[k][j]) * phi + step_t * params.beta * momentum_vec[j]
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

        # Scout Bee
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
            print(f"  [Var B MKP] Iter {it+1:3d}/{params.iterations} | Step: {step_t:.4f} | Mejor: {mejor_val:10.1f} | IterBest: {fit_iter_best:10.1f}")

        if monitor is not None:
            status = monitor.update(mejor_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [WOA-ABC-B Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return VariantBEpochResult(
        epoch_idx = epoch_idx,
        mejor_valor = mejor_val,
        iteraciones = len(historial),
        stagnation_fires = stag_fires,
        historial = historial,
        historial_inst = historial_inst,
        mejor_solucion = mejor_sol,
        dtw_deltas = dtw_deltas,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOMINIO CONTINUO (CEC2022 / Minimización)
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_epoch_continuo(
    func,
    params : VariantBParams,
    epoch_idx : int = 0,
    verbose : bool = True,
    sol_inyectada : np.ndarray | None = None,
) -> VariantBEpochResult:

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

    gbest_idx = np.argmin(obj_vals)
    gbest_val = float(obj_vals[gbest_idx])
    gbest_pos = posiciones[gbest_idx].copy()
    gbest_pos_prev = gbest_pos.copy()

    historial = []
    historial_inst = []
    dtw_deltas = []
    stag_fires = 0

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        step_t = params.step_init * ((params.step_final / max(1e-12, params.step_init)) ** (it / max(1, params.iterations)))
        momentum_vec = gbest_pos - gbest_pos_prev
        M_t = params.beta * momentum_vec
        gbest_pos_prev = gbest_pos.copy()

        a = 2.0 - it * (2.0 / params.iterations)

        for i in range(pop_size):
            if random.random() < params.GP:
                r1, r2 = np.random.random(n_dim), np.random.random(n_dim)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = random.random()
                l = random.uniform(-1.0, 1.0)

                if p < 0.5 and np.linalg.norm(A) / np.sqrt(n_dim) >= 1.0:
                    X_rand = posiciones[random.randint(0, pop_size - 1)]
                    D = np.abs(C * X_rand - posiciones[i])
                    X_new = X_rand - A * D + step_t * M_t
                elif p < 0.5 and np.linalg.norm(A) / np.sqrt(n_dim) < 1.0:
                    D = np.abs(C * gbest_pos - posiciones[i])
                    X_new = gbest_pos - A * D + step_t * M_t
                else:
                    D_prime = np.abs(gbest_pos - posiciones[i])
                    X_new = D_prime * np.exp(params.b_spiral * l) * np.cos(2.0 * np.pi * l) + gbest_pos + step_t * M_t

                X_new = np.clip(X_new, lb, ub)
                f_v = func.func(X_new)

                if f_v <= obj_vals[i]:
                    posiciones[i] = X_new
                    obj_vals[i] = f_v
                    trials[i] = 0
                else:
                    trials[i] += 1

        for i in range(pop_size):
            k = random.choice([idx for idx in range(pop_size) if idx != i])
            j = random.randint(0, n_dim - 1)
            phi = random.uniform(-1.0, 1.0)

            X_new = posiciones[i].copy()
            X_new[j] = X_new[j] + (X_new[j] - posiciones[k][j]) * phi + step_t * params.beta * momentum_vec[j]
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
            X_new[j] = X_new[j] + (X_new[j] - posiciones[k][j]) * phi + step_t * params.beta * momentum_vec[j]
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
            print(f"  [Var B CEC2022] Iter {it+1:3d}/{params.iterations} | Step: {step_t:.4f} | Mejor(Min): {gbest_val:12.4f} | IterBest: {fit_iter_best:12.4f}")

        if monitor is not None:
            status = monitor.update(-gbest_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))
            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [WOA-ABC-B Continuous Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break

    return VariantBEpochResult(
        epoch_idx = epoch_idx,
        mejor_valor = gbest_val,
        iteraciones = len(historial),
        stagnation_fires = stag_fires,
        historial = historial,
        historial_inst = historial_inst,
        mejor_solucion = gbest_pos.tolist(),
        dtw_deltas = dtw_deltas,
    )


if __name__ == "__main__":
    from mkp_core.data_loader import cargar_instancias
    from continuous_benchmark.funciones_cec2022 import get_test_functions

    print("=== Demo WOA-ABC (Variante B) ===")
    inst = MKPInstance.from_dict(cargar_instancias('instancias/mknapcb1.txt')[0])
    res_mkp = ejecutar_epoch(inst, VariantBParams(iterations=30), verbose=True)
    print(f"[MKP] Mejor Valor: {res_mkp.mejor_valor:.1f} | Iteraciones: {res_mkp.iteraciones}")

    func = get_test_functions(10)[0]
    res_cont = ejecutar_epoch_continuo(func, VariantBParams(iterations=30), verbose=True)
    print(f"[CEC2022] Mejor Valor: {res_cont.mejor_valor:.4f} | Iteraciones: {res_cont.iteraciones}")

