"""
mh/aco.py
---------
Ant Colony Optimization (ACO / Max-Min Ant System) para el MKP.

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


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class ACOParams:
    """Hiperparámetros del Ant Colony Optimization (ACO)."""
    pop_size       : int   = 30     # Número de hormigas por iteración
    iterations     : int   = 300
    epochs         : int   = 10
    alpha          : float = 1.0    # Importancia del rastro de feromonas
    beta           : float = 2.0    # Importancia de la información heurística (pseudo-utilidad)
    rho            : float = 0.1    # Tasa de evaporación de feromonas (0 < rho < 1)
    tau_min        : float = 0.1    # Límite inferior de feromona (MMAS)
    tau_max        : float = 5.0    # Límite superior de feromona (MMAS)
    # Inyección de solución (pipeline híbrido)
    injection_mode : str  = "mixed" # "random" | "mutated" | "mixed"
    # Stagnation
    use_stagnation : bool = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class ACOEpochResult:
    """Resultado de un epoch del ACO."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)  # mejor fitness de la iteración
    mejor_solucion   : list[int]   = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)
    dtw_info_hist    : list[dict]  = field(default_factory=list)


@dataclass
class ACOResult:
    """Resultado completo del ACO (todos los epochs)."""
    epochs             : list[ACOEpochResult]
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

def _calcular_heuristica(inst: MKPInstance) -> np.ndarray:
    """Calcula la pseudo-utilidad multiconstricción (eta_j) para cada ítem."""
    n = inst.n
    m = inst.m
    beneficios = np.array(inst.p, dtype=float)
    pesos = np.array(inst.r, dtype=float)       # shape (m, n)
    capacidades = np.array(inst.b, dtype=float) # shape (m,)

    # Relación de peso relativo consumido por cada dimensión
    # sum_i (A_{i,j} / b_i)
    peso_relativo = np.sum(pesos / capacidades[:, np.newaxis], axis=0)
    peso_relativo = np.maximum(peso_relativo, 1e-9)

    eta = beneficios / peso_relativo
    eta_max = np.max(eta)
    if eta_max > 0:
        eta = eta / eta_max
    return eta


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


def _construir_solucion_hormiga(
    inst: MKPInstance,
    tau: np.ndarray,
    eta: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[list[int], float]:
    """Construye una solución probabilística para una hormiga basada en feromonas y heurística."""
    n = inst.n
    m = inst.m
    sol = [0] * n
    consumo = np.zeros(m, dtype=float)
    capacidades = np.array(inst.b, dtype=float)
    pesos = np.array(inst.r, dtype=float)

    candidatos = list(range(n))

    while candidatos:
        # Filtrar candidatos que quepan físicamente
        validos = []
        pesos_seleccion = []

        for j in candidatos:
            if np.all(consumo + pesos[:, j] <= capacidades):
                validos.append(j)
                w = (tau[j] ** alpha) * (eta[j] ** beta)
                pesos_seleccion.append(w)

        if not validos:
            break

        # Calcular deseabilidad tau^alpha * eta^beta
        deseabilidades = (tau[candidatos] ** alpha) * (eta[candidatos] ** beta)
        suma_des = np.sum(deseabilidades)

        if suma_des <= 0:
            probs = np.ones(len(candidatos)) / len(candidatos)
        else:
            probs = deseabilidades / suma_des

        # Selección por ruleta
        elem = np.random.choice(candidatos, p=probs)
        sol[elem] = 1
        candidatos.remove(elem)

    # Reparar la solución completa para garantizar factibilidad y optimizar peso sobrante
    sol_rep, val_rep = reparar_solucion(sol, inst)
    return sol_rep, val_rep


# ── Epoch individual ─────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst          : MKPInstance,
    params        : ACOParams,
    epoch_idx     : int = 0,
    verbose       : bool = True,
    sol_inyectada : list[int] | None = None,
) -> ACOEpochResult:
    """Ejecuta un epoch completo del ACO con detección de estancamiento (abort)."""

    pop_size = params.pop_size
    n = inst.n

    # Matriz/Vector de feromonas e información heurística
    tau = _inicializar_feromonas(n, params.tau_0)
    eta = _calcular_heuristica(inst)

    # Inicialización de la colonia (hormigas)
    poblacion_bin = []
    fitnesses     = []
    for _ in range(pop_size):
        sol_k, val_k = _construir_solucion_hormiga(inst, tau, eta, params.alpha, params.beta)
        poblacion_bin.append(sol_k)
        fitnesses.append(val_k)

    # Inyectar solución del orquestador si se proporciona
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

        # Reforzar feromona inicial en los ítems de la solución inyectada
        for j in range(n):
            if sol_rep[j] == 1:
                tau[j] = min(params.tau_max, tau[j] * 1.5)

    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    mejor_val = fitnesses[best_idx]
    mejor_sol = poblacion_bin[best_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    dtw_info_hist  = []
    stag_fires     = 0

    # Monitor de estancamiento
    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        # 1. Construir nuevas soluciones para cada hormiga
        nuevas_sols = []
        nuevos_vals = []

        for k in range(pop_size):
            sol_k, val_k = _construir_solucion_hormiga(inst, tau, eta, params.alpha, params.beta)
            nuevas_sols.append(sol_k)
            nuevos_vals.append(val_k)

        poblacion_bin = nuevas_sols
        fitnesses = nuevos_vals

        # Actualizar mejor de la iteración y mejor global
        iter_best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        fit_iter_best = fitnesses[iter_best_idx]

        if fit_iter_best > mejor_val:
            mejor_val = fit_iter_best
            mejor_sol = poblacion_bin[iter_best_idx].copy()

        # 2. Actualizar feromonas (Evaporación + Depósito MMAS)
        tau = (1.0 - params.rho) * tau

        # Depósito proporcional a la calidad del mejor global del epoch (o de la iteración)
        valor_base = inst.valor_optimo if inst.valor_optimo > 0 else mejor_val
        deposito = params.rho * (mejor_val / max(1.0, valor_base))
        for j in range(n):
            if mejor_sol[j] == 1:
                tau[j] += deposito

        # Acotar feromonas según Max-Min Ant System (MMAS)
        tau = np.clip(tau, params.tau_min, params.tau_max)

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)

        # ── Stagnation check ──────────────────────────────────────────────
        dtw_info = {}
        if monitor is not None:
            status = monitor.update(mejor_val)
            dtw_info = status.copy()
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))

            if verbose and status.get("ready"):
                dlt = status.get("delta", 0.0)
                td  = status.get("theta_delta", 0.0)
                if dlt > td: estado = "Explorar mucho"
                elif 0 <= dlt <= td: estado = "Explorar poco"
                elif -td <= dlt < 0: estado = "Explotar poco"
                else: estado = "Explotar mucho"
                print(f"i={it:03d} | Estado: {estado:<15} | Delta={dlt:6.1f} | Th_d={td:6.1f} | d1={status.get('D1_vs_ramp', 0.0):.3f} | d2={status.get('D2_vs_const', 0.0):.3f} | best={mejor_val:.1f}")

            if status.get("fire"):
                stag_fires += 1
                dtw_info_hist.append(dtw_info)
                if verbose:
                    print(f"    [Stagnation] Fire #{stag_fires} @ iter {it + 1} -> ABORT")
                break

        dtw_info_hist.append(dtw_info)

    return ACOEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
        dtw_info_hist    = dtw_info_hist,
    )


# ── Ejecución multi-epoch ────────────────────────────────────────────────────

def ejecutar_aco(
    inst: MKPInstance,
    params: ACOParams,
    verbose: bool = True,
) -> ACOResult:
    """Ejecuta el ACO completo (todos los epochs) y retorna el ACOResult."""
    epochs_result    = []
    mejor_val_global = -float("inf")
    mejor_sol_global: list[int] = []

    for e in range(params.epochs):
        epoch_res = ejecutar_epoch(inst, params, epoch_idx=e, verbose=verbose)
        epochs_result.append(epoch_res)

        if epoch_res.mejor_valor > mejor_val_global:
            mejor_val_global = epoch_res.mejor_valor
            mejor_sol_global = epoch_res.mejor_solucion.copy()

    return ACOResult(
        epochs             = epochs_result,
        mejor_valor_global = mejor_val_global,
        mejor_sol_global   = mejor_sol_global,
        valor_optimo       = inst.valor_optimo,
    )
