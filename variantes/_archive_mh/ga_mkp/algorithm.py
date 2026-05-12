"""
ga_mkp/algorithm.py
-------------------
Motor principal del Algoritmo Genetico (GA) para el MKP.

Implementa el bucle evolutivo: inicializacion de poblacion, seleccion
por torneo, cruce y mutacion (con operadores configurables), elitismo
y la integracion del StagnationMonitor con soporte para 9 variantes
de rescate.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from mkp_core.problem    import MKPInstance
from mkp_core.repair     import reparar_solucion
from ga_mkp.operators  import (
    torneo,
    crossover_uniform, crossover_1point, crossover_2point,
    mutate_bitflip, mutate_swap,
    get_crossover, get_mutation,
    CROSSOVER_OPS, MUTATION_OPS,
)
from ga_mkp.rescues    import (
    hill_climb_elite,
    inyectar_heuristicos,
    mutacion_lp,
    ruin_and_recreate_elite,
)
from dtw_stagnation    import StagnationConfig, StagnationMonitor


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class GAParams:
    """Hiperparametros del GA."""
    pop_size        : int   = 60
    generations     : int   = 300
    epochs          : int   = 10
    elitism         : int   = 2
    tournament_size : int   = 3
    crossover_rate  : float = 0.85
    mutation_rate   : float = 0.04
    crossover_op    : str   = "uniform"   # "uniform" | "1point" | "2point"
    mutation_op     : str   = "bitflip"   # "bitflip" | "swap"
    # Inyección de solución (pipeline híbrido)
    injection_mode  : str   = "random"    # "random" | "mutated" | "mixed"
    # Stagnation
    use_stagnation  : bool  = True
    stag_cfg        : StagnationConfig | None = None
    stag_strategy   : str   = "hypermutation"
    stag_max_fires  : int   = 3


@dataclass
class GAEpochResult:
    """Resultado de una epoch del GA."""
    epoch_idx       : int
    mejor_valor     : float
    generaciones    : int
    stagnation_fires: int
    historial       : list[float] = field(default_factory=list)
    mejor_solucion  : list[int]  = field(default_factory=list)   # mejor por generacion


@dataclass
class GAResult:
    """Resultado completo del GA (todas las epochs)."""
    epochs               : list[GAEpochResult]
    mejor_valor_global   : float
    mejor_sol_global     : list[int]
    valor_optimo         : float

    @property
    def gap_pct(self) -> float | None:
        if self.valor_optimo == 0:
            return None
        return 100.0 * (self.valor_optimo - self.mejor_valor_global) / self.valor_optimo

    @property
    def valores_por_epoch(self) -> list[float]:
        return [ep.mejor_valor for ep in self.epochs]


# ── Helpers de población ──────────────────────────────────────────────────────

def _inicializar_poblacion(inst: MKPInstance, pop_size: int) -> tuple[list[list[int]], list[float]]:
    """Genera `pop_size` individuos aleatorios factibles."""
    poblacion = []
    fitnesses = []
    for _ in range(pop_size):
        sol = [random.randint(0, 1) for _ in range(inst.n)]
        sol, val = reparar_solucion(sol, inst)
        poblacion.append(sol)
        fitnesses.append(val)
    return poblacion, fitnesses


def _mutar_solucion(sol: list[int], inst: MKPInstance, n_flips: int = 0) -> tuple[list[int], float]:
    """Crea una copia de `sol` con entre 1 y `n_flips` bits invertidos al azar y la repara."""
    copia = list(sol)
    n = len(copia)
    if n_flips <= 0:
        n_flips = random.randint(1, 3)
    indices = random.sample(range(n), min(n_flips, n))
    for idx in indices:
        copia[idx] = 1 - copia[idx]
    copia, val = reparar_solucion(copia, inst)
    return copia, val


# ── Una epoch del GA ─────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst      : MKPInstance,
    params    : GAParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: list[int] | None = None,
) -> GAEpochResult:
    """Ejecuta una epoch completa del GA y devuelve el resultado."""

    # Inicializar poblacion
    poblacion, fitnesses = _inicializar_poblacion(inst, params.pop_size)

    # Inyectar solución del orquestador según el modo de inyección
    if sol_inyectada is not None:
        sol_rep = list(sol_inyectada)
        sol_rep, val_rep = reparar_solucion(sol_rep, inst)
        mode = params.injection_mode

        if mode == "random":
            # Reemplazar al peor individuo con la solución inyectada
            peor_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            poblacion[peor_idx] = sol_rep
            fitnesses[peor_idx] = val_rep

        elif mode == "mutated":
            # Toda la población se genera a partir de mutaciones de la inyectada
            poblacion[0] = sol_rep
            fitnesses[0] = val_rep
            for i in range(1, params.pop_size):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion[i] = msol
                fitnesses[i] = mval

        elif mode == "mixed":
            # Mitad mutaciones de la inyectada, mitad aleatorios
            poblacion[0] = sol_rep
            fitnesses[0] = val_rep
            n_mutados = params.pop_size // 2
            for i in range(1, n_mutados):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion[i] = msol
                fitnesses[i] = mval
            # El resto (desde n_mutados) ya es aleatorio de _inicializar_poblacion

    mejor_idx  = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    mejor_val  = fitnesses[mejor_idx]
    mejor_sol  = poblacion[mejor_idx].copy()

    historial   = []
    stag_fires  = 0

    # Estado dinamico de operadores (puede cambiar con la variante)
    cx_name      = params.crossover_op
    mut_name     = params.mutation_op
    mut_rate     = params.mutation_rate
    v2_exploring = True   # Para V2 cyclic alternation
    tabu_genes   = []     # Para V7

    # Inicializar monitor
    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    if verbose:
        print(f"\n== Epoch {epoch_idx + 1}/{params.epochs} ====================")

    for gen in range(params.generations):
        # -- Elitismo: los mejores pasan directamente ──────────────────────────
        elite_idx  = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:params.elitism]
        nueva_pob  = [poblacion[i].copy() for i in elite_idx]
        nuevos_fit = [fitnesses[i]        for i in elite_idx]

        # -- Reproduccion hasta completar la poblacion ─────────────────────────
        fn_cx  = get_crossover(cx_name)
        fn_mut = get_mutation(mut_name)

        while len(nueva_pob) < params.pop_size:
            padre_a = torneo(poblacion, fitnesses, params.tournament_size)
            padre_b = torneo(poblacion, fitnesses, params.tournament_size)

            # Cruce
            if random.random() < params.crossover_rate:
                hijo_a, hijo_b = fn_cx(padre_a, padre_b)
            else:
                hijo_a, hijo_b = padre_a.copy(), padre_b.copy()

            # Mutacion
            hijo_a = fn_mut(hijo_a, mut_rate)
            hijo_b = fn_mut(hijo_b, mut_rate)

            # Reparar para garantizar factibilidad
            hijo_a, val_a = reparar_solucion(hijo_a, inst)
            hijo_b, val_b = reparar_solucion(hijo_b, inst)

            nueva_pob  += [hijo_a, hijo_b]
            nuevos_fit += [val_a,  val_b]

        # Truncar al tamano exacto
        poblacion = nueva_pob[:params.pop_size]
        fitnesses = nuevos_fit[:params.pop_size]

        # -- Actualizar mejor ──────────────────────────────────────────────────
        mejor_gen_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[mejor_gen_idx] > mejor_val:
            mejor_val = fitnesses[mejor_gen_idx]
            mejor_sol = poblacion[mejor_gen_idx].copy()

        historial.append(mejor_val)

        # -- Stagnation check (una vez por generacion) ─────────────────────────
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
                    f"iter={gen:03d}",
                    f"n={ns}",
                    f"D1={d1:.3f}",
                    f"D2={d2:.3f}",
                    f"Δ={dlt:.3f}",
                    f"theta={td}",
                    f"θc={tc:.2f}",
                    f"θr={tr:.2f}",
                    f"no_improve={ni}",
                    f"fire={fr}",
                    f"best={mejor_val:.1f}",
                )

            if status.get("fire"):
                max_f = params.stag_max_fires
                if max_f > 0 and stag_fires >= max_f:
                    pass  # Limite de fires alcanzado
                else:
                    stag_fires += 1
                    monitor.reset()

                    s = params.stag_strategy
                    if verbose:
                        print(f"    [Stagnation] Fire #{stag_fires} @ gen {gen + 1} -> {s}")

                    if s == "abort":
                        break  # Sale del for gen

                    elif s == "hypermutation":
                        # Sube temporalmente la tasa de mutacion
                        mut_rate = min(0.20, params.mutation_rate * 5)

                    elif s == "v1_exploit":
                        # Cruce 1point + Hill Climbing local
                        cx_name  = "1point"
                        mut_rate = 0.0
                        poblacion, fitnesses = hill_climb_elite(poblacion, fitnesses, inst, 0.10)

                    elif s == "v2_cycle":
                        # Alterna entre exploracion y explotacion
                        if v2_exploring:
                            cx_name      = "2point"
                            mut_name     = "swap"
                            mut_rate     = params.mutation_rate * 3
                            v2_exploring = False
                        else:
                            cx_name      = "uniform"
                            mut_name     = "bitflip"
                            mut_rate     = params.mutation_rate * 0.5
                            v2_exploring = True

                    elif s == "v3_explore":
                        # Cruce 2point + swap + inmigrantes masivos
                        cx_name  = "2point"
                        mut_name = "swap"
                        mut_rate = params.mutation_rate * 2
                        # 50% de la poblacion se reinicia aleatoriamente
                        n_inmig = params.pop_size // 2
                        peores  = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:n_inmig]
                        for idx in peores:
                            sol = [random.randint(0, 1) for _ in range(inst.n)]
                            sol, val       = reparar_solucion(sol, inst)
                            poblacion[idx] = sol
                            fitnesses[idx] = val

                    elif s == "v4_nonlinear":
                        # Tasa de mutacion crece exponencialmente con los fires
                        mut_rate = params.mutation_rate * math.exp(0.7 * stag_fires)
                        if mut_rate > 0.35:   # si es muy alta, cambia a swap
                            mut_name = "swap"

                    elif s == "v5_heuristic":
                        poblacion, fitnesses = inyectar_heuristicos(poblacion, fitnesses, inst, 0.50)

                    elif s == "v6_lp":
                        poblacion, fitnesses = mutacion_lp(poblacion, fitnesses, inst, None)

                    elif s == "v7_tabu_lp":
                        poblacion, fitnesses = mutacion_lp(poblacion, fitnesses, inst, tabu_genes)
                        # Extender lista tabu con genes nuevos
                        tabu_genes.extend(random.sample(range(inst.n), min(5, inst.n)))
                        if len(tabu_genes) > 25:
                            tabu_genes = tabu_genes[-25:]

                    elif s == "v8_ruin_recreate":
                        poblacion, fitnesses = ruin_and_recreate_elite(poblacion, fitnesses, inst, 0.30)

                    # Despues de un rescate agresivo, restaurar operadores a default
                    # en los casos que no son ciclicos
                    if s not in ("v2_cycle", "v4_nonlinear"):
                        cx_name  = params.crossover_op
                        mut_name = params.mutation_op
                        if s not in ("v1_exploit",):
                            mut_rate = params.mutation_rate

    # Fin de la epoch
    if verbose:
        print(
            f"  [Epoch {epoch_idx + 1}] "
            f"gen={params.generations} | mejor={mejor_val:.1f} | "
            f"cx={params.crossover_op} mut={params.mutation_op} | "
            f"stag_fires={stag_fires}"
        )

    return GAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        generaciones     = params.generations,
        stagnation_fires = stag_fires,
        historial        = historial,
        mejor_solucion   = mejor_sol,
    )


# ── Ejecucion completa (todas las epochs) ────────────────────────────────────

def ejecutar_ga(
    inst    : MKPInstance,
    params  : GAParams,
    verbose : bool = True,
) -> GAResult:
    """Ejecuta el GA completo (todas las epochs) y retorna el GAResult."""
    epochs_result       = []
    mejor_val_global    = -float("inf")
    mejor_sol_global    = []

    for e in range(params.epochs):
        epoch_res = ejecutar_epoch(inst, params, epoch_idx=e, verbose=verbose)
        epochs_result.append(epoch_res)

        if epoch_res.mejor_valor > mejor_val_global:
            mejor_val_global = epoch_res.mejor_valor
            mejor_sol_global = []   # No guardamos la sol entera por ahora

    return GAResult(
        epochs             = epochs_result,
        mejor_valor_global = mejor_val_global,
        mejor_sol_global   = mejor_sol_global,
        valor_optimo       = inst.valor_optimo,
    )
