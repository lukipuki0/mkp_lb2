"""
mh/ga.py
--------
Algoritmo Genético (GA) para el MKP.

Versión limpia para el pipeline híbrido: solo usa estrategia "abort"
cuando el monitor DTW detecta estancamiento.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mkp_core.problem   import MKPInstance
from mkp_core.repair    import reparar_solucion
from mh.ga_operators    import torneo, get_crossover, get_mutation
from dtw_stagnation     import StagnationConfig, StagnationMonitor


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class GAParams:
    """Hiperparámetros del GA."""
    pop_size        : int   = 60
    generations     : int   = 300
    epochs          : int   = 10
    elitism         : int   = 2
    tournament_size : int   = 3
    crossover_rate  : float = 0.85
    mutation_rate   : float = 0.04
    crossover_op    : str   = "uniform"    # "uniform" | "1point" | "2point"
    mutation_op     : str   = "bitflip"    # "bitflip" | "swap"
    # Inyección de solución (pipeline híbrido)
    injection_mode  : str   = "random"     # "random" | "mutated" | "mixed"
    # Stagnation
    use_stagnation  : bool  = True
    stag_cfg        : StagnationConfig | None = None


@dataclass
class GAEpochResult:
    """Resultado de un epoch del GA."""
    epoch_idx       : int
    mejor_valor     : float
    generaciones    : int
    stagnation_fires: int
    historial       : list[float] = field(default_factory=list)
    historial_inst  : list[float] = field(default_factory=list)  # fitness best de cada gen
    mejor_solucion  : list[int]  = field(default_factory=list)
    dtw_deltas      : list[float] = field(default_factory=list)


@dataclass
class GAResult:
    """Resultado completo del GA (todos los epochs)."""
    epochs             : list[GAEpochResult]
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


# ── Epoch individual ─────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst      : MKPInstance,
    params    : GAParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: list[int] | None = None,
) -> GAEpochResult:
    """Ejecuta un epoch completo del GA con detección de estancamiento (abort)."""

    # Inicializar población
    poblacion, fitnesses = _inicializar_poblacion(inst, params.pop_size)

    # Inyectar solución del orquestador según el modo de inyección
    if sol_inyectada is not None:
        sol_rep = list(sol_inyectada)
        sol_rep, val_rep = reparar_solucion(sol_rep, inst)
        mode = params.injection_mode

        if mode == "random":
            peor_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            poblacion[peor_idx] = sol_rep
            fitnesses[peor_idx] = val_rep

        elif mode == "mutated":
            poblacion[0] = sol_rep
            fitnesses[0] = val_rep
            for i in range(1, params.pop_size):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion[i] = msol
                fitnesses[i] = mval

        elif mode == "mixed":
            poblacion[0] = sol_rep
            fitnesses[0] = val_rep
            n_mutados = params.pop_size // 2
            for i in range(1, n_mutados):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion[i] = msol
                fitnesses[i] = mval

    mejor_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    mejor_val = fitnesses[mejor_idx]
    mejor_sol = poblacion[mejor_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0

    fn_cx  = get_crossover(params.crossover_op)
    fn_mut = get_mutation(params.mutation_op)

    # Inicializar monitor
    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for gen in range(params.generations):
        # -- Elitismo --
        elite_idx  = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:params.elitism]
        nueva_pob  = [poblacion[i].copy() for i in elite_idx]
        nuevos_fit = [fitnesses[i]        for i in elite_idx]

        # -- Reproducción --
        while len(nueva_pob) < params.pop_size:
            padre_a = torneo(poblacion, fitnesses, params.tournament_size)
            padre_b = torneo(poblacion, fitnesses, params.tournament_size)

            # Cruce
            if random.random() < params.crossover_rate:
                hijo_a, hijo_b = fn_cx(padre_a, padre_b)
            else:
                hijo_a, hijo_b = padre_a.copy(), padre_b.copy()

            # Mutación
            hijo_a = fn_mut(hijo_a, params.mutation_rate)
            hijo_b = fn_mut(hijo_b, params.mutation_rate)

            # Reparar
            hijo_a, val_a = reparar_solucion(hijo_a, inst)
            hijo_b, val_b = reparar_solucion(hijo_b, inst)

            nueva_pob  += [hijo_a, hijo_b]
            nuevos_fit += [val_a,  val_b]

        # Truncar al tamaño exacto
        poblacion = nueva_pob[:params.pop_size]
        fitnesses = nuevos_fit[:params.pop_size]

        # -- Actualizar mejor --
        mejor_gen_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        fit_gen_actual = fitnesses[mejor_gen_idx]  # mejor fitness de ESTA generacion
        if fit_gen_actual > mejor_val:
            mejor_val = fit_gen_actual
            mejor_sol = poblacion[mejor_gen_idx].copy()

        historial.append(mejor_val)
        historial_inst.append(fit_gen_actual)  # puede ser <= mejor_val

        # -- Stagnation check --
        if monitor is not None:
            status = monitor.update(mejor_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))

            if verbose and status.get("ready"):
                dlt = status.get("delta", 0.0)
                td  = status.get("theta_delta", 0.0)
                if dlt > td: estado = "Explorar mucho"
                elif 0 <= dlt <= td: estado = "Explorar poco"
                elif -td <= dlt < 0: estado = "Explotar poco"
                else: estado = "Explotar mucho"
                print(f"i={gen:03d} | Estado: {estado:<15} | Delta={dlt:6.1f} | Th_d={td:6.1f} | d1={status.get('D1_vs_ramp', 0.0):.3f} | d2={status.get('D2_vs_const', 0.0):.3f} | best={mejor_val:.1f}")

            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [Stagnation] Fire #{stag_fires} @ gen {gen + 1} -> ABORT")
                break

    return GAEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        generaciones     = params.generations,
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
    )


# ── Ejecución multi-epoch ────────────────────────────────────────────────────

def ejecutar_ga(
    inst: MKPInstance,
    params: GAParams,
    verbose: bool = True,
) -> GAResult:
    """Ejecuta el GA completo (todos los epochs) y retorna el GAResult."""
    epochs_result    = []
    mejor_val_global = -float("inf")
    mejor_sol_global: list[int] = []

    for e in range(params.epochs):
        epoch_res = ejecutar_epoch(inst, params, epoch_idx=e, verbose=verbose)
        epochs_result.append(epoch_res)

        if epoch_res.mejor_valor > mejor_val_global:
            mejor_val_global = epoch_res.mejor_valor
            mejor_sol_global = epoch_res.mejor_solucion.copy()

    return GAResult(
        epochs             = epochs_result,
        mejor_valor_global = mejor_val_global,
        mejor_sol_global   = mejor_sol_global,
        valor_optimo       = inst.valor_optimo,
    )

