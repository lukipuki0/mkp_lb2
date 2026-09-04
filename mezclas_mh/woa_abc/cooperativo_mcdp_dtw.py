"""WOA--ABC híbrido con una población compartida para MCDP.

La metodología sigue la idea del MSE-ABC: no existen dos islas separadas.
Hay una única población y las fases de exploración y explotación modifican
esa misma población en secuencia. Aquí WOA ocupa la fase exploratoria y ABC
las fases empleada/observadora/scout. El mejor global y un término de momentum
comunican inmediatamente las fases. DTW se conserva como controlador externo
para adaptar la intensidad de búsqueda cuando hay estancamiento.

Las soluciones MCDP son asignaciones discretas máquina -> celda. Para poder
usar las ecuaciones continuas de WOA y ABC se mantiene una representación
latente continua, que se redondea y repara antes de cada evaluación.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Permite ejecutar este archivo directamente desde Windows o desde una
# terminal ubicada fuera de la raíz del repositorio. Al importarlo como
# paquete no se modifica el path.
if __package__ in {None, ""}:
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from dtw_stagnation import StagnationConfig, StagnationMonitor
from mcdp_core.environment import MCDPEvaluation, MCDP_Instance


@dataclass
class CooperativeMCDPParams:
    """Parámetros del híbrido WOA--ABC con población única."""

    pop_size: int = 30
    iterations: int = 300
    epochs: int = 1

    # WOA / exploración
    b_spiral: float = 1.0
    woa_exploration: float = 1.0

    # ABC / explotación
    limit: int | None = None
    abc_guide_strength: float = 0.20
    abc_phi_scale: float = 1.0

    # Inspirados en la metodología MSE-ABC
    exploration_probability: float = 0.50
    step_initial: float = 1.0
    step_final: float = 0.05
    momentum_factor: float = 0.20

    # DTW
    use_dtw: bool = True
    stag_cfg: StagnationConfig = field(
        default_factory=lambda: StagnationConfig(
            window=75,
            band=0,
            min_slope=0.1,
            plateau_max=15,
            patience=25,
            use_ddtw=True,
            adapt_thresholds=True,
            p_low=30.0,
            p_high=70.0,
        )
    )
    adaptation_cooldown: int = 3
    adaptation_decay: float = 0.90
    max_woa_exploration: float = 2.50
    max_abc_phi_scale: float = 2.50
    min_abc_limit: int = 2

    # Diversidad durante una adaptación DTW
    random_reassignment_rate: float = 0.04
    max_random_reassignment_rate: float = 0.35
    rescue_fraction: float = 0.25
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.pop_size < 2:
            raise ValueError("pop_size debe ser al menos 2")
        if self.iterations < 1:
            raise ValueError("iterations debe ser positivo")
        if self.epochs < 1:
            raise ValueError("epochs debe ser positivo")
        if self.b_spiral <= 0:
            raise ValueError("b_spiral debe ser positivo")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit debe ser positivo")
        if not 0.0 <= self.abc_guide_strength:
            raise ValueError("abc_guide_strength no puede ser negativo")
        if not 0.0 <= self.abc_phi_scale:
            raise ValueError("abc_phi_scale no puede ser negativo")
        if not 0.0 <= self.exploration_probability <= 1.0:
            raise ValueError("exploration_probability debe estar entre 0 y 1")
        if self.step_initial <= 0 or self.step_final <= 0:
            raise ValueError("step_initial y step_final deben ser positivos")
        if self.momentum_factor < 0:
            raise ValueError("momentum_factor no puede ser negativo")
        if not 0.0 <= self.rescue_fraction < 1.0:
            raise ValueError("rescue_fraction debe estar entre 0 y 1")
        if not 0.0 <= self.random_reassignment_rate <= 1.0:
            raise ValueError("random_reassignment_rate debe estar entre 0 y 1")
        if not 0.0 <= self.max_random_reassignment_rate <= 1.0:
            raise ValueError("max_random_reassignment_rate debe estar entre 0 y 1")
        if self.adaptation_cooldown < 0:
            raise ValueError("adaptation_cooldown no puede ser negativo")
        if self.adaptation_decay < 0.0 or self.adaptation_decay > 1.0:
            raise ValueError("adaptation_decay debe estar entre 0 y 1")
        if self.max_woa_exploration < 1.0:
            raise ValueError("max_woa_exploration debe ser al menos 1")
        if self.max_abc_phi_scale < 1.0:
            raise ValueError("max_abc_phi_scale debe ser al menos 1")
        if self.min_abc_limit < 1:
            raise ValueError("min_abc_limit debe ser positivo")


@dataclass
class CooperativeMCDPEpochResult:
    """Traza y resultado de un epoch MCDP."""

    epoch_idx: int
    mejor_costo: float
    iteraciones: int
    stagnation_fires: int
    mejor_solucion: list[int] = field(default_factory=list)
    mejor_evaluacion: MCDPEvaluation | None = None
    historial: list[float] = field(default_factory=list)
    historial_inst: list[float] = field(default_factory=list)
    historial_woa: list[float] = field(default_factory=list)
    historial_abc: list[float] = field(default_factory=list)
    dtw_deltas: list[float] = field(default_factory=list)
    dtw_info_hist: list[dict[str, Any]] = field(default_factory=list)
    eventos_cooperacion: list[dict[str, Any]] = field(default_factory=list)
    eventos_adaptacion: list[dict[str, Any]] = field(default_factory=list)
    parametros_historial: list[dict[str, float | int]] = field(default_factory=list)


@dataclass
class CooperativeMCDPResult:
    """Resultado de todos los epochs configurados."""

    epochs: list[CooperativeMCDPEpochResult]
    mejor_costo_global: float
    mejor_sol_global: list[int]
    evaluacion_global: MCDPEvaluation

    @property
    def gap_pct(self) -> None:
        """MCDP no tiene un óptimo conocido en el archivo de instancias."""

        return None


@dataclass
class _AdaptiveState:
    woa_exploration: float = 1.0
    abc_phi_scale: float = 1.0
    abc_guide_strength: float = 0.20
    abc_limit: int = 2
    random_reassignment_rate: float = 0.04


def _check_capacity(inst: MCDP_Instance) -> None:
    if inst.num_machines > inst.max_cells * inst.max_machines_per_cell:
        raise ValueError(
            "La instancia MCDP no puede ser factible: "
            f"{inst.num_machines} máquinas > "
            f"{inst.max_cells}×{inst.max_machines_per_cell} de capacidad"
        )


def _repair_assignment(
    raw_assignment: Sequence[float | int],
    inst: MCDP_Instance,
    random_rate: float = 0.0,
) -> list[int]:
    """Redondea una solución latente y la repara para respetar capacidades."""

    _check_capacity(inst)
    latent = np.asarray(raw_assignment, dtype=float)
    if latent.ndim != 1 or latent.size != inst.num_machines:
        raise ValueError(f"se esperaban {inst.num_machines} asignaciones")
    latent = np.nan_to_num(
        latent,
        nan=0.0,
        posinf=inst.max_cells - 1,
        neginf=0.0,
    )
    latent = np.clip(latent, 0.0, float(inst.max_cells - 1))
    preferred = np.rint(latent).astype(int)

    order = list(range(inst.num_machines))
    random.shuffle(order)
    assignments = [-1] * inst.num_machines
    loads = np.zeros(inst.max_cells, dtype=int)

    for machine in order:
        preferred_cell = int(preferred[machine])
        if random.random() < random_rate:
            available = np.flatnonzero(loads < inst.max_machines_per_cell)
            if available.size:
                preferred_cell = int(random.choice(available.tolist()))

        if loads[preferred_cell] < inst.max_machines_per_cell:
            chosen = preferred_cell
        else:
            available = np.flatnonzero(loads < inst.max_machines_per_cell)
            if not available.size:
                raise ValueError("no quedan celdas con capacidad disponible")
            distances = np.abs(available - latent[machine])
            chosen = int(available[int(np.argmin(distances))])

        assignments[machine] = chosen
        loads[chosen] += 1

    valid, reason = inst.validate_assignment(assignments)
    if not valid:
        raise RuntimeError(f"la reparación MCDP produjo una solución inválida: {reason}")
    return assignments


def _latent_from_assignment(assignment: Sequence[int], inst: MCDP_Instance) -> np.ndarray:
    values = np.asarray(assignment, dtype=float)
    return np.clip(
        values + np.random.uniform(-0.20, 0.20, size=inst.num_machines),
        0.0,
        float(inst.max_cells - 1),
    )


def _random_individual(
    inst: MCDP_Instance,
    random_rate: float = 0.0,
) -> tuple[np.ndarray, list[int], float]:
    latent = np.random.uniform(0.0, float(inst.max_cells - 1), size=inst.num_machines)
    assignment = _repair_assignment(latent, inst, random_rate=random_rate)
    return latent, assignment, float(inst.objective(assignment))


def _initialise_population(
    inst: MCDP_Instance,
    size: int,
    random_rate: float,
) -> tuple[np.ndarray, list[list[int]], np.ndarray, np.ndarray]:
    latent = np.empty((size, inst.num_machines), dtype=float)
    assignments: list[list[int]] = []
    costs = np.empty(size, dtype=float)
    trials = np.zeros(size, dtype=int)
    for i in range(size):
        latent[i], assignment, cost = _random_individual(inst, random_rate)
        assignments.append(assignment)
        costs[i] = cost
    return latent, assignments, costs, trials


def _inject_solution(
    latent: np.ndarray,
    assignments: list[list[int]],
    costs: np.ndarray,
    trials: np.ndarray,
    incoming: Sequence[int],
    inst: MCDP_Instance,
) -> None:
    valid, reason = inst.validate_assignment(incoming)
    if not valid:
        raise ValueError(f"sol_inyectada no es válida para MCDP: {reason}")
    assignment = [int(value) for value in incoming]
    index = int(np.argmax(costs))
    latent[index] = _latent_from_assignment(assignment, inst)
    assignments[index] = assignment
    costs[index] = float(inst.objective(assignment))
    trials[index] = 0


def _population_best(
    latent: np.ndarray,
    assignments: list[list[int]],
    costs: np.ndarray,
) -> tuple[np.ndarray, list[int], float, int]:
    index = int(np.argmin(costs))
    return (
        latent[index].copy(),
        assignments[index].copy(),
        float(costs[index]),
        index,
    )


def _rescue_worst(
    latent: np.ndarray,
    assignments: list[list[int]],
    costs: np.ndarray,
    trials: np.ndarray,
    inst: MCDP_Instance,
    fraction: float,
    random_rate: float,
) -> int:
    """Reinicia peores individuos, protegiendo siempre al mejor de la población."""

    if fraction <= 0:
        return 0
    leader = int(np.argmin(costs))
    count = max(1, int(len(assignments) * fraction))
    candidates = [int(i) for i in np.argsort(costs)[::-1] if int(i) != leader][:count]
    for index in candidates:
        latent[index], assignment, cost = _random_individual(inst, random_rate)
        assignments[index] = assignment
        costs[index] = cost
        trials[index] = 0
    return len(candidates)


def _relax_state(
    state: _AdaptiveState,
    params: CooperativeMCDPParams,
    base_limit: int,
) -> None:
    decay = params.adaptation_decay
    state.woa_exploration = 1.0 + (state.woa_exploration - 1.0) * decay
    state.abc_phi_scale = 1.0 + (state.abc_phi_scale - 1.0) * decay
    state.abc_guide_strength = params.abc_guide_strength + (
        state.abc_guide_strength - params.abc_guide_strength
    ) * decay
    state.abc_limit = max(
        params.min_abc_limit,
        int(round(base_limit + (state.abc_limit - base_limit) * decay)),
    )
    state.random_reassignment_rate = params.random_reassignment_rate + (
        state.random_reassignment_rate - params.random_reassignment_rate
    ) * decay


def _adapt_after_stagnation(
    state: _AdaptiveState,
    params: CooperativeMCDPParams,
    iteration: int,
    dtw_status: dict[str, Any],
) -> dict[str, Any]:
    before = {
        "woa_exploration": state.woa_exploration,
        "abc_phi_scale": state.abc_phi_scale,
        "abc_guide_strength": state.abc_guide_strength,
        "abc_limit": state.abc_limit,
        "random_reassignment_rate": state.random_reassignment_rate,
    }
    state.woa_exploration = min(
        params.max_woa_exploration,
        state.woa_exploration * 1.35,
    )
    state.abc_phi_scale = min(
        params.max_abc_phi_scale,
        state.abc_phi_scale * 1.30,
    )
    state.abc_guide_strength = max(0.02, state.abc_guide_strength * 0.60)
    state.abc_limit = max(params.min_abc_limit, int(round(state.abc_limit * 0.70)))
    state.random_reassignment_rate = min(
        params.max_random_reassignment_rate,
        max(params.random_reassignment_rate, state.random_reassignment_rate * 1.8),
    )
    return {
        "iteracion": iteration,
        "motivo": "stagnation_fire",
        "dtw_delta": float(dtw_status.get("delta", 0.0)),
        "antes": before,
        "despues": {
            "woa_exploration": state.woa_exploration,
            "abc_phi_scale": state.abc_phi_scale,
            "abc_guide_strength": state.abc_guide_strength,
            "abc_limit": state.abc_limit,
            "random_reassignment_rate": state.random_reassignment_rate,
        },
    }


def _record_parameters(
    history: list[dict[str, float | int]],
    state: _AdaptiveState,
    iteration: int,
    step_size: float,
    momentum_norm: float,
) -> None:
    history.append(
        {
            "iteracion": iteration,
            "step_size": float(step_size),
            "momentum_norm": float(momentum_norm),
            "woa_exploration": float(state.woa_exploration),
            "abc_phi_scale": float(state.abc_phi_scale),
            "abc_guide_strength": float(state.abc_guide_strength),
            "abc_limit": int(state.abc_limit),
            "random_reassignment_rate": float(state.random_reassignment_rate),
        }
    )


def ejecutar_epoch(
    inst: MCDP_Instance,
    params: CooperativeMCDPParams,
    epoch_idx: int = 0,
    verbose: bool = True,
    sol_inyectada: Sequence[int] | None = None,
) -> CooperativeMCDPEpochResult:
    """Ejecuta un epoch con una población compartida WOA--ABC."""

    _check_capacity(inst)
    if params.seed is not None:
        random.seed(params.seed + epoch_idx)
        np.random.seed(params.seed + epoch_idx)

    latent, assignments, costs, trials = _initialise_population(
        inst,
        params.pop_size,
        params.random_reassignment_rate,
    )
    if sol_inyectada is not None:
        _inject_solution(latent, assignments, costs, trials, sol_inyectada, inst)

    best_latent, best_assignment, best_cost, _ = _population_best(
        latent, assignments, costs
    )
    best_evaluation = inst.evaluate(best_assignment)
    best_pos_prev = best_latent.copy()

    base_limit = params.limit
    if base_limit is None:
        base_limit = max(params.min_abc_limit, params.pop_size * inst.num_machines // 2)
    base_limit = max(params.min_abc_limit, int(base_limit))
    state = _AdaptiveState(
        woa_exploration=params.woa_exploration,
        abc_phi_scale=params.abc_phi_scale,
        abc_guide_strength=params.abc_guide_strength,
        abc_limit=base_limit,
        random_reassignment_rate=params.random_reassignment_rate,
    )

    monitor = StagnationMonitor(params.stag_cfg) if params.use_dtw else None
    last_adaptation = -10**9
    stagnation_fires = 0

    historial: list[float] = []
    historial_inst: list[float] = []
    historial_woa: list[float] = []
    historial_abc: list[float] = []
    dtw_deltas: list[float] = []
    dtw_info_hist: list[dict[str, Any]] = []
    cooperation_events: list[dict[str, Any]] = []
    adaptation_events: list[dict[str, Any]] = []
    parameters_history: list[dict[str, float | int]] = []

    def update_global() -> None:
        nonlocal best_latent, best_assignment, best_cost, best_evaluation
        candidate_latent, candidate_assignment, candidate_cost, _ = _population_best(
            latent, assignments, costs
        )
        if candidate_cost < best_cost:
            best_latent = candidate_latent
            best_assignment = candidate_assignment
            best_cost = candidate_cost
            best_evaluation = inst.evaluate(best_assignment)

    for iteration in range(params.iterations):
        # Momentum entre los mejores globales consecutivos, como en MSE-ABC.
        momentum = best_latent - best_pos_prev
        best_pos_prev = best_latent.copy()

        status: dict[str, Any] = {"ready": False, "fire": False, "n": 0}
        adaptation_event: dict[str, Any] | None = None
        rescue_count = 0
        if monitor is not None:
            status = monitor.update(-best_cost)
            dtw_deltas.append(float(status.get("delta", 0.0)))
            if (
                status.get("fire", False)
                and iteration - last_adaptation >= params.adaptation_cooldown
            ):
                adaptation_event = _adapt_after_stagnation(
                    state, params, iteration, status
                )
                last_adaptation = iteration
                stagnation_fires += 1
                rescue_count = _rescue_worst(
                    latent,
                    assignments,
                    costs,
                    trials,
                    inst,
                    params.rescue_fraction,
                    state.random_reassignment_rate,
                )
                adaptation_event["rescate"] = rescue_count
                adaptation_events.append(adaptation_event)
                monitor.trigger_streak = 0
        else:
            dtw_deltas.append(0.0)

        step_size = params.step_initial * (
            params.step_final / params.step_initial
        ) ** ((iteration + 1) / params.iterations)
        momentum_term = params.momentum_factor * momentum
        _record_parameters(
            parameters_history,
            state,
            iteration,
            step_size,
            float(np.linalg.norm(momentum)),
        )

        if adaptation_event is not None and verbose:
            print(
                f"  [Coop MCDP] DTW fire en iteración {iteration + 1}: "
                f"WOA×{state.woa_exploration:.2f}, "
                f"ABC phi×{state.abc_phi_scale:.2f}, "
                f"rescates={rescue_count}",
                flush=True,
            )

        # FASE WOA: exploración sobre la misma población compartida.
        woa_cost_before = float(np.min(costs))
        for i in range(params.pop_size):
            r1, r2 = random.random(), random.random()
            a = (2.0 - 2.0 * (iteration / max(1, params.iterations - 1)))
            a *= state.woa_exploration
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            p = random.random()
            l = random.uniform(-1.0, 1.0)
            if p < params.exploration_probability:
                if abs(A) < 1.0:
                    distance = np.abs(C * best_latent - latent[i])
                    new_latent = best_latent - A * distance
                else:
                    random_index = random.randrange(params.pop_size)
                    distance = np.abs(C * latent[random_index] - latent[i])
                    new_latent = latent[random_index] - A * distance
            else:
                distance = np.abs(best_latent - latent[i])
                new_latent = (
                    distance
                    * math.exp(params.b_spiral * l)
                    * math.cos(2.0 * math.pi * l)
                    + best_latent
                )
            new_latent = new_latent + step_size * momentum_term
            new_latent = np.clip(new_latent, 0.0, float(inst.max_cells - 1))
            candidate = _repair_assignment(
                new_latent,
                inst,
                random_rate=state.random_reassignment_rate,
            )
            candidate_cost = float(inst.objective(candidate))
            if candidate_cost <= costs[i]:
                latent[i] = _latent_from_assignment(candidate, inst)
                assignments[i] = candidate
                costs[i] = candidate_cost
                trials[i] = 0
            else:
                trials[i] += 1

        update_global()
        woa_best = float(np.min(costs))
        historial_woa.append(woa_best)

        # FASE ABC EMPLEADA: vecinos sobre la población que WOA acaba de dejar.
        def abc_attempt(index: int) -> None:
            neighbour_candidates = [j for j in range(params.pop_size) if j != index]
            neighbour = random.choice(neighbour_candidates)
            coordinate = random.randrange(inst.num_machines)
            phi = random.uniform(-1.0, 1.0) * state.abc_phi_scale
            candidate_latent = latent[index].copy()
            candidate_latent[coordinate] += phi * (
                latent[index, coordinate] - latent[neighbour, coordinate]
            )
            candidate_latent[coordinate] += step_size * momentum_term[coordinate]

            # El mejor global guía la explotación, pero la población sigue siendo única.
            candidate_latent += (
                state.abc_guide_strength
                * random.random()
                * (best_latent - latent[index])
            )
            candidate_latent = np.clip(
                candidate_latent,
                0.0,
                float(inst.max_cells - 1),
            )
            candidate = _repair_assignment(
                candidate_latent,
                inst,
                random_rate=state.random_reassignment_rate,
            )
            candidate_cost = float(inst.objective(candidate))
            if candidate_cost <= costs[index]:
                latent[index] = _latent_from_assignment(candidate, inst)
                assignments[index] = candidate
                costs[index] = candidate_cost
                trials[index] = 0
            else:
                trials[index] += 1

        for i in range(params.pop_size):
            abc_attempt(i)
        update_global()

        # FASE ABC OBSERVADORA: roulette sobre las mismas fuentes de alimento.
        quality = 1.0 / (1.0 + costs - np.min(costs))
        probabilities = quality / np.sum(quality)
        selected = np.random.choice(
            params.pop_size,
            size=params.pop_size,
            p=probabilities,
        )
        for index in selected:
            abc_attempt(int(index))
        update_global()

        # FASE SCOUT: se reinicia la fuente más estancada, como en MSE-ABC.
        scout_index = int(np.argmax(trials))
        if trials[scout_index] >= state.abc_limit:
            latent[scout_index], assignment, cost = _random_individual(
                inst,
                state.random_reassignment_rate,
            )
            assignments[scout_index] = assignment
            costs[scout_index] = cost
            trials[scout_index] = 0
        update_global()

        abc_best = float(np.min(costs))
        historial_abc.append(abc_best)
        cooperation_events.append(
            {
                "iteracion": iteration,
                "tipo": "handoff_woa_abc",
                "woa_best": woa_best,
                "abc_best": abc_best,
                "lider_compartido": best_cost,
                "woa_mejora": woa_best < woa_cost_before,
            }
        )

        current_cost = float(np.min(costs))
        historial.append(best_cost)
        historial_inst.append(current_cost)

        info = status.copy()
        info.update(
            {
                "iteracion": iteration,
                "adaptado": adaptation_event is not None,
                "rescate": rescue_count,
                "fase": "poblacion_compartida",
                "woa_best": woa_best,
                "abc_best": abc_best,
            }
        )
        dtw_info_hist.append(info)

        if verbose:
            print(
                f"  [Coop MCDP] Iter {iteration + 1:3d}/{params.iterations} | "
                f"WOA={woa_best:6.1f} | ABC={abc_best:6.1f} | "
                f"Global={best_cost:6.1f}",
                flush=True,
            )

        _relax_state(state, params, base_limit)

    return CooperativeMCDPEpochResult(
        epoch_idx=epoch_idx,
        mejor_costo=best_cost,
        iteraciones=len(historial),
        stagnation_fires=stagnation_fires,
        mejor_solucion=best_assignment,
        mejor_evaluacion=best_evaluation,
        historial=historial,
        historial_inst=historial_inst,
        historial_woa=historial_woa,
        historial_abc=historial_abc,
        dtw_deltas=dtw_deltas,
        dtw_info_hist=dtw_info_hist,
        eventos_cooperacion=cooperation_events,
        eventos_adaptacion=adaptation_events,
        parametros_historial=parameters_history,
    )


def ejecutar_mcdp_cooperativo(
    inst: MCDP_Instance,
    params: CooperativeMCDPParams,
    verbose: bool = True,
) -> CooperativeMCDPResult:
    """Ejecuta todos los epochs y devuelve la mejor evaluación MCDP."""

    _check_capacity(inst)
    epoch_results: list[CooperativeMCDPEpochResult] = []
    best_cost = float("inf")
    best_assignment: list[int] = []
    best_evaluation: MCDPEvaluation | None = None

    for epoch in range(params.epochs):
        result = ejecutar_epoch(inst, params, epoch_idx=epoch, verbose=verbose)
        epoch_results.append(result)
        if result.mejor_costo < best_cost:
            best_cost = result.mejor_costo
            best_assignment = result.mejor_solucion.copy()
            best_evaluation = result.mejor_evaluacion

    if best_evaluation is None:
        best_evaluation = inst.evaluate(best_assignment)
    return CooperativeMCDPResult(
        epochs=epoch_results,
        mejor_costo_global=best_cost,
        mejor_sol_global=best_assignment,
        evaluacion_global=best_evaluation,
    )


cooperative_mcdp_epoch = ejecutar_epoch


if __name__ == "__main__":
    from mcdp_core.data import load_mcdp_instances

    _repo_root = Path(__file__).resolve().parents[2]
    instance = load_mcdp_instances(
        str(_repo_root / "mcdp_core" / "instances" / "instancias.txt"),
        max_cells=3,
        max_machines_per_cell=6,
    )[0]
    result = ejecutar_mcdp_cooperativo(
        instance,
        CooperativeMCDPParams(pop_size=20, iterations=40, seed=42),
        verbose=True,
    )
    print(f"Resultado MCDP: costo={result.mejor_costo_global:.1f}")
