"""Shared-population binary WOA followed by a full ABC cycle for the MKP.

A complete cycle is WOA, employed bees, onlookers, then eligible scouts.
Initialization happens once, using the WOA sign-bit initializer. After each
complete cycle the existing stagnation monitor may restart only the ``a`` and
``rho`` clock. Incomplete cycles are recorded and do not update the monitor.
``a2`` stays on the ordinary cycle index.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from dtw_stagnation import StagnationConfig, StagnationMonitor
from lb2.binarization import _calcular_L1, _calcular_L2
from mkp_core.problem import MKPInstance
from mkp_core.repair import reparar_solucion


PHASE_INITIALIZATION = "initialization"
PHASE_WOA = "woa"
PHASE_EMPLOYED = "employed"
PHASE_ONLOOKERS = "onlookers"
PHASE_SCOUTS = "scouts"
PHASES = (
    PHASE_INITIALIZATION,
    PHASE_WOA,
    PHASE_EMPLOYED,
    PHASE_ONLOOKERS,
    PHASE_SCOUTS,
)

TERMINATION_ITERATIONS = "iterations"
TERMINATION_EVALUATION_BUDGET = "evaluation_budget"

VARIANTS = ("v0", "v1", "v2", "v3", "v4")
VARIANT_ERROR = "variant must be one of v0, v1, v2, v3, v4"

# One initialization, plus a cycle that proposes three LB2 pairs and one scout
# for every source. This is an upper bound, not a hidden extra search.
_EVALUATIONS_PER_SOURCE_PER_CYCLE = 7


@dataclass
class CycleRecord:
    """One ordinary cycle, including a cycle stopped at a phase boundary.

    ``evaluations`` is the cumulative objective count when the record is
    closed. ``schedule_index`` is the clock used for ``a`` and ``rho``; it
    matches ``cycle`` until a future recovery resets it. ``a2`` is not stored
    because it follows ``cycle`` and is not part of the controlled pair.
    """

    cycle: int
    schedule_index: int
    a: float
    rho: float
    best: float
    evaluations: int
    completeness: bool


@dataclass
class HybridConfig:
    """Fixed baseline controls. ``None`` budgets are resolved by ``solve``."""

    population_size: int = 30
    iterations: int = 300
    max_evaluations: int | None = None
    v_max: float = 6.0
    b: float = 1.0
    G1: float = 0.5
    G2: float = 0.5
    G3: float = 0.5
    scout_limit: int | None = None
    use_stagnation: bool = True
    stag_cfg: StagnationConfig | None = None
    variant: str = "v0"

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass
class HybridResult:
    """JSON-friendly search report. Live positions and bit rows are omitted."""

    best_solution: list[int]
    best_value: float
    gap: float | None
    evaluations: int
    evaluations_by_phase: dict[str, int]
    cycles_completed: int
    termination_reason: str
    history: list[CycleRecord]
    phases_executed: list[str]
    source_trials: list[int]
    archive_size: int
    scout_limit: int
    diagnostics: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class _RunState:
    """Mutable shared population. T2's hook observes this object in place."""

    inst: MKPInstance
    config: HybridConfig
    rng: np.random.Generator
    positions: np.ndarray
    bits: list[list[int]]
    fitness: list[float]
    trials: np.ndarray
    max_evaluations: int
    scout_limit: int
    schedule_index: int = 0
    recovery_active: bool = False
    recovery_span: int | None = None
    evaluations: int = 0
    best_value: float = float("-inf")
    best_solution: list[int] = field(default_factory=list)
    phases_executed: list[str] = field(default_factory=list)
    diagnostics: list = field(default_factory=list)
    by_phase: dict[str, int] = field(default_factory=dict)
    monitor: StagnationMonitor | None = None

    def __post_init__(self) -> None:
        if not self.by_phase:
            self.by_phase = {phase: 0 for phase in PHASES}


def _require_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validate_config(config: HybridConfig) -> None:
    if not isinstance(config, HybridConfig):
        raise ValueError("config must be a HybridConfig")
    _require_int("population_size", config.population_size, 2)
    _require_int("iterations", config.iterations, 2)
    if config.max_evaluations is not None:
        _require_int("max_evaluations", config.max_evaluations, config.population_size)
    v_max = _require_finite("v_max", config.v_max)
    if v_max <= 0.0:
        raise ValueError("v_max must be a finite number > 0")
    _require_finite("b", config.b)
    _require_finite("G1", config.G1)
    _require_finite("G2", config.G2)
    _require_finite("G3", config.G3)
    if config.scout_limit is not None:
        _require_int("scout_limit", config.scout_limit, 1)
    if not isinstance(config.use_stagnation, bool):
        raise ValueError("use_stagnation must be a bool")
    _validate_variant(config.variant)
    if config.stag_cfg is not None:
        _validate_stagnation(config.stag_cfg)


def _validate_stagnation(cfg: StagnationConfig) -> None:
    """Reject invalid monitor options without mutating the shared detector."""

    if not isinstance(cfg, StagnationConfig):
        raise ValueError("stag_cfg must be a StagnationConfig or None")
    _require_int("window", cfg.window, 2)
    if isinstance(cfg.band, bool) or not isinstance(cfg.band, int) or cfg.band < 0:
        raise ValueError("band must be an integer >= 0")
    slope = _require_finite("min_slope", cfg.min_slope)
    if slope < 0.0:
        raise ValueError("min_slope must be a finite number >= 0")
    _require_int("plateau_max", cfg.plateau_max, 0)
    _require_int("patience", cfg.patience, 1)
    if not isinstance(cfg.use_ddtw, bool):
        raise ValueError("use_ddtw must be a bool")
    if not isinstance(cfg.adapt_thresholds, bool):
        raise ValueError("adapt_thresholds must be a bool")
    p_low = _require_finite("p_low", cfg.p_low)
    p_high = _require_finite("p_high", cfg.p_high)
    if not 0.0 <= p_low <= p_high <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= p_low <= p_high <= 100")
    tolerance = _require_finite("improvement_tol", cfg.improvement_tol)
    if tolerance < 0.0:
        raise ValueError("improvement_tol must be a finite number >= 0")


def _fresh_monitor(config: HybridConfig) -> StagnationMonitor | None:
    """Copy the caller's detector settings into a run-local monitor.

    ``stag_cfg is None`` selects the existing defaults. It does not disable
    detection; ``use_stagnation=False`` does that. The copy absorbs the
    monitor's own band normalization so the user's object stays unchanged.
    """

    if not config.use_stagnation:
        return None
    copied = StagnationConfig() if config.stag_cfg is None else replace(config.stag_cfg)
    return StagnationMonitor(cfg=copied)


def _validate_instance(inst: MKPInstance) -> None:
    """Reject inputs that would crash or escape the supported MKP domain.

    The supported domain is conventional: nonnegative finite profits, weights
    and capacities, with each item carrying positive total weight. A zero
    reference is an unknown optimum, not a malformed instance.
    """

    if not isinstance(inst, MKPInstance):
        raise ValueError("inst must be an MKPInstance")
    n = inst.n
    m = inst.m
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("instance n must be a positive integer")
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError("instance m must be a positive integer")
    try:
        profits = np.asarray(inst.p, dtype=float)
        weights = np.asarray(inst.r, dtype=float)
        capacities = np.asarray(inst.b, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("instance arrays must be numeric") from exc
    if profits.shape != (n,):
        raise ValueError("profits must have shape (n,)")
    if weights.shape != (m, n):
        raise ValueError("weights must have shape (m, n)")
    if capacities.shape != (m,):
        raise ValueError("capacities must have shape (m,)")
    if not np.isfinite(profits).all() or np.any(profits < 0):
        raise ValueError("profits must be finite and nonnegative")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    if np.any(np.sum(weights, axis=0) <= 0.0):
        raise ValueError("each item must have positive weight")
    if not np.isfinite(capacities).all() or np.any(capacities < 0):
        raise ValueError("capacities must be finite and nonnegative")
    if isinstance(inst.valor_optimo, bool) or not isinstance(inst.valor_optimo, (int, float)):
        raise ValueError("reference value must be a finite nonnegative number")
    reference = float(inst.valor_optimo)
    if not math.isfinite(reference) or reference < 0.0:
        raise ValueError("reference value must be a finite nonnegative number")


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be a nonnegative integer")
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _resolved_max_evaluations(config: HybridConfig) -> int:
    if config.max_evaluations is None:
        return config.population_size * (
            1 + _EVALUATIONS_PER_SOURCE_PER_CYCLE * config.iterations
        )
    return config.max_evaluations


def _validate_variant(value: object) -> str:
    if not isinstance(value, str) or value not in VARIANTS:
        raise ValueError(VARIANT_ERROR)
    return value


def _resolved_scout_limit(config: HybridConfig, n_items: int) -> int:
    if config.scout_limit is None:
        if config.variant == "v1":
            return 100
        return config.population_size * n_items
    return config.scout_limit


def _control_parameters(
    schedule_index: int,
    iterations: int,
    recovery_active: bool,
    variant: str = "v0",
    recovery_span: int | None = None,
) -> tuple[float, float]:
    """Return ``(a, rho)`` for the current clock.

    Before any recovery, rho stays 1. After a hook reset, the next cycle is
    called with schedule index 0 and ``recovery_active`` so rho restarts at 2
    on v0. v2 changes only the post-fire amplitudes. v3 changes only the
    post-fire horizon. v1 and v4 keep this baseline clock.
    """

    if recovery_active and variant == "v2":
        span = iterations - 1
        return (
            max(0.0, 1.0 - 2.0 * schedule_index / span),
            max(1.0, 1.5 - schedule_index / span),
        )
    if recovery_active and variant == "v3":
        if recovery_span is None:
            raise RuntimeError("v3 recovery is missing its remaining-cycle span")
        if recovery_span == 0:
            return 0.0, 1.0
        return (
            max(0.0, 2.0 - 2.0 * schedule_index / recovery_span),
            max(1.0, 2.0 - schedule_index / recovery_span),
        )
    span = iterations - 1
    a = max(0.0, 2.0 - 2.0 * schedule_index / span)
    if recovery_active:
        rho = max(1.0, 2.0 - schedule_index / span)
    else:
        rho = 1.0
    return a, rho


def _a2(cycle_index: int, iterations: int) -> float:
    """WOA spiral schedule. This follows the ordinary cycle, not the a/rho clock."""

    return -1.0 + cycle_index * (-1.0 / (iterations - 1))


def _remaining(state: _RunState) -> int:
    return state.max_evaluations - state.evaluations


def _freeze_leader(positions: np.ndarray, index: int) -> np.ndarray:
    """Copy the leader so later writes to its row cannot move the target."""

    return positions[index].copy()


def _woa_destination(
    position: np.ndarray,
    leader: np.ndarray,
    peer: np.ndarray,
    A: float,
    C: float,
    p: float,
    l: float,
    b: float,
) -> np.ndarray:
    """WOA move from ``mh/woa.py``, including the peer branch and the spiral."""

    if p < 0.5:
        if abs(A) < 1.0:
            distance = np.abs(C * leader - position)
            return leader - A * distance
        distance = np.abs(C * peer - position)
        return peer - A * distance
    distance = np.abs(leader - position)
    return distance * np.exp(b * l) * np.cos(2.0 * np.pi * l) + leader


def _clip_coordinates(position: np.ndarray, v_max: float) -> np.ndarray:
    return np.clip(np.asarray(position, dtype=float), -v_max, v_max)


def _distinct_index(rng: np.random.Generator, index: int, population_size: int) -> int:
    peer = int(rng.integers(0, population_size - 1))
    if peer >= index:
        peer += 1
    return peer


def _roulette_probabilities(fitness: list[float]) -> np.ndarray:
    weights = np.asarray(fitness, dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(weights.size, 1.0 / weights.size)
    probabilities = weights / total
    return probabilities / float(probabilities.sum())


def _prefer_l1(value_l1: float, value_l2: float) -> bool:
    """LB2 keeps L1 only on a strict win. Ties stay with L2."""

    return value_l1 > value_l2


def _flip_candidate(
    solution: list[int],
    probabilities: np.ndarray,
    draws: np.ndarray,
) -> list[int]:
    candidate = list(solution)
    for index, probability in enumerate(probabilities):
        if draws[index] < probability:
            candidate[index] = 1 - candidate[index]
    return candidate


def _evaluate(state: _RunState, raw_bits: list[int], phase: str) -> tuple[list[int], float]:
    """Count one repaired candidate. Do not call the instance objective again."""

    if _remaining(state) < 1:
        raise RuntimeError(f"{phase} evaluation exceeded the budget")
    repaired, value = reparar_solucion(raw_bits, state.inst)
    repaired = [int(bit) for bit in repaired]
    value = float(value)
    state.evaluations += 1
    state.by_phase[phase] += 1
    if value > state.best_value:
        state.best_value = value
        state.best_solution = list(repaired)
    return repaired, value


def _lb2_pair(
    state: _RunState,
    position: np.ndarray,
    current_bits: list[int],
    phase: str,
) -> tuple[list[int], float]:
    """Evaluate both transfer candidates through the same counted wrapper."""

    if _remaining(state) < 2:
        raise RuntimeError("an LB2 proposal needs two remaining evaluations")
    config = state.config
    probabilities_l1 = _calcular_L1(position, config.v_max, config.G1, config.G2, config.G3)
    probabilities_l2 = _calcular_L2(position, config.v_max, config.G1, config.G2, config.G3)
    draws_l1 = state.rng.random(state.inst.n)
    draws_l2 = state.rng.random(state.inst.n)
    candidate_l1 = _flip_candidate(current_bits, probabilities_l1, draws_l1)
    candidate_l2 = _flip_candidate(current_bits, probabilities_l2, draws_l2)
    solution_l1, value_l1 = _evaluate(state, candidate_l1, phase)
    solution_l2, value_l2 = _evaluate(state, candidate_l2, phase)
    if _prefer_l1(value_l1, value_l2):
        return solution_l1, value_l1
    return solution_l2, value_l2


def _commit_woa(
    state: _RunState,
    index: int,
    updated_position: np.ndarray,
    chosen_solution: list[int],
    chosen_value: float,
) -> None:
    """Always store the latent move. Accept bits on a tie; reset trials only on a gain."""

    state.positions[index] = _clip_coordinates(updated_position, state.config.v_max)
    if chosen_value < state.fitness[index]:
        return
    improved = chosen_value > state.fitness[index]
    state.bits[index] = list(chosen_solution)
    state.fitness[index] = chosen_value
    if improved:
        state.trials[index] = 0


def _commit_abc(
    state: _RunState,
    index: int,
    updated_position: np.ndarray,
    chosen_solution: list[int],
    chosen_value: float,
) -> None:
    """Strict ABC acceptance updates position, bits and fitness together."""

    if chosen_value > state.fitness[index]:
        state.positions[index] = _clip_coordinates(updated_position, state.config.v_max)
        state.bits[index] = list(chosen_solution)
        state.fitness[index] = chosen_value
        state.trials[index] = 0
        return
    state.trials[index] += 1


def _woa_step(state: _RunState, index: int, a: float, a2: float, leader: np.ndarray) -> None:
    rng = state.rng
    r1 = float(rng.random())
    r2 = float(rng.random())
    A = 2.0 * a * r1 - a
    C = 2.0 * r2
    p = float(rng.random())
    l = (a2 - 1.0) * float(rng.random()) + 1.0
    current = state.positions[index].copy()
    if p < 0.5 and abs(A) >= 1.0:
        peer_index = int(rng.integers(0, state.config.population_size))
        peer = state.positions[peer_index].copy()
    else:
        peer = current
    destination = _woa_destination(current, leader, peer, A, C, p, l, state.config.b)
    clipped = _clip_coordinates(destination, state.config.v_max)
    chosen_solution, chosen_value = _lb2_pair(state, clipped, state.bits[index], PHASE_WOA)
    _commit_woa(state, index, clipped, chosen_solution, chosen_value)


def _abc_step(state: _RunState, index: int, rho: float, phase: str) -> None:
    peer_index = _distinct_index(state.rng, index, state.config.population_size)
    phi = float(state.rng.uniform(-rho, rho))
    current = state.positions[index]
    peer = state.positions[peer_index]
    if state.config.variant == "v4":
        coordinate = int(state.rng.integers(0, state.inst.n))
        neighbor = np.array(current, dtype=float, copy=True)
        neighbor[coordinate] = current[coordinate] + phi * (
            current[coordinate] - peer[coordinate]
        )
    else:
        neighbor = current + phi * (current - peer)
    clipped = _clip_coordinates(neighbor, state.config.v_max)
    chosen_solution, chosen_value = _lb2_pair(state, clipped, state.bits[index], phase)
    _commit_abc(state, index, clipped, chosen_solution, chosen_value)


def _replace_scout(state: _RunState, index: int) -> None:
    """Replace one exhausted source. The archived best is left untouched."""

    n_items = state.inst.n
    v_max = state.config.v_max
    state.positions[index] = state.rng.uniform(-v_max, v_max, size=n_items)
    raw_bits = [int(bit) for bit in state.rng.integers(0, 2, size=n_items)]
    repaired, value = _evaluate(state, raw_bits, PHASE_SCOUTS)
    state.bits[index] = repaired
    state.fitness[index] = value
    state.trials[index] = 0


def _initialize(state: _RunState) -> None:
    """WOA initializer: one uniform draw in [-1, 1], sign bits, one repair each."""

    population_size = state.config.population_size
    n_items = state.inst.n
    state.phases_executed.append(PHASE_INITIALIZATION)
    state.positions = state.rng.uniform(-1.0, 1.0, size=(population_size, n_items))
    state.bits = []
    state.fitness = []
    for index in range(population_size):
        raw_bits = [1 if coordinate > 0.0 else 0 for coordinate in state.positions[index]]
        repaired, value = _evaluate(state, raw_bits, PHASE_INITIALIZATION)
        state.bits.append(repaired)
        state.fitness.append(value)
    state.trials = np.zeros(population_size, dtype=int)


def _run_woa(state: _RunState, a: float, a2: float) -> bool:
    if _remaining(state) < 2:
        return False
    state.phases_executed.append(PHASE_WOA)
    leader_index = max(range(state.config.population_size), key=lambda index: state.fitness[index])
    leader = _freeze_leader(state.positions, leader_index)
    for index in range(state.config.population_size):
        if _remaining(state) < 2:
            return False
        _woa_step(state, index, a, a2, leader)
    return True


def _run_employed(state: _RunState, rho: float) -> bool:
    if _remaining(state) < 2:
        return False
    state.phases_executed.append(PHASE_EMPLOYED)
    for index in range(state.config.population_size):
        if _remaining(state) < 2:
            return False
        _abc_step(state, index, rho, PHASE_EMPLOYED)
    return True


def _run_onlookers(state: _RunState, rho: float) -> bool:
    if _remaining(state) < 2:
        return False
    state.phases_executed.append(PHASE_ONLOOKERS)
    probabilities = _roulette_probabilities(state.fitness)
    population_size = state.config.population_size
    for _ in range(population_size):
        if _remaining(state) < 2:
            return False
        index = int(state.rng.choice(population_size, p=probabilities))
        _abc_step(state, index, rho, PHASE_ONLOOKERS)
    return True


def _run_scouts(state: _RunState) -> bool:
    needs_scout = any(
        int(state.trials[index]) >= state.scout_limit
        for index in range(state.config.population_size)
    )
    if needs_scout and _remaining(state) < 1:
        return False
    state.phases_executed.append(PHASE_SCOUTS)
    for index in range(state.config.population_size):
        if int(state.trials[index]) < state.scout_limit:
            continue
        if _remaining(state) < 1:
            return False
        _replace_scout(state, index)
    return True


def _run_cycle(state: _RunState, cycle_index: int) -> CycleRecord:
    a, rho = _control_parameters(
        state.schedule_index,
        state.config.iterations,
        state.recovery_active,
        state.config.variant,
        state.recovery_span,
    )
    # a2 intentionally uses cycle_index. Recovery must not restart this draw.
    a2 = _a2(cycle_index, state.config.iterations)
    complete = True
    if not _run_woa(state, a, a2):
        complete = False
    elif not _run_employed(state, rho):
        complete = False
    elif not _run_onlookers(state, rho):
        complete = False
    elif not _run_scouts(state):
        complete = False
    return CycleRecord(
        cycle=cycle_index,
        schedule_index=state.schedule_index,
        a=float(a),
        rho=float(rho),
        best=float(state.best_value),
        evaluations=state.evaluations,
        completeness=complete,
    )


def _status_snapshot(status: dict) -> dict:
    """Copy detector fields before ``reset`` clears the monitor."""

    snapshot: dict[str, object] = {
        "ready": bool(status.get("ready", False)),
        "fire": bool(status.get("fire", False)),
    }
    for key in ("no_improve_len", "trigger_streak", "n"):
        value = status.get(key)
        snapshot[key] = None if value is None else int(value)
    for key in (
        "D1_vs_ramp",
        "D2_vs_const",
        "delta",
        "theta_c",
        "theta_r",
        "theta_delta",
    ):
        value = status.get(key)
        snapshot[key] = None if value is None else float(value)
    return snapshot


def _after_complete_cycle(state: _RunState, record: CycleRecord) -> bool:
    """Observe one complete cycle. Return True when the next cycle should restart.

    The archived best is passed through once. A fire resets only the monitor.
    Population, source trials, and the archive are left in place. ``solve``
    marks ``recovery_applied`` only when the next indexed cycle can run a WOA pair.
    """

    monitor = state.monitor
    if monitor is None:
        return False
    status = monitor.update(float(record.best))
    diagnostic = {
        "cycle": record.cycle,
        "schedule_index": record.schedule_index,
        "evaluations": record.evaluations,
        "a": float(record.a),
        "rho": float(record.rho),
        "best": float(record.best),
        "recovery_applied": False,
    }
    diagnostic.update(_status_snapshot(status))
    state.diagnostics.append(diagnostic)
    if diagnostic["fire"]:
        monitor.reset()
        return True
    return False


def _stagnation_metadata(monitor: StagnationMonitor | None) -> dict | None:
    if monitor is None:
        return None
    cfg = monitor.cfg
    return {
        "window": int(cfg.window),
        "band": int(cfg.band),
        "min_slope": float(cfg.min_slope),
        "plateau_max": int(cfg.plateau_max),
        "patience": int(cfg.patience),
        "use_ddtw": bool(cfg.use_ddtw),
        "adapt_thresholds": bool(cfg.adapt_thresholds),
        "p_low": float(cfg.p_low),
        "p_high": float(cfg.p_high),
        "improvement_tol": float(cfg.improvement_tol),
    }


def _run_metadata(
    config: HybridConfig,
    seed: int,
    max_evaluations: int,
    scout_limit: int,
    monitor: StagnationMonitor | None,
    caller_metadata: dict | None,
) -> dict:
    metadata = dict(caller_metadata or {})
    metadata["seed"] = seed
    metadata["config"] = {
        "population_size": config.population_size,
        "iterations": config.iterations,
        "max_evaluations": max_evaluations,
        "scout_limit": scout_limit,
        "v_max": float(config.v_max),
        "b": float(config.b),
        "G1": float(config.G1),
        "G2": float(config.G2),
        "G3": float(config.G3),
        "use_stagnation": config.use_stagnation,
        "stagnation": _stagnation_metadata(monitor),
    }
    if config.variant != "v0":
        metadata["config"]["variant"] = config.variant
    return metadata


def _gap(reference: float, best_value: float) -> float | None:
    if reference <= 0.0:
        return None
    return 100.0 * (reference - best_value) / reference


def solve(
    inst: MKPInstance,
    config: HybridConfig | None = None,
    *,
    seed: int,
    metadata: dict | None = None,
) -> HybridResult:
    """Run the hybrid with a local generator seeded by ``seed``.

    ``metadata`` is copied into the result and is not mutated. ``solve`` owns
    the ``seed`` and ``config`` keys in that copy.
    """

    if config is None:
        config = HybridConfig()
    _validate_config(config)
    _validate_instance(inst)
    resolved_seed = _validate_seed(seed)
    max_evaluations = _resolved_max_evaluations(config)
    scout_limit = _resolved_scout_limit(config, inst.n)
    monitor = _fresh_monitor(config)
    state = _RunState(
        inst=inst,
        config=config,
        rng=np.random.default_rng(resolved_seed),
        positions=np.empty((config.population_size, inst.n), dtype=float),
        bits=[],
        fitness=[],
        trials=np.zeros(config.population_size, dtype=int),
        max_evaluations=max_evaluations,
        scout_limit=scout_limit,
        monitor=monitor,
    )
    _initialize(state)

    history: list[CycleRecord] = []
    termination_reason = TERMINATION_ITERATIONS
    for cycle_index in range(config.iterations):
        record = _run_cycle(state, cycle_index)
        history.append(record)
        if not record.completeness:
            termination_reason = TERMINATION_EVALUATION_BUDGET
            break
        fired = _after_complete_cycle(state, record)
        # A claimed restart requires a later cycle that can execute one LB2 pair.
        recovery_applied = bool(
            fired
            and cycle_index + 1 < config.iterations
            and _remaining(state) >= 2
        )
        if state.diagnostics and "recovery_applied" in state.diagnostics[-1]:
            state.diagnostics[-1]["recovery_applied"] = recovery_applied
        if recovery_applied:
            state.schedule_index = 0
            state.recovery_active = True
            if config.variant == "v3":
                remaining_cycles = config.iterations - (cycle_index + 1)
                state.recovery_span = remaining_cycles - 1
        else:
            state.schedule_index += 1

    if sum(state.by_phase.values()) != state.evaluations:
        raise RuntimeError("evaluation accounting diverged from the phase counts")
    if not state.best_solution:
        raise RuntimeError("search archived no feasible candidate")

    return HybridResult(
        best_solution=list(state.best_solution),
        best_value=float(state.best_value),
        gap=_gap(float(inst.valor_optimo), state.best_value),
        evaluations=state.evaluations,
        evaluations_by_phase=dict(state.by_phase),
        cycles_completed=sum(1 for record in history if record.completeness),
        termination_reason=termination_reason,
        history=history,
        phases_executed=list(state.phases_executed),
        source_trials=[int(trial) for trial in state.trials],
        archive_size=state.evaluations,
        scout_limit=scout_limit,
        diagnostics=list(state.diagnostics),
        metadata=_run_metadata(
            config,
            resolved_seed,
            max_evaluations,
            scout_limit,
            monitor,
            metadata,
        ),
    )


def _local_instance_file(path: str) -> Path:
    """Accept only an existing local file. Reject URLs before any loader import."""

    text = path.strip()
    if "://" in text:
        raise ValueError("instance path must be an existing local file, not a URL")
    candidate = Path(text)
    if not candidate.is_file():
        raise ValueError("instance path must be an existing local file, not a URL")
    return candidate


def _read_orlibrary(path: Path) -> list[dict]:
    """Parse a local OR-Library file. The loader is imported only after path checks."""

    from mkp_core.data_loader import parsear_instancias

    return parsear_instancias(path.read_text(encoding="utf-8"))


def _instance_from_row(row: dict) -> MKPInstance:
    return MKPInstance(
        n=int(row["n"]),
        m=int(row["m"]),
        valor_optimo=float(row["valor_optimo"]),
        p=np.asarray(row["p"], dtype=float),
        r=np.asarray(row["r"], dtype=float),
        b=np.asarray(row["b"], dtype=float),
    )


def _load_local_instance(path: Path, index: int) -> tuple[MKPInstance, dict]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("instance index must be a nonnegative integer")
    try:
        rows = _read_orlibrary(path)
    except (OSError, ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"could not parse local OR-Library file: {exc}") from exc
    if index >= len(rows):
        raise ValueError(f"instance index must be in 0..{len(rows) - 1}")
    instance = _instance_from_row(rows[index])
    info = {
        "path": str(path.resolve()),
        "index": index,
        "n": instance.n,
        "m": instance.m,
        "reference": float(instance.valor_optimo),
    }
    return instance, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="woa_abc",
        description=(
            "Run the shared-population binary WOA-ABC hybrid on one local "
            "OR-Library MKP instance. This command does not download data."
        ),
    )
    parser.add_argument("--instance", required=True, help="Path to a local OR-Library file. URLs are rejected.")
    parser.add_argument("--instance-index", type=int, default=0, help="Zero-based instance index.")
    parser.add_argument("--seed", type=int, required=True, help="Nonnegative local RNG seed.")
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300, help="Fixed cycle horizon T.")
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--scout-limit", type=int, default=None)
    parser.add_argument("--no-stagnation", action="store_true", help="Disable detector recovery and keep rho at 1.")
    parser.add_argument("--ddtw", action="store_true", help="Use derivative DTW instead of DTW.")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--band", type=int, default=0, help="Sakoe-Chiba band. 0 selects the monitor default.")
    parser.add_argument("--min-slope", type=float, default=0.0)
    parser.add_argument("--plateau-max", type=int, default=15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--variant",
        default="v0",
        metavar="{v0,v1,v2,v3,v4}",
        help=(
            "One additive variant. v0 is the unchanged baseline. "
            "v1 resolves an omitted scout limit to 100. "
            "v2 restarts post-fire a at 1 and rho at 1.5 on the original horizon. "
            "v3 restarts post-fire a/rho over the cycles still left. "
            "v4 perturbs one ABC coordinate before the unchanged LB2 repair. "
            "Default: v0."
        ),
    )
    return parser


def _validate_cli_monitor(args: argparse.Namespace) -> None:
    if isinstance(args.window, bool) or not isinstance(args.window, int) or args.window < 2:
        raise ValueError("window must be an integer >= 2")
    if isinstance(args.band, bool) or not isinstance(args.band, int) or args.band < 0:
        raise ValueError("band must be an integer >= 0")
    if isinstance(args.min_slope, bool) or not isinstance(args.min_slope, (int, float)):
        raise ValueError("min_slope must be a finite number >= 0")
    if not math.isfinite(float(args.min_slope)) or float(args.min_slope) < 0.0:
        raise ValueError("min_slope must be a finite number >= 0")
    if isinstance(args.plateau_max, bool) or not isinstance(args.plateau_max, int) or args.plateau_max < 0:
        raise ValueError("plateau_max must be an integer >= 0")
    if isinstance(args.patience, bool) or not isinstance(args.patience, int) or args.patience < 1:
        raise ValueError("patience must be an integer >= 1")


def _config_from_args(args: argparse.Namespace) -> HybridConfig:
    _validate_variant(args.variant)
    _validate_cli_monitor(args)
    stag_cfg = None
    if not args.no_stagnation:
        stag_cfg = StagnationConfig(
            window=args.window,
            band=args.band,
            min_slope=args.min_slope,
            plateau_max=args.plateau_max,
            patience=args.patience,
            use_ddtw=bool(args.ddtw),
        )
    return HybridConfig(
        population_size=args.population_size,
        iterations=args.iterations,
        max_evaluations=args.max_evaluations,
        scout_limit=args.scout_limit,
        use_stagnation=not args.no_stagnation,
        stag_cfg=stag_cfg,
        variant=args.variant,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. JSON goes to stdout; errors go to stderr."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else int(code)
    try:
        _validate_variant(args.variant)
        path = _local_instance_file(args.instance)
        config = _config_from_args(args)
        instance, info = _load_local_instance(path, args.instance_index)
        result = solve(instance, config, seed=args.seed, metadata={"instance": info})
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
