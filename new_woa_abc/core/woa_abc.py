"""Motor WOA--ABC continuo independiente del dominio y del sensor DTW."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

from new_woa_abc.config import WOAABCConfig
from new_woa_abc.domains.base import Problem
from new_woa_abc.dtw.monitor import StagnationMonitor
from new_woa_abc.dtw.strategies import (
    AdaptationController,
    AdaptationDecision,
    M0_NO_DTW,
    VARIANT_NAMES,
)

from .profiles import EffectiveParameters, resolve_parameters


@dataclass
class OptimizationResult:
    problem: str
    variant: str
    seed: int
    best_cost: float
    best_position: list[float]
    objective_evaluations: int
    fire_count: int
    transition_count: int
    parameter_update_count: int
    convergence_history: list[float] = field(default_factory=list)
    woa_history: list[float] = field(default_factory=list)
    abc_history: list[float] = field(default_factory=list)
    mode_history: list[str] = field(default_factory=list)
    intensity_history: list[float | None] = field(default_factory=list)
    dtw_history: list[dict[str, Any]] = field(default_factory=list)
    parameter_history: list[dict[str, Any]] = field(default_factory=list)
    control_events: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "variant": self.variant,
            "seed": self.seed,
            "best_cost": self.best_cost,
            "objective_evaluations": self.objective_evaluations,
            "fire_count": self.fire_count,
            "transition_count": self.transition_count,
            "parameter_update_count": self.parameter_update_count,
        }


class WOAABCOptimizer:
    """WOA seguido por ABC sobre una única población compartida."""

    def __init__(self, problem: Problem, config: WOAABCConfig) -> None:
        if config.variant not in VARIANT_NAMES:
            raise ValueError(f"variante desconocida: {config.variant}")
        self.problem = problem
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.population = np.empty((0, problem.dimension), dtype=float)
        self.costs = np.empty(0, dtype=float)
        self.trials = np.empty(0, dtype=int)
        self.best_position = np.empty(problem.dimension, dtype=float)
        self.best_cost = float("inf")
        self.objective_evaluations = 0
        self.previous_best_position = np.empty(problem.dimension, dtype=float)

    def _evaluate(self, position: np.ndarray) -> float:
        self.objective_evaluations += 1
        return self.problem.evaluate(position)

    def _prepare(self, position: np.ndarray) -> np.ndarray:
        return self.problem.repair(position)

    def initialize(self) -> None:
        self.population = self.rng.uniform(
            self.problem.lower,
            self.problem.upper,
            size=(self.config.pop_size, self.problem.dimension),
        )
        self.population = np.asarray(
            [self._prepare(position) for position in self.population],
            dtype=float,
        )
        self.costs = np.asarray(
            [self._evaluate(position) for position in self.population],
            dtype=float,
        )
        self.trials = np.zeros(self.config.pop_size, dtype=int)
        self._update_global_best()
        self.previous_best_position = self.best_position.copy()

    def _update_global_best(self) -> None:
        index = int(np.argmin(self.costs))
        cost = float(self.costs[index])
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_position = self.population[index].copy()

    def _woa_step(self, parameters: EffectiveParameters) -> None:
        source = self.population.copy()
        leader = self.best_position.copy()
        a = parameters.woa_a_effective
        momentum = (
            parameters.momentum_factor
            * parameters.momentum_scale
            * (self.best_position - self.previous_best_position)
        )
        for index in range(self.config.pop_size):
            r1, r2 = self.rng.random(), self.rng.random()
            coefficient_a = 2.0 * a * r1 - a
            coefficient_c = 2.0 * r2
            if self.rng.random() < 0.5:
                if abs(coefficient_a) < 1.0:
                    distance = np.abs(coefficient_c * leader - source[index])
                    candidate = leader - coefficient_a * distance
                else:
                    random_index = int(self.rng.integers(self.config.pop_size))
                    random_whale = source[random_index]
                    distance = np.abs(coefficient_c * random_whale - source[index])
                    candidate = random_whale - coefficient_a * distance
            else:
                distance = np.abs(leader - source[index])
                spiral = self.rng.uniform(-1.0, 1.0)
                candidate = (
                    distance
                    * math.exp(self.config.b_spiral * spiral)
                    * math.cos(2.0 * math.pi * spiral)
                    + leader
                )
            candidate = candidate + parameters.step_effective * momentum
            candidate = self._prepare(candidate)
            candidate_cost = self._evaluate(candidate)
            if candidate_cost <= self.costs[index]:
                self.population[index] = candidate
                self.costs[index] = candidate_cost
                self.trials[index] = 0
        self._update_global_best()

    def _abc_attempt(self, index: int, parameters: EffectiveParameters) -> None:
        neighbour = int(self.rng.integers(self.config.pop_size - 1))
        if neighbour >= index:
            neighbour += 1
        coordinate = int(self.rng.integers(self.problem.dimension))
        candidate = self.population[index].copy()
        phi = self.rng.uniform(-1.0, 1.0) * parameters.abc_phi_effective
        candidate[coordinate] += phi * (
            self.population[index, coordinate] - self.population[neighbour, coordinate]
        )
        if self.rng.random() < parameters.abc_vector_probability:
            candidate += self.rng.normal(
                0.0,
                parameters.abc_vector_scale
                * parameters.step_effective
                * (self.problem.upper - self.problem.lower),
                size=self.problem.dimension,
            )
        candidate += (
            parameters.abc_guide_strength
            * self.rng.random()
            * (self.best_position - candidate)
        )
        candidate = self._prepare(candidate)
        candidate_cost = self._evaluate(candidate)
        if candidate_cost <= self.costs[index]:
            self.population[index] = candidate
            self.costs[index] = candidate_cost
            self.trials[index] = 0
        else:
            self.trials[index] += 1

    def _selection_probabilities(self) -> np.ndarray:
        """Probabilidades por ranking, estables ante escalas CEC distintas."""
        finite = np.isfinite(self.costs)
        if not finite.any():
            return np.full(self.config.pop_size, 1.0 / self.config.pop_size)
        safe = np.where(finite, self.costs, float("inf"))
        order = np.argsort(safe, kind="stable")
        ranks = np.empty(self.config.pop_size, dtype=float)
        ranks[order] = np.arange(self.config.pop_size, dtype=float)
        quality = self.config.pop_size - ranks
        quality[~finite] = 0.0
        total = float(np.sum(quality))
        if total <= 0 or not np.isfinite(total):
            return np.full(self.config.pop_size, 1.0 / self.config.pop_size)
        return quality / total

    def _abc_step(self, parameters: EffectiveParameters) -> None:
        for index in range(self.config.pop_size):
            self._abc_attempt(index, parameters)
        probabilities = self._selection_probabilities()
        selected = self.rng.choice(
            self.config.pop_size,
            size=self.config.pop_size,
            p=probabilities,
        )
        for index in selected:
            self._abc_attempt(int(index), parameters)
        exhausted = np.flatnonzero(self.trials >= parameters.abc_limit_effective)
        if exhausted.size:
            # Un solo scout por iteración conserva la población construida por
            # WOA y ABC y evita reinicios masivos cuando la escala es grande.
            index = int(exhausted[np.argmax(self.trials[exhausted])])
            self.population[index] = self._prepare(self.rng.uniform(
                self.problem.lower,
                self.problem.upper,
            ))
            self.costs[index] = self._evaluate(self.population[index])
            self.trials[index] = 0
        self._update_global_best()

    def run(self, verbose: bool = False) -> OptimizationResult:
        self.initialize()
        monitor = (
            None
            if self.config.variant == M0_NO_DTW
            else StagnationMonitor(self.config.dtw)
        )
        controller = AdaptationController(self.config.variant, self.config.strategy)
        decision = AdaptationDecision("base", None, "initial")

        convergence: list[float] = []
        woa_history: list[float] = []
        abc_history: list[float] = []
        modes: list[str] = []
        intensities: list[float | None] = []
        dtw_history: list[dict[str, Any]] = []
        parameter_history: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        fire_count = 0
        transition_count = 0
        update_count = 0

        for iteration in range(self.config.iterations):
            applied = decision
            parameters = resolve_parameters(
                self.config,
                applied,
                iteration,
                self.problem.dimension,
            )
            best_before_iteration = self.best_position.copy()
            parameter_row = {"iteration": iteration, **parameters.to_dict()}
            parameter_history.append(parameter_row)
            modes.append(applied.mode)
            intensities.append(applied.intensity)

            self._woa_step(parameters)
            woa_best = self.best_cost
            self._abc_step(parameters)
            abc_best = self.best_cost
            self.previous_best_position = best_before_iteration
            woa_history.append(float(woa_best))
            abc_history.append(float(abc_best))
            convergence.append(float(self.best_cost))

            if monitor is None:
                status: dict[str, Any] = {
                    "ready": False,
                    "fire": False,
                    "n": iteration + 1,
                }
            else:
                status = monitor.update(-self.best_cost)
            next_decision = controller.decide(status)
            status = dict(status)
            status.update({
                "iteration": iteration,
                "applied_mode": applied.mode,
                "applied_intensity": applied.intensity,
                "next_mode": next_decision.mode,
                "next_intensity": next_decision.intensity,
                "reason": next_decision.reason,
                "transition": next_decision.transition,
                "parameter_changed": next_decision.parameter_changed,
                "raw_fire": next_decision.raw_fire,
            })
            dtw_history.append(status)

            if next_decision.transition:
                transition_count += 1
            if next_decision.parameter_changed:
                update_count += 1
                events.append({
                    "detected_iteration": iteration,
                    "applied_iteration": iteration + 1,
                    "previous_mode": applied.mode,
                    "next_mode": next_decision.mode,
                    "previous_intensity": applied.intensity,
                    "next_intensity": next_decision.intensity,
                    "reason": next_decision.reason,
                    "raw_fire": next_decision.raw_fire,
                })
            if (
                next_decision.raw_fire
                and next_decision.mode == "explore"
                and applied.mode != "explore"
            ):
                fire_count += 1
            decision = next_decision

            if verbose:
                intensity = "base" if applied.intensity is None else f"{applied.intensity:.3f}"
                print(
                    f"[{self.config.variant}] {iteration + 1:4d}/{self.config.iterations} "
                    f"best={self.best_cost:.10g} mode={applied.mode} intensity={intensity}",
                    flush=True,
                )

        return OptimizationResult(
            problem=self.problem.name,
            variant=self.config.variant,
            seed=self.config.seed,
            best_cost=float(self.best_cost),
            best_position=self.best_position.tolist(),
            objective_evaluations=self.objective_evaluations,
            fire_count=fire_count,
            transition_count=transition_count,
            parameter_update_count=update_count,
            convergence_history=convergence,
            woa_history=woa_history,
            abc_history=abc_history,
            mode_history=modes,
            intensity_history=intensities,
            dtw_history=dtw_history,
            parameter_history=parameter_history,
            control_events=events,
        )


__all__ = ["OptimizationResult", "WOAABCOptimizer"]
