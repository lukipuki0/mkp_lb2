"""WOA--ABC binaria para MKP con binarización LB2 y control DTW/DDTW.

WOA y ABC comparten una única población.  DTW nunca cambia de algoritmo ni
inyecta una trayectoria: observa el mejor beneficio y modifica únicamente los
parámetros que se aplican en la iteración siguiente.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from new_woa_abc.config import WOAABCConfig
from new_woa_abc.domains.mkp import MKPInstance
from new_woa_abc.dtw.monitor import StagnationMonitor
from new_woa_abc.dtw.strategies import (
    AdaptationController,
    AdaptationDecision,
    M0_NO_DTW,
    VARIANT_NAMES,
)

from .profiles import EffectiveParameters, resolve_parameters


@dataclass(frozen=True)
class BinaryWOAABCConfig:
    """Configuración específica de la representación binaria y LB2."""

    algorithm: WOAABCConfig
    v_max: float = 6.0
    g1_initial: float = 0.5
    g1_final: float = 1.0
    g2_initial: float = 0.5
    g2_final: float = 7.2
    g3_initial: float = 0.5
    g3_final: float = 0.0
    greedy_fraction: float = 0.30
    initialization_noise: float = 0.35
    repair_latent_weight: float = 0.15
    equality_acceptance: float = 0.10
    abc_limit_population_factor: float = 1.0
    local_search_interval: int = 0
    local_search_start: int = 50
    local_search_passes: int = 2
    local_search_ejections: int = 15
    local_search_depth: int = 3
    local_search_elites: int = 3
    early_stop_at_best_known: bool = True

    def __post_init__(self) -> None:
        if self.v_max <= 0:
            raise ValueError("v_max debe ser positivo")
        if not 0 <= self.greedy_fraction <= 1:
            raise ValueError("greedy_fraction debe estar entre 0 y 1")
        if self.initialization_noise < 0:
            raise ValueError("initialization_noise no puede ser negativo")
        if not 0 <= self.repair_latent_weight <= 1:
            raise ValueError("repair_latent_weight debe estar entre 0 y 1")
        if not 0 <= self.equality_acceptance <= 1:
            raise ValueError("equality_acceptance debe estar entre 0 y 1")
        if self.abc_limit_population_factor <= 0:
            raise ValueError("abc_limit_population_factor debe ser positivo")
        if self.local_search_interval < 0 or self.local_search_start < 0:
            raise ValueError("intervalo/inicio de búsqueda local no pueden ser negativos")
        if (
            self.local_search_passes < 0
            or self.local_search_ejections < 0
            or self.local_search_depth < 0
            or self.local_search_elites < 0
        ):
            raise ValueError("los parámetros de búsqueda local no pueden ser negativos")


@dataclass
class MKPOptimizationResult:
    problem: str
    variant: str
    seed: int
    best_profit: float
    best_solution: list[int]
    best_known: float | None
    objective_evaluations: int
    iterations_completed: int
    optimum_reached: bool
    stop_reason: str
    fire_count: int
    transition_count: int
    parameter_update_count: int
    local_search_improvements: int
    convergence_history: list[float] = field(default_factory=list)
    woa_history: list[float] = field(default_factory=list)
    abc_history: list[float] = field(default_factory=list)
    mode_history: list[str] = field(default_factory=list)
    intensity_history: list[float | None] = field(default_factory=list)
    dtw_history: list[dict[str, Any]] = field(default_factory=list)
    parameter_history: list[dict[str, Any]] = field(default_factory=list)
    control_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def gap_percent(self) -> float | None:
        if self.best_known is None or self.best_known == 0:
            return None
        return 100.0 * (self.best_known - self.best_profit) / self.best_known

    def summary(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "variant": self.variant,
            "seed": self.seed,
            "best_profit": self.best_profit,
            "best_known": self.best_known,
            "gap_percent": self.gap_percent,
            "optimum_reached": self.optimum_reached,
            "iterations_completed": self.iterations_completed,
            "objective_evaluations": self.objective_evaluations,
            "fire_count": self.fire_count,
            "transition_count": self.transition_count,
            "parameter_update_count": self.parameter_update_count,
            "local_search_improvements": self.local_search_improvements,
        }


def _linear(initial: float, final: float, iteration: int, total: int) -> float:
    if total <= 1:
        return float(final)
    return float(initial + (final - initial) * iteration / (total - 1))


def lb2_probabilities(
    signal: NDArray[np.float64],
    v_max: float,
    g1: float,
    g2: float,
    g3: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Funciones L1/L2 del LB2 original, vectorizadas y numéricamente seguras."""

    denominator = v_max - g2
    if abs(denominator) < 1e-12:
        denominator = math.copysign(1e-12, denominator if denominator else 1.0)
    l1 = np.clip(-g1 * signal / denominator + g3, 0.0, 1.0)
    l2 = np.clip(g1 * signal / denominator + g3, 0.0, 1.0)
    return np.asarray(l1, dtype=float), np.asarray(l2, dtype=float)


class BinaryWOAABCOptimizer:
    """WOA y ABC binarias con élite factible, LB2 y reparación MKP."""

    def __init__(self, problem: MKPInstance, config: BinaryWOAABCConfig) -> None:
        if config.algorithm.variant not in VARIANT_NAMES:
            raise ValueError(f"variante desconocida: {config.algorithm.variant}")
        self.problem = problem
        self.binary_config = config
        self.config = config.algorithm
        self.rng = np.random.default_rng(self.config.seed)
        self.local_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 0x4D4B50])
        )
        self.positions = np.empty((0, problem.dimension), dtype=float)
        self.solutions = np.empty((0, problem.dimension), dtype=np.int8)
        self.profits = np.empty(0, dtype=float)
        self.trials = np.empty(0, dtype=int)
        self.best_position = np.empty(problem.dimension, dtype=float)
        self.best_solution = np.empty(problem.dimension, dtype=np.int8)
        self.best_profit = -float("inf")
        self.previous_best_position = np.empty(problem.dimension, dtype=float)
        self.objective_evaluations = 0
        self.local_search_improvements = 0
        self.polished_solutions: set[bytes] = set()

    def _priority(self, latent: NDArray[np.float64]) -> NDArray[np.float64]:
        modulation = 1.0 + self.binary_config.repair_latent_weight * np.tanh(
            np.asarray(latent, dtype=float) / self.binary_config.v_max
        )
        return self.problem.efficiency * modulation

    def _repair(
        self,
        raw: NDArray[np.int8],
        latent: NDArray[np.float64],
    ) -> tuple[NDArray[np.int8], float]:
        solution, value = self.problem.repair(raw, self._priority(latent))
        self.objective_evaluations += 1
        return solution, value

    def _latent_from_solution(
        self,
        solution: NDArray[np.int8],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        generator = self.rng if rng is None else rng
        magnitude = generator.uniform(
            0.5 * self.binary_config.v_max,
            self.binary_config.v_max,
            self.problem.dimension,
        )
        return np.where(solution > 0, magnitude, -magnitude)

    def initialize(self) -> None:
        pop_size = self.config.pop_size
        n = self.problem.dimension
        self.positions = self.rng.uniform(
            -self.binary_config.v_max,
            self.binary_config.v_max,
            size=(pop_size, n),
        )
        self.solutions = np.zeros((pop_size, n), dtype=np.int8)
        self.profits = np.zeros(pop_size, dtype=float)
        self.trials = np.zeros(pop_size, dtype=int)

        greedy_count = max(1, int(round(pop_size * self.binary_config.greedy_fraction)))
        for index in range(pop_size):
            if index < greedy_count:
                noise = self.binary_config.initialization_noise * index / max(1, greedy_count - 1)
                solution, value = self.problem.greedy_solution(self.rng, noise=noise)
                self.objective_evaluations += 1
                self.positions[index] = self._latent_from_solution(solution)
            else:
                probability = 1.0 / (1.0 + np.exp(-self.positions[index]))
                raw = (self.rng.random(n) < probability).astype(np.int8)
                solution, value = self._repair(raw, self.positions[index])
            self.solutions[index] = solution
            self.profits[index] = value

        self._update_global_best()
        self.previous_best_position = self.best_position.copy()

    def _update_global_best(self) -> None:
        index = int(np.argmax(self.profits))
        value = float(self.profits[index])
        if value > self.best_profit + 1e-12:
            self.best_profit = value
            self.best_solution = self.solutions[index].copy()
            self.best_position = self.positions[index].copy()

    def _ensure_global_elite(self) -> None:
        """Mantiene una copia exacta del mejor global dentro de la población."""

        if any(
            np.array_equal(solution, self.best_solution)
            for solution in self.solutions
        ):
            return
        index = int(np.argmin(self.profits))
        self.positions[index] = self.best_position.copy()
        self.solutions[index] = self.best_solution.copy()
        self.profits[index] = self.best_profit
        self.trials[index] = 0

    def _guide_bits(
        self,
        solution: NDArray[np.int8],
        strength: float,
    ) -> NDArray[np.int8]:
        if strength <= 0 or not self.best_solution.size:
            return solution
        # La guía es moderada: ABC sigue explorando y el mejor solo fija una
        # fracción de los bits distintos.
        probability = min(0.35, 0.50 * strength)
        different = solution != self.best_solution
        copy = different & (self.rng.random(self.problem.dimension) < probability)
        guided = solution.copy()
        guided[copy] = self.best_solution[copy]
        return guided

    def _binarize_lb2(
        self,
        signal: NDArray[np.float64],
        current: NDArray[np.int8],
        latent: NDArray[np.float64],
        g1: float,
        g2: float,
        g3: float,
        guide_strength: float,
    ) -> tuple[NDArray[np.int8], float]:
        l1, l2 = lb2_probabilities(
            np.asarray(signal, dtype=float),
            self.binary_config.v_max,
            g1,
            g2,
            g3,
        )
        candidates: list[tuple[NDArray[np.int8], float]] = []
        for probabilities in (l1, l2):
            raw = current.copy()
            flips = self.rng.random(self.problem.dimension) < probabilities
            raw[flips] = 1 - raw[flips]
            raw = self._guide_bits(raw, guide_strength)
            candidates.append(self._repair(raw, latent))
        if candidates[0][1] > candidates[1][1] + 1e-12:
            return candidates[0]
        if candidates[1][1] > candidates[0][1] + 1e-12:
            return candidates[1]
        return candidates[int(self.rng.integers(2))]

    def _accept(
        self,
        index: int,
        latent: NDArray[np.float64],
        solution: NDArray[np.int8],
        value: float,
        count_failure: bool,
    ) -> None:
        improvement = value > self.profits[index] + 1e-12
        tie = abs(value - self.profits[index]) <= 1e-12
        if improvement or (tie and self.rng.random() < self.binary_config.equality_acceptance):
            self.positions[index] = latent
            self.solutions[index] = solution
            self.profits[index] = value
        if improvement:
            self.trials[index] = 0
        elif count_failure:
            self.trials[index] += 1

    def _woa_step(
        self,
        parameters: EffectiveParameters,
        g_values: tuple[float, float, float],
    ) -> None:
        source = self.positions.copy()
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
            candidate += parameters.step_effective * momentum
            candidate = np.clip(candidate, -self.binary_config.v_max, self.binary_config.v_max)
            solution, value = self._binarize_lb2(
                candidate,
                self.solutions[index],
                candidate,
                *g_values,
                parameters.abc_guide_strength,
            )
            self._accept(index, candidate, solution, value, count_failure=False)
        self._update_global_best()
        self._ensure_global_elite()

    def _abc_attempt(
        self,
        index: int,
        parameters: EffectiveParameters,
        g_values: tuple[float, float, float],
    ) -> None:
        neighbour = int(self.rng.integers(self.config.pop_size - 1))
        if neighbour >= index:
            neighbour += 1
        candidate = self.positions[index].copy()
        coordinate = int(self.rng.integers(self.problem.dimension))
        phi = self.rng.uniform(-1.0, 1.0) * parameters.abc_phi_effective
        candidate[coordinate] += phi * (
            self.positions[index, coordinate] - self.positions[neighbour, coordinate]
        )
        if self.rng.random() < parameters.abc_vector_probability:
            candidate += self.rng.normal(
                0.0,
                parameters.abc_vector_scale * parameters.step_effective * self.binary_config.v_max,
                self.problem.dimension,
            )
        candidate += (
            parameters.abc_guide_strength
            * self.rng.random()
            * (self.best_position - candidate)
        )
        candidate = np.clip(candidate, -self.binary_config.v_max, self.binary_config.v_max)
        solution, value = self._binarize_lb2(
            candidate,
            self.solutions[index],
            candidate,
            *g_values,
            parameters.abc_guide_strength,
        )
        self._accept(index, candidate, solution, value, count_failure=True)

    def _selection_probabilities(self) -> NDArray[np.float64]:
        order = np.argsort(-self.profits, kind="stable")
        ranks = np.empty(self.config.pop_size, dtype=float)
        ranks[order] = np.arange(self.config.pop_size, dtype=float)
        quality = self.config.pop_size - ranks
        return quality / np.sum(quality)

    def _scout(self, index: int) -> None:
        latent = self.rng.uniform(
            -self.binary_config.v_max,
            self.binary_config.v_max,
            self.problem.dimension,
        )
        raw = self.rng.integers(0, 2, self.problem.dimension, dtype=np.int8)
        solution, value = self._repair(raw, latent)
        self.positions[index] = latent
        self.solutions[index] = solution
        self.profits[index] = value
        self.trials[index] = 0

    def _abc_step(
        self,
        parameters: EffectiveParameters,
        g_values: tuple[float, float, float],
    ) -> None:
        for index in range(self.config.pop_size):
            self._abc_attempt(index, parameters, g_values)
        selected = self.rng.choice(
            self.config.pop_size,
            size=self.config.pop_size,
            p=self._selection_probabilities(),
        )
        for index in selected:
            self._abc_attempt(int(index), parameters, g_values)

        exhausted = np.flatnonzero(self.trials >= parameters.abc_limit_effective)
        if exhausted.size:
            non_elite = exhausted[self.profits[exhausted] < self.best_profit - 1e-12]
            if non_elite.size:
                index = int(non_elite[np.argmax(self.trials[non_elite])])
                self._scout(index)
        self._update_global_best()
        self._ensure_global_elite()

    def _polish_elites(self) -> None:
        elite_count = min(self.binary_config.local_search_elites, self.config.pop_size)
        if elite_count <= 0:
            return
        selected: list[int] = []
        seen_now: set[bytes] = set()
        for raw_index in np.argsort(-self.profits, kind="stable"):
            index = int(raw_index)
            signature = self.solutions[index].tobytes()
            if signature in seen_now:
                continue
            seen_now.add(signature)
            selected.append(index)
            if len(selected) >= elite_count:
                break

        for index in selected:
            signature = self.solutions[index].tobytes()
            if signature in self.polished_solutions:
                continue
            self.polished_solutions.add(signature)
            solution, value, evaluations = self.problem.improve(
                self.solutions[index],
                self.local_rng,
                passes=self.binary_config.local_search_passes,
                ejection_candidates=self.binary_config.local_search_ejections,
                exchange_depth=self.binary_config.local_search_depth,
            )
            self.objective_evaluations += evaluations
            if value <= self.profits[index] + 1e-12:
                continue
            self.local_search_improvements += 1
            latent = self._latent_from_solution(solution, self.local_rng)
            self.positions[index] = latent
            self.solutions[index] = solution
            self.profits[index] = value
            self.trials[index] = 0
        self._update_global_best()

    def _optimum_reached(self) -> bool:
        return bool(
            self.problem.best_known is not None
            and self.best_profit >= self.problem.best_known - 1e-9
        )

    def run(self, verbose: bool = False) -> MKPOptimizationResult:
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
        stop_reason = "iteration_budget"

        if self.binary_config.early_stop_at_best_known and self._optimum_reached():
            stop_reason = "best_known_reached_at_initialization"

        for iteration in range(self.config.iterations):
            if stop_reason != "iteration_budget":
                break
            applied = decision
            parameters = resolve_parameters(
                self.config,
                applied,
                iteration,
                self.problem.dimension,
            )
            if self.config.abc_limit is None:
                # En MKP cada intento cambia un vector binario completo. Por
                # eso el límite se escala con la población y no con n ítems,
                # como sí ocurre en ABC continuo por coordenadas.
                limit_base = max(
                    self.config.min_abc_limit,
                    int(round(
                        self.config.pop_size
                        * self.binary_config.abc_limit_population_factor
                    )),
                )
                parameters = replace(
                    parameters,
                    abc_limit_base=limit_base,
                    abc_limit_effective=max(
                        self.config.min_abc_limit,
                        int(round(limit_base * parameters.abc_limit_factor)),
                    ),
                )
            g_values = (
                _linear(self.binary_config.g1_initial, self.binary_config.g1_final, iteration, self.config.iterations),
                _linear(self.binary_config.g2_initial, self.binary_config.g2_final, iteration, self.config.iterations),
                _linear(self.binary_config.g3_initial, self.binary_config.g3_final, iteration, self.config.iterations),
            )
            best_before_iteration = self.best_position.copy()
            parameter_history.append({
                "iteration": iteration,
                **parameters.to_dict(),
                "lb2_g1": g_values[0],
                "lb2_g2": g_values[1],
                "lb2_g3": g_values[2],
            })
            modes.append(applied.mode)
            intensities.append(applied.intensity)

            self._woa_step(parameters, g_values)
            woa_best = self.best_profit
            self._abc_step(parameters, g_values)
            abc_best = self.best_profit
            if (
                self.binary_config.local_search_interval > 0
                and iteration + 1 >= self.binary_config.local_search_start
                and (
                    iteration + 1 - self.binary_config.local_search_start
                ) % self.binary_config.local_search_interval == 0
            ):
                self._polish_elites()
                abc_best = self.best_profit
            self.previous_best_position = best_before_iteration
            woa_history.append(float(woa_best))
            abc_history.append(float(abc_best))
            convergence.append(float(self.best_profit))

            if monitor is None:
                status: dict[str, Any] = {
                    "ready": False,
                    "fire": False,
                    "n": iteration + 1,
                }
            else:
                # El monitor original es de maximización: MKP entra directo.
                status = monitor.update(self.best_profit)
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
                    f"profit={self.best_profit:.10g} mode={applied.mode} intensity={intensity}",
                    flush=True,
                )

            if self.binary_config.early_stop_at_best_known and self._optimum_reached():
                stop_reason = "best_known_reached"

        if not self.problem.is_feasible(self.best_solution):
            raise RuntimeError("el mejor MKP final es infactible")

        return MKPOptimizationResult(
            problem=self.problem.name,
            variant=self.config.variant,
            seed=self.config.seed,
            best_profit=float(self.best_profit),
            best_solution=self.best_solution.astype(int).tolist(),
            best_known=self.problem.best_known,
            objective_evaluations=self.objective_evaluations,
            iterations_completed=len(convergence),
            optimum_reached=self._optimum_reached(),
            stop_reason=stop_reason,
            fire_count=fire_count,
            transition_count=transition_count,
            parameter_update_count=update_count,
            local_search_improvements=self.local_search_improvements,
            convergence_history=convergence,
            woa_history=woa_history,
            abc_history=abc_history,
            mode_history=modes,
            intensity_history=intensities,
            dtw_history=dtw_history,
            parameter_history=parameter_history,
            control_events=events,
        )


__all__ = [
    "BinaryWOAABCConfig",
    "BinaryWOAABCOptimizer",
    "MKPOptimizationResult",
    "lb2_probabilities",
]
