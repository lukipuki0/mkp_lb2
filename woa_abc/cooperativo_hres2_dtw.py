"""Motor WOA--ABC adaptativo exclusivo para HRES2-H2/WPEB.

Este motor conserva los límites vectoriales, la evaluación y la decodificación
propias de HRES2. Comparte con CEC únicamente el controlador de parámetros DTW.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dtw_stagnation import StagnationConfig
from woa_abc.adaptive import (
    AdaptiveParameterController,
    M0_NO_DTW,
    M1_LEGACY_FIRE_FULL,
    VARIANT_NAMES,
    validate_variant,
)


@dataclass
class CooperativeHRES2Params:
    """Hiperparámetros del WOA--ABC exclusivo para HRES2."""

    pop_size: int = 30
    iterations: int = 300
    variant: str = M1_LEGACY_FIRE_FULL
    b_spiral: float = 1.0
    woa_exploration: float = 1.0
    abc_limit: int | None = None
    abc_guide_strength: float = 0.20
    abc_phi_scale: float = 1.0
    exploration_probability: float = 0.50
    step_initial: float = 1.0
    step_final: float = 0.05
    momentum_factor: float = 0.20
    abc_limit_divisor: int = 4
    use_dtw: bool = True
    stag_cfg: StagnationConfig = field(default_factory=StagnationConfig)
    adaptation_cooldown: int = 3
    adaptation_decay: float = 0.90
    max_woa_exploration: float = 2.50
    max_abc_phi_scale: float = 2.50
    min_abc_limit: int = 2
    seed: int | None = None

    def __post_init__(self) -> None:
        validate_variant(self.variant)
        if self.pop_size < 2:
            raise ValueError("pop_size debe ser al menos 2")
        if self.iterations < 1:
            raise ValueError("iterations debe ser positivo")
        if self.b_spiral <= 0:
            raise ValueError("b_spiral debe ser positivo")
        if self.abc_limit is not None and self.abc_limit < 1:
            raise ValueError("abc_limit debe ser positivo")
        if self.abc_guide_strength < 0 or self.abc_phi_scale < 0:
            raise ValueError("los parámetros ABC no pueden ser negativos")
        if not 0.0 <= self.exploration_probability <= 1.0:
            raise ValueError("exploration_probability debe estar entre 0 y 1")
        if self.step_initial <= 0 or self.step_final <= 0:
            raise ValueError("step_initial y step_final deben ser positivos")
        if self.momentum_factor < 0:
            raise ValueError("momentum_factor no puede ser negativo")
        if self.abc_limit_divisor < 1:
            raise ValueError("abc_limit_divisor debe ser positivo")
        if self.adaptation_cooldown < 0:
            raise ValueError("adaptation_cooldown no puede ser negativo")
        if not 0.0 <= self.adaptation_decay <= 1.0:
            raise ValueError("adaptation_decay debe estar entre 0 y 1")
        if self.max_woa_exploration < 1 or self.max_abc_phi_scale < 1:
            raise ValueError("los máximos adaptativos deben ser al menos 1")
        if self.min_abc_limit < 1:
            raise ValueError("min_abc_limit debe ser positivo")

    @property
    def effective_variant(self) -> str:
        return self.variant if self.use_dtw else M0_NO_DTW


@dataclass
class CooperativeHRES2EpochResult:
    epoch_idx: int
    variant: str
    mejor_valor: float
    iteraciones: int
    stagnation_fires: int
    objective_evaluations: int
    mode_transitions: int
    parameter_updates: int
    mejor_solucion: list[float] = field(default_factory=list)
    mejor_info: dict[str, Any] = field(default_factory=dict)
    historial: list[float] = field(default_factory=list)
    historial_inst: list[float] = field(default_factory=list)
    historial_woa: list[float] = field(default_factory=list)
    historial_abc: list[float] = field(default_factory=list)
    dtw_deltas: list[float] = field(default_factory=list)
    dtw_info_hist: list[dict[str, Any]] = field(default_factory=list)
    eventos_cooperacion: list[dict[str, Any]] = field(default_factory=list)
    eventos_adaptacion: list[dict[str, Any]] = field(default_factory=list)
    parametros_historial: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CooperativeHRES2Result:
    variant: str
    epochs: list[CooperativeHRES2EpochResult]
    mejor_valor_global: float
    mejor_sol_global: list[float]
    mejor_info_global: dict[str, Any]

    @property
    def objective_evaluations(self) -> int:
        return sum(epoch.objective_evaluations for epoch in self.epochs)


def _function_bounds(func: Any) -> tuple[int, np.ndarray, np.ndarray]:
    try:
        n_dim = int(func.n_dim)
        lower = np.asarray(func.lb_vector, dtype=float)
        upper = np.asarray(func.ub_vector, dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("HRES2Function debe exponer n_dim, lb_vector y ub_vector") from exc
    if lower.shape != (n_dim,) or upper.shape != (n_dim,):
        raise ValueError("los límites HRES2 deben tener exactamente n_dim componentes")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("HRES2Function debe definir límites vectoriales finitos válidos")
    return n_dim, lower, upper


def _evaluate(func: Any, position: np.ndarray) -> float:
    value = float(func.func(np.asarray(position, dtype=float)))
    return value if np.isfinite(value) else float("inf")


def _get_info(func: Any, position: np.ndarray) -> dict[str, Any]:
    if not hasattr(func, "get_info"):
        return {}
    return dict(func.get_info(np.asarray(position, dtype=float)))


def _selection_probabilities(costs: np.ndarray) -> np.ndarray:
    finite = np.isfinite(costs)
    if not finite.any():
        return np.full(costs.size, 1.0 / costs.size)
    safe = costs.copy()
    worst = float(np.max(safe[finite]))
    safe[~finite] = worst + max(1.0, abs(worst))
    shifted = np.maximum(0.0, safe - float(np.min(safe)))
    quality = 1.0 / (1.0 + shifted)
    total = float(np.sum(quality))
    return quality / total if total > 0 and np.isfinite(total) else np.full(costs.size, 1.0 / costs.size)


def ejecutar_hres2_epoch(
    func: Any,
    params: CooperativeHRES2Params,
    epoch_idx: int = 0,
    verbose: bool = True,
) -> CooperativeHRES2EpochResult:
    """Ejecuta un epoch WOA--ABC con el contrato propio de HRES2."""

    n_dim, lower, upper = _function_bounds(func)
    seed = None if params.seed is None else params.seed + epoch_idx
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    population = np_rng.uniform(lower, upper, size=(params.pop_size, n_dim))
    costs = np.asarray([_evaluate(func, row) for row in population], dtype=float)
    objective_evaluations = params.pop_size
    trials = np.zeros(params.pop_size, dtype=int)
    best_index = int(np.argmin(costs))
    best_position = population[best_index].copy()
    best_value = float(costs[best_index])
    previous_best_position = best_position.copy()
    base_limit = params.abc_limit
    if base_limit is None:
        base_limit = max(
            params.min_abc_limit,
            params.pop_size * n_dim // params.abc_limit_divisor,
        )
    base_limit = max(params.min_abc_limit, int(base_limit))

    controller = AdaptiveParameterController(
        params.effective_variant,
        params.stag_cfg,
        base_phi_scale=params.abc_phi_scale,
        base_guide_strength=params.abc_guide_strength,
        cooldown=params.adaptation_cooldown,
        decay=params.adaptation_decay,
        max_woa_scale=params.max_woa_exploration,
        max_phi_scale=params.max_abc_phi_scale,
    )
    decision = controller.initial_decision()
    history: list[float] = []
    history_inst: list[float] = []
    history_woa: list[float] = []
    history_abc: list[float] = []
    dtw_deltas: list[float] = []
    dtw_info: list[dict[str, Any]] = []
    cooperation: list[dict[str, Any]] = []
    adaptations: list[dict[str, Any]] = []
    parameters_history: list[dict[str, Any]] = []
    accepted_fires = 0

    def update_global() -> None:
        nonlocal best_position, best_value
        index = int(np.argmin(costs))
        value = float(costs[index])
        if value < best_value:
            best_position = population[index].copy()
            best_value = value

    for iteration in range(params.iterations):
        profile = decision.profile
        applied_mode = decision.mode.value
        momentum = best_position - previous_best_position
        previous_best_position = best_position.copy()
        base_step = params.step_initial * (
            params.step_final / params.step_initial
        ) ** ((iteration + 1) / params.iterations)
        step_size = base_step * profile.step_scale
        momentum_term = params.momentum_factor * profile.momentum_scale * momentum
        effective_limit = max(
            params.min_abc_limit,
            int(round(base_limit * profile.abc_limit_factor)),
        )
        a_base = (2.0 - 2.0 * iteration / max(1, params.iterations - 1)) * params.woa_exploration
        a = min(profile.woa_a_cap, max(profile.woa_a_floor, a_base * profile.woa_a_scale))
        parameters_history.append({
            "iteracion": iteration,
            "variant": params.effective_variant,
            "mode": applied_mode,
            "profile": profile.name,
            "step_size": float(step_size),
            "momentum_norm": float(np.linalg.norm(momentum)),
            "woa_a": float(a),
            "woa_a_scale": float(profile.woa_a_scale),
            "abc_phi_scale": float(profile.abc_phi_scale),
            "abc_guide_strength": float(profile.abc_guide_strength),
            "abc_limit": int(effective_limit),
        })

        woa_before = float(np.min(costs))
        for index in range(params.pop_size):
            r1, r2 = py_rng.random(), py_rng.random()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            if py_rng.random() < params.exploration_probability:
                if abs(A) < 1.0:
                    distance = np.abs(C * best_position - population[index])
                    candidate = best_position - A * distance
                else:
                    random_index = py_rng.randrange(params.pop_size)
                    distance = np.abs(C * population[random_index] - population[index])
                    candidate = population[random_index] - A * distance
            else:
                distance = np.abs(best_position - population[index])
                spiral_l = py_rng.uniform(-1.0, 1.0)
                candidate = (
                    distance * math.exp(params.b_spiral * spiral_l)
                    * math.cos(2.0 * math.pi * spiral_l)
                    + best_position
                )
            candidate = np.clip(candidate + step_size * momentum_term, lower, upper)
            value = _evaluate(func, candidate)
            objective_evaluations += 1
            if value <= costs[index]:
                population[index] = candidate
                costs[index] = value
                trials[index] = 0
            else:
                trials[index] += 1
        update_global()
        woa_best = float(np.min(costs))
        history_woa.append(woa_best)

        def abc_attempt(index: int) -> None:
            nonlocal objective_evaluations
            neighbour = py_rng.randrange(params.pop_size - 1)
            if neighbour >= index:
                neighbour += 1
            coordinate = py_rng.randrange(n_dim)
            candidate = population[index].copy()
            phi = py_rng.uniform(-1.0, 1.0) * profile.abc_phi_scale
            candidate[coordinate] += phi * (
                population[index, coordinate] - population[neighbour, coordinate]
            )
            candidate += step_size * momentum_term
            candidate += profile.abc_guide_strength * py_rng.random() * (
                best_position - population[index]
            )
            candidate = np.clip(candidate, lower, upper)
            value = _evaluate(func, candidate)
            objective_evaluations += 1
            if value <= costs[index]:
                population[index] = candidate
                costs[index] = value
                trials[index] = 0
            else:
                trials[index] += 1

        for index in range(params.pop_size):
            abc_attempt(index)
        update_global()
        selected = np_rng.choice(
            params.pop_size,
            size=params.pop_size,
            p=_selection_probabilities(costs),
        )
        for index in selected:
            abc_attempt(int(index))
        update_global()

        scout_index = int(np.argmax(trials))
        if trials[scout_index] >= effective_limit:
            scout = np_rng.uniform(lower, upper, size=n_dim)
            population[scout_index] = scout
            costs[scout_index] = _evaluate(func, scout)
            objective_evaluations += 1
            trials[scout_index] = 0
        update_global()

        abc_best = float(np.min(costs))
        history_abc.append(abc_best)
        history.append(best_value)
        history_inst.append(abc_best)
        cooperation.append({
            "iteracion": iteration,
            "tipo": "handoff_woa_abc_hres2",
            "woa_best": woa_best,
            "abc_best": abc_best,
            "global": best_value,
            "woa_mejora": woa_best < woa_before,
        })

        next_decision, status = controller.observe(best_value)
        if next_decision.fire_accepted:
            accepted_fires += 1
        dtw_deltas.append(float(status.get("delta", float("nan"))))
        info = dict(status)
        info.update({
            "iteracion": iteration,
            "mode_aplicado": applied_mode,
            "profile_aplicado": profile.name,
            "mode_siguiente": next_decision.mode.value,
            "profile_siguiente": next_decision.profile.name,
            "parametros_actualizados": next_decision.changed,
            "transicion": next_decision.transition,
            "fire_aceptado": next_decision.fire_accepted,
            "fase": "poblacion_compartida_hres2",
            "woa_best": woa_best,
            "abc_best": abc_best,
        })
        dtw_info.append(info)
        if next_decision.changed or next_decision.transition or next_decision.fire_accepted:
            adaptations.append({
                "iteracion": iteration,
                "motivo": next_decision.reason,
                "mode_anterior": applied_mode,
                "mode_siguiente": next_decision.mode.value,
                "perfil_anterior": profile.to_dict(),
                "perfil_siguiente": next_decision.profile.to_dict(),
                "dtw_delta": status.get("delta"),
                "fire_aceptado": next_decision.fire_accepted,
            })
        decision = next_decision

        if verbose:
            print(
                f"  [WOA-ABC HRES2:{params.effective_variant}] "
                f"Iter {iteration + 1:4d}/{params.iterations} | "
                f"WOA={woa_best:.6g} | ABC={abc_best:.6g} | "
                f"Global={best_value:.6g} | modo={applied_mode}",
                flush=True,
            )

    return CooperativeHRES2EpochResult(
        epoch_idx=epoch_idx,
        variant=params.effective_variant,
        mejor_valor=best_value,
        iteraciones=len(history),
        stagnation_fires=accepted_fires,
        objective_evaluations=objective_evaluations,
        mode_transitions=controller.transition_count,
        parameter_updates=controller.parameter_update_count,
        mejor_solucion=best_position.tolist(),
        mejor_info=_get_info(func, best_position),
        historial=history,
        historial_inst=history_inst,
        historial_woa=history_woa,
        historial_abc=history_abc,
        dtw_deltas=dtw_deltas,
        dtw_info_hist=dtw_info,
        eventos_cooperacion=cooperation,
        eventos_adaptacion=adaptations,
        parametros_historial=parameters_history,
    )


def ejecutar_hres2_cooperativo(
    func: Any,
    params: CooperativeHRES2Params,
    verbose: bool = True,
) -> CooperativeHRES2Result:
    """Ejecuta una única traza continua de una variante sobre HRES2."""

    epochs = [ejecutar_hres2_epoch(func, params, epoch_idx=0, verbose=verbose)]
    best_epoch = epochs[0]
    return CooperativeHRES2Result(
        variant=params.effective_variant,
        epochs=epochs,
        mejor_valor_global=best_epoch.mejor_valor,
        mejor_sol_global=best_epoch.mejor_solucion.copy(),
        mejor_info_global=dict(best_epoch.mejor_info),
    )


__all__ = [
    "CooperativeHRES2Params",
    "CooperativeHRES2EpochResult",
    "CooperativeHRES2Result",
    "VARIANT_NAMES",
    "ejecutar_hres2_cooperativo",
    "ejecutar_hres2_epoch",
]
