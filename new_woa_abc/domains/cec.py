"""Adaptador exclusivo de las funciones CEC2022."""

from __future__ import annotations

import numpy as np

from continuous_benchmark.funciones_cec2022 import get_test_functions

from .base import Problem


def load_cec_problems(dimension: int) -> list[Problem]:
    if dimension not in {10, 20}:
        raise ValueError("los experimentos nuevos admiten CEC D=10 o D=20")
    problems: list[Problem] = []
    for function in get_test_functions(dimension):
        lower = np.full(dimension, float(function.lb), dtype=float)
        upper = np.full(dimension, float(function.ub), dtype=float)
        problems.append(
            Problem(
                name=function.name,
                dimension=dimension,
                lower=lower,
                upper=upper,
                objective=function.func,
                optimum=float(function.optimum),
            )
        )
    return problems


__all__ = ["load_cec_problems"]
