"""Adaptador exclusivo del modelo HRES2-H2/WPEB."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .base import Problem


def load_hres2_problem() -> tuple[Problem, Any]:
    module = importlib.import_module("HRES2-H2.wpeb_model")
    model = module.HRES2Function()

    def decode(position: np.ndarray) -> dict:
        return dict(module.decode_solution(position, model.config))

    def repair(position: np.ndarray) -> np.ndarray:
        """Canonicaliza la codificación mixta antes de evaluar HRES2."""
        value = np.asarray(position, dtype=float).copy()
        value[1] = np.rint(value[1])
        value[2] = np.rint(value[2] / 5.0) * 5.0
        value[3] = np.rint(value[3])
        return value

    problem = Problem(
        name=str(model.name),
        dimension=int(model.n_dim),
        lower=np.asarray(model.lb_vector, dtype=float),
        upper=np.asarray(model.ub_vector, dtype=float),
        objective=model.func,
        # HRES2 no dispone de un óptimo analítico certificado. El valor 0.30
        # del wrapper original es solo una referencia aproximada y no debe
        # dibujarse ni usarse como error de optimalidad.
        optimum=None,
        decode_fn=decode,
        metrics_fn=model.get_info,
        repair_fn=repair,
    )
    return problem, model


__all__ = ["load_hres2_problem"]
