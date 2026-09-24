"""
Runner Vanilla: orquesta el loop básico de la MH sin DTW ni adaptación.
Las MH corren con sus parámetros por defecto (modo exploit estático).
"""

import time
from typing import List, Type

import numpy as np

from mkp_common.base import BaseMH


def run_experiment(
    mh_class: Type[BaseMH],
    inst: dict,
    num_particulas: int = 20,
    num_iteraciones: int = 100,
    semilla: int = 42,
) -> dict:
    """
    Ejecuta UNA corrida de la MH estándar (sin adaptar parámetros).
    """
    rng = np.random.default_rng(semilla)
    mh = mh_class(inst, rng, num_particulas=num_particulas)

    mh.initialize()

    historial_fitness: List[float] = []
    historial_modos: List[str] = []

    for _ in range(num_iteraciones):
        fitness = mh.step()
        historial_fitness.append(fitness)
        historial_modos.append(mh.mode)

    sol, fit = mh.get_best()
    optimo = inst["optimo"]

    return {
        "mejor_fitness": fit,
        "mejor_solucion": sol,
        "optimo_conocido": optimo,
        "historial_fitness": historial_fitness,
        "historial_modos": historial_modos,
        "ganancia": (fit / optimo * 100) if optimo > 0 else 0,
        "fire_count": 0,
    }


def run_epochs(
    mh_class: Type[BaseMH],
    inst: dict,
    num_particulas: int = 20,
    num_iteraciones: int = 100,
    epochs: int = 10,
) -> List[dict]:
    """
    Ejecuta múltiples epochs sin DTW.
    """
    resultados = []
    for ep in range(epochs):
        semilla = ep + 1
        t0 = time.perf_counter()
        res = run_experiment(
            mh_class, inst,
            num_particulas=num_particulas,
            num_iteraciones=num_iteraciones,
            semilla=semilla,
        )
        t1 = time.perf_counter()
        res["tiempo"] = t1 - t0
        res["epoch"] = ep + 1
        res["semilla"] = semilla
        resultados.append(res)
    return resultados
