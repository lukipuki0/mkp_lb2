"""
Runner: orquesta MH + DTW con adaptación continua (sigmoid B1).
"""

import math
import time
from typing import Dict, List, Type

import numpy as np

from mkp_common.base import BaseMH
from mkp_common.monitor import StagnationConfig, StagnationMonitor


def _sigmoid(x: float) -> float:
    """Sigmoid: mapea (−∞, +∞) → (0, 1)."""
    if x < -100:
        return 0.0
    if x > 100:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def run_experiment(
    mh_class: Type[BaseMH],
    inst: dict,
    monitor_cfg: StagnationConfig,
    num_particulas: int = 20,
    num_iteraciones: int = 100,
    semilla: int = 42,
    verbose: bool = False,
    k: float = 5.0,
    center: float = 0.5,
) -> dict:
    """
    Ejecuta UNA corrida de la MH con DTW B1 sigmoid.

    Args:
        mh_class:        Clase de la MH
        inst:            Instancia MKP cargada
        monitor_cfg:     Configuración del DTW
        num_particulas:  Tamaño de la población
        num_iteraciones: Presupuesto de iteraciones
        semilla:         Seed para reproducibilidad
        verbose:         Imprimir diagnóstico del DTW
        k:               Steepness del sigmoid (default 5.0)
        center:          Inflection point en r_balance (default 0.5)

    Returns:
        dict con resultados de la corrida
    """
    rng = np.random.default_rng(semilla)
    mh = mh_class(inst, rng, num_particulas=num_particulas)
    monitor = StagnationMonitor(cfg=monitor_cfg)

    mh.initialize()

    historial_fitness: List[float] = []
    historial_dtw: List[Dict] = []
    historial_modos: List[str] = []
    historial_intensity: List[float] = []

    eps = 1e-10

    for it in range(num_iteraciones):
        fitness = mh.step()

        out = monitor.update(fitness)
        historial_dtw.append(out)

        if out.get("ready"):
            r_balance = out["delta"] / (out["theta_delta"] + eps)
            intensity = _sigmoid(k * (r_balance - center))
            mh.adapt_continuous(intensity)

            historial_intensity.append(float(intensity))

            if verbose:
                print(
                    f"  [B1] iter={it:03d} | "
                    f"delta={out['delta']:+.1f} | "
                    f"r_bal={r_balance:.2f} | "
                    f"intensity={intensity:.3f} | "
                    f"mode={mh.mode}"
                )
        else:
            historial_intensity.append(0.0)

        historial_modos.append(mh.mode)
        historial_fitness.append(fitness)

    sol, fit = mh.get_best()
    optimo = inst["optimo"]

    return {
        "mejor_fitness": fit,
        "mejor_solucion": sol,
        "optimo_conocido": optimo,
        "historial_fitness": historial_fitness,
        "historial_dtw": historial_dtw,
        "historial_modos": historial_modos,
        "historial_intensity": historial_intensity,
        "intensity_promedio": float(np.mean(historial_intensity)) if historial_intensity else 0.0,
        "fire_count": 0,  # No aplica
        "ganancia": (fit / optimo * 100) if optimo > 0 else 0,
    }


def run_epochs(
    mh_class: Type[BaseMH],
    inst: dict,
    monitor_cfg: StagnationConfig,
    num_particulas: int = 20,
    num_iteraciones: int = 100,
    epochs: int = 10,
    verbose: bool = False,
    k: float = 5.0,
    center: float = 0.5,
) -> List[dict]:
    """
    Ejecuta múltiples corridas (epochs) y retorna lista de resultados.
    Cada epoch usa semilla = epoch_index + 1.
    """
    resultados = []
    for ep in range(epochs):
        semilla = ep + 1
        t0 = time.perf_counter()
        res = run_experiment(
            mh_class, inst, monitor_cfg,
            num_particulas=num_particulas,
            num_iteraciones=num_iteraciones,
            semilla=semilla,
            verbose=verbose,
            k=k,
            center=center,
        )
        t1 = time.perf_counter()
        res["tiempo"] = t1 - t0
        res["epoch"] = ep + 1
        res["semilla"] = semilla
        resultados.append(res)
    return resultados
