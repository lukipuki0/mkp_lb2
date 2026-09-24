"""
Runner: orquesta MH + DTW con adaptación continua (B3 — D2 direct).
intensity = 1 - clip(D2 / (theta_c * scale), 0, 1)
"""

import time
from typing import Dict, List, Type

import numpy as np

from mkp_common.base import BaseMH
from mkp_common.monitor import StagnationConfig, StagnationMonitor


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clip value to [lo, hi]."""
    return max(lo, min(hi, value))


def run_experiment(
    mh_class: Type[BaseMH],
    inst: dict,
    monitor_cfg: StagnationConfig,
    num_particulas: int = 20,
    num_iteraciones: int = 100,
    semilla: int = 42,
    verbose: bool = False,
    scale: float = 2.0,
) -> dict:
    """
    Ejecuta UNA corrida de la MH con DTW B3 D2-direct.

    Args:
        mh_class:        Clase de la MH
        inst:            Instancia MKP cargada
        monitor_cfg:     Configuración del DTW
        num_particulas:  Tamaño de la población
        num_iteraciones: Presupuesto de iteraciones
        semilla:         Seed para reproducibilidad
        verbose:         Imprimir diagnóstico del DTW
        scale:           Factor de escala sobre theta_c (default 2.0)

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
            # B3: intensity = 1 - clip(D2 / (theta_c * scale), 0, 1)
            # D2 bajo (curva plana, estancado) → intensity alto → explore
            # D2 alto (curva activa) → intensity bajo → exploit
            ratio = out["D2_vs_const"] / (out["theta_c"] * scale + eps)
            intensity = 1.0 - _clip(ratio, 0.0, 1.0)
            mh.adapt_continuous(intensity)

            historial_intensity.append(float(intensity))

            if verbose:
                print(
                    f"  [B3] iter={it:03d} | "
                    f"D2={out['D2_vs_const']:.1f} | "
                    f"th_c={out['theta_c']:.1f} | "
                    f"ratio={ratio:.3f} | "
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
    scale: float = 2.0,
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
            scale=scale,
        )
        t1 = time.perf_counter()
        res["tiempo"] = t1 - t0
        res["epoch"] = ep + 1
        res["semilla"] = semilla
        resultados.append(res)
    return resultados
