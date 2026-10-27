"""
Runner genérico: orquesta MH + DTW + loop de iteraciones.
Acepta una función de decisión (fire_fn) para desacoplar
la estrategia de adaptación del loop de ejecución.

Cada estrategia (fire_binario, fire_d2, etc.) pasa su propia fire_fn.
"""

import time
from typing import Callable, Dict, List, Optional, Type

import numpy as np

from .base import BaseMH
from .monitor import StagnationConfig, StagnationMonitor


# Tipo para la función de decisión: recibe el dict de salida del monitor,
# retorna True (explorar) o False (explotar).
FireFn = Callable[[Dict], bool]
FireFnFactory = Callable[[], FireFn]


def _default_fire_fn(out: Dict) -> bool:
    """Estrategia A4 (baseline): usa el fire del monitor (3 condiciones + patience)."""
    return out["fire"]


def run_experiment(
    mh_class: Type[BaseMH],
    inst: dict,
    monitor_cfg: StagnationConfig,
    num_particulas: int = 20,
    num_iteraciones: int = 100,
    semilla: int = 42,
    verbose: bool = False,
    fire_fn: Optional[FireFn] = None,
    initial_mode: Optional[str] = None,
    decision_on_early: bool = False,
) -> dict:
    """
    Ejecuta UNA corrida de la MH con DTW auto-adaptativo.

    Args:
        mh_class:        Clase de la MH (ej: BinaryPSO)
        inst:            Instancia MKP cargada
        monitor_cfg:     Configuración del DTW
        num_particulas:  Tamaño de la población
        num_iteraciones: Presupuesto de iteraciones
        semilla:         Seed para reproducibilidad
        verbose:         Imprimir diagnóstico del DTW
        fire_fn:         Función de decisión. Recibe out (dict del monitor),
                         retorna bool. Default: out["fire"] (baseline A4).
        initial_mode:    Modo inicial opcional ("exploit" o "explore").
                         None preserva el modo por defecto de cada MH.
        decision_on_early: Invocar fire_fn aunque el monitor todavía no
                           tenga una ventana completa.

    Returns:
        dict con resultados de la corrida. ``historial_modos`` is aligned with
        ``historial_fitness`` and records the mode used by each ``step()``.
    """
    if fire_fn is None:
        fire_fn = _default_fire_fn

    rng = np.random.default_rng(semilla)
    mh = mh_class(inst, rng, num_particulas=num_particulas)
    monitor = StagnationMonitor(cfg=monitor_cfg)

    mh.initialize()

    if initial_mode is not None:
        if initial_mode not in {"exploit", "explore"}:
            raise ValueError("initial_mode must be 'exploit' or 'explore'")
        mh.adapt(initial_mode == "explore")

    historial_fitness: List[float] = []
    historial_dtw: List[Dict] = []
    historial_modos: List[str] = []
    fire_count = 0

    for it in range(num_iteraciones):
        # The fitness below is produced with this mode; adaptation happens
        # only after the monitor observes that fitness.
        mode_used = mh.mode
        fitness = mh.step()

        out = monitor.update(fitness)
        historial_dtw.append(out)

        mode_before_adaptation = mh.mode
        if decision_on_early or out.get("ready"):
            fire = fire_fn(out)
            mh.adapt(fire)

        if mh.mode == "explore" and mode_before_adaptation != "explore":
            fire_count += 1
            if verbose:
                print(
                    f"  [DTW FIRE] iter={it:03d} | "
                    f"no_imp={out['no_improve_len']} | "
                    f"D1={out.get('D1_vs_ramp', 0):.1f} "
                    f"D2={out.get('D2_vs_const', 0):.1f} "
                    f"delta={out.get('delta', 0):+.1f} >> EXPLORE"
                )
        elif mh.mode == "exploit" and mode_before_adaptation == "explore":
            if verbose:
                print(
                    f"  [DTW COOL] iter={it:03d} | "
                    f"no_imp={out['no_improve_len']} >> EXPLOIT"
                )

        historial_modos.append(mode_used)
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
        "fire_count": fire_count,
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
    fire_fn: Optional[FireFn] = None,
    fire_fn_factory: Optional[FireFnFactory] = None,
    initial_mode: Optional[str] = None,
    decision_on_early: bool = False,
) -> List[dict]:
    """
    Ejecuta múltiples corridas (epochs) y retorna lista de resultados.
    Cada epoch usa semilla = epoch_index + 1.
    """
    resultados = []
    for ep in range(epochs):
        semilla = ep + 1
        t0 = time.perf_counter()
        epoch_fire_fn = fire_fn_factory() if fire_fn_factory is not None else fire_fn
        res = run_experiment(
            mh_class, inst, monitor_cfg,
            num_particulas=num_particulas,
            num_iteraciones=num_iteraciones,
            semilla=semilla,
            verbose=verbose,
            fire_fn=epoch_fire_fn,
            initial_mode=initial_mode,
            decision_on_early=decision_on_early,
        )
        t1 = time.perf_counter()
        res["tiempo"] = t1 - t0
        res["epoch"] = ep + 1
        res["semilla"] = semilla
        resultados.append(res)
    return resultados
