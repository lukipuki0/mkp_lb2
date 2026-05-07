"""
algorithm.py
────────────
Implementación del algoritmo Simulated Annealing (SA) para el MKP,
con detección de estancamiento basada en DTW (StagnationMonitor).

Contiene:
  - SAParams    : hiperparámetros del SA + configuración del stagnation.
  - EpochResult : resultado de un epoch.
  - SAResult    : resultado de toda la ejecución multi-epoch.
  - ejecutar_epoch() : un epoch completo con soporte de stagnation.
  - ejecutar_sa()    : múltiples epochs independientes → SAResult.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from sa_mkp.neighborhood import flip_bits
from mkp_core.problem import MKPInstance
from mkp_core.repair import reparar_solucion
from dtw_stagnation import StagnationConfig, StagnationMonitor
from sa_mkp.rescues import heuristic_rebuild, lp_guided_rebuild, tabu_lp_rebuild, ruin_and_recreate


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class SAParams:
    """Hiperparámetros del algoritmo SA + configuración de stagnation."""

    # SA
    T_inicial:  float = 5_000.0
    T_final:    float = 1.0
    alpha:      float = 0.97
    iter_por_T: int   = 50
    num_flip:   int   = 3
    epochs:     int   = 10

    # Stagnation monitor
    use_stagnation:   bool  = True
    stag_cfg:         StagnationConfig = field(
        default_factory=StagnationConfig
    )
    # Estrategia al detectar estancamiento: "reheat" | "random"
    stag_strategy:        str   = "reheat"
    stag_reheat_fraction: float = 0.4   # fraccion de T_inicial al recalentar
    stag_max_fires:       int   = 3     # 0 = ilimitado (cuidado: puede ser bucle infinito)


@dataclass
class EpochResult:
    """Resultado de un único epoch SA."""
    epoch_idx:          int
    mejor_solucion:     list[int]
    mejor_valor:        float
    historial:          list[float]   # mejor valor por iteración interna
    iteraciones:        int
    T_final_alcanzada:  float
    stagnation_fires:   int = 0       # veces que se disparó el monitor


@dataclass
class SAResult:
    """Resultado agregado de todos los epochs."""
    mejor_solucion_global: list[int]
    mejor_valor_global:    float
    epochs:                list[EpochResult]
    valor_optimo:          float

    @property
    def valores_por_epoch(self) -> list[float]:
        return [e.mejor_valor for e in self.epochs]

    @property
    def gap_pct(self) -> float | None:
        """Gap relativo respecto al óptimo conocido (%).
        Devuelve None si el óptimo es 0 (desconocido).
        """
        if self.valor_optimo == 0:
            return None
        return (
            (self.valor_optimo - self.mejor_valor_global)
            / self.valor_optimo
            * 100
        )


# ── Epoch individual ──────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst: MKPInstance,
    params: SAParams,
    epoch_idx: int = 0,
    verbose: bool = True,
) -> EpochResult:
    """Ejecuta un epoch completo de SA con detección de estancamiento.

    El StagnationMonitor observa el historial de mejoras dentro del epoch.
    Cuando detecta estancamiento sostenido dispara una de dos estrategias:
      - "reheat"  → recalienta la temperatura a una fracción de T_inicial.
      - "random"  → reinicia la solución actual con una aleatoria reparada.

    Returns
    -------
    EpochResult
    """
    # Solución inicial aleatoria, reparada
    sol_actual = [random.randint(0, 1) for _ in range(inst.n)]
    sol_actual, val_actual = reparar_solucion(sol_actual, inst)

    mejor_sol = sol_actual.copy()
    mejor_val = val_actual

    T = params.T_inicial
    historial: list[float] = []
    iteraciones   = 0
    stag_fires    = 0
    
    # Para V2 y V7
    v2_exploring = False
    tabu_list = []
    
    current_num_flip = params.num_flip

    # Inicializar monitor
    monitor: StagnationMonitor | None = None
    if params.use_stagnation:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    while T > params.T_final:
        for _ in range(params.iter_por_T):
            # ── Vecino ────────────────────────────────────────────────────
            vecino, val_vecino = flip_bits(sol_actual, inst, current_num_flip)

            delta = val_vecino - val_actual

            # Criterio de aceptación Metropolis
            if delta > 0 or random.random() < math.exp(delta / T):
                sol_actual = vecino
                val_actual = val_vecino

            # Actualizar mejor del epoch
            if val_actual > mejor_val:
                mejor_val = val_actual
                mejor_sol = sol_actual.copy()

            historial.append(mejor_val)
            iteraciones += 1

        # ── Stagnation check (una vez por nivel de temperatura) ────────────
        if monitor is not None:
            status = monitor.update(mejor_val)
            if status.get("fire"):
                # Respetar el limite de fires por epoch
                max_fires = params.stag_max_fires
                if max_fires > 0 and stag_fires >= max_fires:
                    pass   # limite alcanzado, ignorar
                else:
                    stag_fires += 1
                    monitor.reset()   # reiniciar para el siguiente ciclo

                    if params.stag_strategy == "reheat":
                        T = params.T_inicial * params.stag_reheat_fraction
                        if verbose:
                            print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> reheat T={T:.1f}")
                            
                    elif params.stag_strategy == "random":
                        sol_actual = [random.randint(0, 1) for _ in range(inst.n)]
                        sol_actual, val_actual = reparar_solucion(sol_actual, inst)
                        if verbose:
                            print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> random restart")
                            
                    elif params.stag_strategy == "v1_exploit":
                        T = params.T_inicial * 0.1  # Baja T
                        current_num_flip = 1        # Búsqueda muy local
                        if verbose:
                            print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> V1 Exploit (T={T:.1f}, flip=1)")
                            
                    elif params.stag_strategy == "v2_cycle":
                        if v2_exploring:
                            T = params.T_inicial * 0.1
                            current_num_flip = 1
                            v2_exploring = False
                            action_name = "Exploit"
                        else:
                            T = params.T_inicial * 0.8
                            current_num_flip = 5
                            v2_exploring = True
                            action_name = "Explore"
                        if verbose:
                            print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> V2 Cycle ({action_name})")
                            
                    elif params.stag_strategy == "v3_explore":
                        T = params.T_inicial * 0.9  # Sube mucho T
                        current_num_flip = min(inst.n // 4, 10)  # Flips grandes
                        if verbose:
                            print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> V3 Explore (T={T:.1f}, flip={current_num_flip})")
                            
                    elif params.stag_strategy == "v4_nonlinear":
                        T = params.T_inicial * math.exp(-0.5 * stag_fires) # Caída exponencial según fires
                        if verbose:
                            print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> V4 Nonlinear (T={T:.1f})")
                            
                    elif params.stag_strategy == "v5_heuristic":
                        sol_actual, val_actual = heuristic_rebuild(inst)
                        if verbose:
                            print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> V5 Heuristic Rebuild")
                            
                    elif params.stag_strategy == "v6_lp":
                        sol_actual, val_actual = lp_guided_rebuild(sol_actual, inst)
                        if verbose:
                            print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> V6 LP Guided")
                            
                    elif params.stag_strategy == "v7_tabu_lp":
                        sol_actual, val_actual = tabu_lp_rebuild(sol_actual, inst, tabu_list)
                        # Agregar a tabú los bits cambiados recientemente (simplificado: tomamos 5 al azar)
                        tabu_list.extend(random.sample(range(inst.n), 5))
                        if len(tabu_list) > 20:
                            tabu_list = tabu_list[-20:] # Mantener tamaño
                        if verbose:
                            print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> V7 Tabu LP")
                            
                    elif params.stag_strategy == "v8_ruin_recreate":
                        sol_actual, val_actual = ruin_and_recreate(sol_actual, inst)
                        if verbose:
                            print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> V8 Ruin & Recreate")

        T *= params.alpha   # Enfriamiento geométrico

    if verbose:
        print(
            f"  [Epoch {epoch_idx + 1}] "
            f"T_final={T:.4f} | mejor={mejor_val:.1f} | "
            f"iters={iteraciones} | stag_fires={stag_fires}"
        )

    return EpochResult(
        epoch_idx=epoch_idx,
        mejor_solucion=mejor_sol,
        mejor_valor=mejor_val,
        historial=historial,
        iteraciones=iteraciones,
        T_final_alcanzada=T,
        stagnation_fires=stag_fires,
    )


# ── Ejecución multi-epoch ─────────────────────────────────────────────────────

def ejecutar_sa(
    inst: MKPInstance,
    params: SAParams,
    verbose: bool = True,
) -> SAResult:
    """Ejecuta *params.epochs* epochs independientes y devuelve el mejor global.

    Parameters
    ----------
    inst : MKPInstance
        Instancia del problema.
    params : SAParams
        Hiperparámetros del SA (incluye configuración de stagnation).
    verbose : bool
        Si True, imprime progreso por epoch.

    Returns
    -------
    SAResult
    """
    mejor_solucion_global: list[int] | None = None
    mejor_valor_global: float = float("-inf")
    epochs_results: list[EpochResult] = []

    for idx in range(params.epochs):
        if verbose:
            print(f"\n== Epoch {idx + 1}/{params.epochs} ====================")

        epoch_result = ejecutar_epoch(
            inst, params, epoch_idx=idx, verbose=verbose
        )
        epochs_results.append(epoch_result)

        if epoch_result.mejor_valor > mejor_valor_global:
            mejor_valor_global    = epoch_result.mejor_valor
            mejor_solucion_global = epoch_result.mejor_solucion.copy()

    return SAResult(
        mejor_solucion_global=mejor_solucion_global,
        mejor_valor_global=mejor_valor_global,
        epochs=epochs_results,
        valor_optimo=inst.valor_optimo,
    )
