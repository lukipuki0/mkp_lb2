"""
mh/sa.py
--------
Simulated Annealing (SA) para el MKP.

Versión limpia para el pipeline híbrido: solo usa estrategia "abort"
cuando el monitor DTW detecta estancamiento.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from mh.sa_neighborhood import get_operator
from mkp_core.problem   import MKPInstance
from mkp_core.repair    import reparar_solucion
from dtw_stagnation     import StagnationConfig, StagnationMonitor


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class SAParams:
    """Hiperparámetros del SA."""
    T_inicial:      float = 5_000.0
    T_final:        float = 1.0
    alpha:          float = 0.97
    iter_por_T:     int   = 50
    num_flip:       int   = 3
    epochs:         int   = 10
    neighborhood_op: str  = "flip_bits"
    # Stagnation
    use_stagnation: bool  = True
    stag_cfg:       StagnationConfig | None = None


@dataclass
class SAEpochResult:
    """Resultado de un epoch del SA."""
    epoch_idx:         int
    mejor_solucion:    list[int]
    mejor_valor:       float
    historial:         list[float]
    iteraciones:       int
    T_final_alcanzada: float
    stagnation_fires:  int = 0
    dtw_deltas:        list[float] = field(default_factory=list)
    historial_inst:    list[float] = field(default_factory=list)  # val_actual al fin de cada T


@dataclass
class SAResult:
    """Resultado completo del SA (todos los epochs)."""
    mejor_solucion_global: list[int]
    mejor_valor_global:    float
    epochs:                list[SAEpochResult]
    valor_optimo:          float

    @property
    def valores_por_epoch(self) -> list[float]:
        return [e.mejor_valor for e in self.epochs]

    @property
    def gap_pct(self) -> float | None:
        if self.valor_optimo == 0:
            return None
        return (self.valor_optimo - self.mejor_valor_global) / self.valor_optimo * 100


# ── Epoch individual ──────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst: MKPInstance,
    params: SAParams,
    epoch_idx: int = 0,
    verbose: bool = True,
    sol_inicial: list[int] | None = None,
) -> SAEpochResult:
    """Ejecuta un epoch completo de SA con detección de estancamiento (abort)."""

    # Solución inicial: semilla del orquestador o aleatoria
    if sol_inicial is not None:
        sol_actual = list(sol_inicial)
        sol_actual, val_actual = reparar_solucion(sol_actual, inst)
    else:
        sol_actual = [random.randint(0, 1) for _ in range(inst.n)]
        sol_actual, val_actual = reparar_solucion(sol_actual, inst)

    mejor_sol = sol_actual.copy()
    mejor_val = val_actual

    T = params.T_inicial
    historial:      list[float] = []
    historial_inst: list[float] = []
    dtw_deltas:     list[float] = []
    iteraciones = 0
    stag_fires  = 0

    fn_vecindad = get_operator(params.neighborhood_op)

    # Inicializar monitor
    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    status = {}  # estado inicial del monitor (evita NameError si monitor es None)

    while T > params.T_final:
        # ── Ciclo de evaluaciones a temperatura T ─────────────────────────
        for _ in range(params.iter_por_T):
            vecino, val_vecino = fn_vecindad(sol_actual, inst, params.num_flip)

            delta_sa = val_vecino - val_actual

            # Criterio de aceptación Metropolis
            if delta_sa > 0 or random.random() < math.exp(delta_sa / T):
                sol_actual = vecino
                val_actual = val_vecino

            # Actualizar mejor del epoch (solo tracking interno, sin reportar al DTW)
            if val_actual > mejor_val:
                mejor_val = val_actual
                mejor_sol = sol_actual.copy()

        # ── Al terminar el nivel de temperatura: 1 tick del historial y DTW ──
        # 1 iteración SA = 1 nivel de temperatura = iter_por_T evaluaciones reales
        historial.append(mejor_val)
        historial_inst.append(val_actual)  # val_actual = última solución aceptada (puede ser peor)
        iteraciones += 1

        if monitor is not None:
            status = monitor.update(mejor_val)
            if status.get("ready"):
                dtw_deltas.append(status.get("delta", 0.0))

            if verbose and status.get("ready"):
                dlt = status.get("delta", 0.0)
                td  = status.get("theta_delta", 0.0)
                if dlt > td: estado = "Explorar mucho"
                elif 0 <= dlt <= td: estado = "Explorar poco"
                elif -td <= dlt < 0: estado = "Explotar poco"
                else: estado = "Explotar mucho"
                print(f"i={iteraciones:03d} | Estado: {estado:<15} | Delta={dlt:6.1f} | Th_d={td:6.1f} | d1={status.get('D1_vs_ramp', 0.0):.3f} | d2={status.get('D2_vs_const', 0.0):.3f} | best={mejor_val:.1f}")

            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [Stagnation] Fire #{stag_fires} @ iter {iteraciones} -> ABORT")
                break  # Sale del bucle while

        T *= params.alpha   # Enfriamiento geométrico

    return SAEpochResult(
        epoch_idx=epoch_idx,
        mejor_solucion=mejor_sol,
        mejor_valor=mejor_val,
        historial=historial,
        historial_inst=historial_inst,
        iteraciones=iteraciones,
        T_final_alcanzada=T,
        stagnation_fires=stag_fires,
        dtw_deltas=dtw_deltas,
    )


# ── Ejecución multi-epoch ────────────────────────────────────────────────────

def ejecutar_sa(
    inst: MKPInstance,
    params: SAParams,
    verbose: bool = True,
) -> SAResult:
    """Ejecuta *params.epochs* epochs independientes y devuelve el mejor global."""
    mejor_solucion_global: list[int] | None = None
    mejor_valor_global: float = float("-inf")
    epochs_results: list[SAEpochResult] = []

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
        mejor_solucion_global=mejor_solucion_global or [],
        mejor_valor_global=mejor_valor_global,
        epochs=epochs_results,
        valor_optimo=inst.valor_optimo,
    )

