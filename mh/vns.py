"""
mh/vns.py
---------
Variable Neighborhood Search (VNS) para el MKP.

Versión limpia para el pipeline híbrido: solo usa estrategia "abort"
cuando el monitor DTW detecta estancamiento.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mkp_core.problem       import MKPInstance
from mkp_core.repair        import reparar_solucion
from dtw_stagnation         import StagnationConfig, StagnationMonitor
from mh.sa_neighborhood     import get_operator


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class VNSParams:
    """Hiperparámetros de VNS."""
    epochs:          int = 10
    iterations:      int = 2000
    k_max:           int = 5
    ls_max_iters:    int = 50
    neighborhood_op: str = "flip_bits"
    # Stagnation
    use_stagnation:  bool = True
    stag_cfg:        StagnationConfig | None = None


@dataclass
class VNSEpochResult:
    """Resultado de un epoch de VNS."""
    epoch_idx:         int
    mejor_valor:       float
    iteraciones:       int
    stagnation_fires:  int
    historial:         list[float] = field(default_factory=list)
    historial_inst:    list[float] = field(default_factory=list)  # fitness de sol_actual al final de cada iteración
    mejor_solucion:    list[int] = field(default_factory=list)
    dtw_deltas:        list[float] = field(default_factory=list)


@dataclass
class VNSResult:
    """Resultado completo de VNS (todos los epochs)."""
    epochs:             list[VNSEpochResult]
    mejor_valor_global: float
    mejor_sol_global:   list[int]
    valor_optimo:       float

    @property
    def gap_pct(self) -> float | None:
        if self.valor_optimo == 0:
            return None
        return 100.0 * (self.valor_optimo - self.mejor_valor_global) / self.valor_optimo

    @property
    def valores_por_epoch(self) -> list[float]:
        return [ep.mejor_valor for ep in self.epochs]


# ── Búsqueda Local (First-Improvement Hill Climbing) ──────────────────────────

def ejecutar_busqueda_local(
    sol: list[int],
    inst: MKPInstance,
    max_iters: int = 50,
) -> tuple[list[int], float]:
    """Búsqueda local First-Improvement usando flips de 1 bit."""
    sol_curr = list(sol)
    sol_curr, val_curr = reparar_solucion(sol_curr, inst)
    
    for _ in range(max_iters):
        indices = list(range(inst.n))
        random.shuffle(indices)
        mejoro = False
        for idx in indices:
            vecino = list(sol_curr)
            vecino[idx] = 1 - vecino[idx]
            vecino, val_vecino = reparar_solucion(vecino, inst)
            if val_vecino > val_curr:
                sol_curr = vecino
                val_curr = val_vecino
                mejoro = True
                break
        if not mejoro:
            break
            
    return sol_curr, val_curr


# ── Epoch individual ──────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst: MKPInstance,
    params: VNSParams,
    epoch_idx: int = 0,
    verbose: bool = True,
    sol_inicial: list[int] | None = None,
) -> VNSEpochResult:
    """Ejecuta un epoch completo de VNS con detección de estancamiento (abort)."""
    
    # Solución inicial: semilla del orquestador o aleatoria
    if sol_inicial is not None:
        sol_actual = list(sol_inicial)
        sol_actual, val_actual = reparar_solucion(sol_actual, inst)
    else:
        sol_actual = [random.randint(0, 1) for _ in range(inst.n)]
        sol_actual, val_actual = reparar_solucion(sol_actual, inst)
        
    # Búsqueda local inicial para empezar desde un óptimo local
    sol_actual, val_actual = ejecutar_busqueda_local(sol_actual, inst, params.ls_max_iters)
    
    mejor_sol = list(sol_actual)
    mejor_val = val_actual
    
    historial = []
    historial_inst = []
    dtw_deltas = []
    stag_fires = 0
    
    fn_shaking = get_operator(params.neighborhood_op)
    
    monitor = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)
        
    k = 1
    it = 0
    for it in range(params.iterations):
        # 1. Shaking (Perturbación con tamaño k)
        vecino, _ = fn_shaking(sol_actual, inst, k)
        
        # 2. Búsqueda local desde el vecino sacudido
        sol_candidato, val_candidato = ejecutar_busqueda_local(vecino, inst, params.ls_max_iters)
        
        # 3. Cambio de vecindario (Criterio de aceptación estándar de VNS)
        if val_candidato > val_actual:
            sol_actual = list(sol_candidato)
            val_actual = val_candidato
            k = 1  # Se encontró mejor solución, reiniciar tamaño de vecindad
        else:
            k += 1  # No hubo mejora, aumentar tamaño de vecindad para explorar más
            if k > params.k_max:
                k = 1  # Si superamos k_max, volvemos a la vecindad más pequeña
                
        # Actualizar el mejor global del epoch
        if val_actual > mejor_val:
            mejor_val = val_actual
            mejor_sol = list(sol_actual)
            
        historial.append(mejor_val)
        historial_inst.append(val_actual)
        
        # ── Stagnation check ──────────────────────────────────────────────
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
                print(f"i={it:03d} | Estado: {estado:<15} | Delta={dlt:6.1f} | Th_d={td:6.1f} | d1={status.get('D1_vs_ramp', 0.0):.3f} | d2={status.get('D2_vs_const', 0.0):.3f} | best={mejor_val:.1f}")

            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [Stagnation] Fire #{stag_fires} @ iter {it + 1} -> ABORT")
                break
                
    return VNSEpochResult(
        epoch_idx=epoch_idx,
        mejor_valor=mejor_val,
        iteraciones=it + 1,
        stagnation_fires=stag_fires,
        historial=historial,
        historial_inst=historial_inst,
        mejor_solucion=mejor_sol,
        dtw_deltas=dtw_deltas,
    )


# ── Ejecución multi-epoch ────────────────────────────────────────────────────

def ejecutar_vns(
    inst: MKPInstance,
    params: VNSParams,
    verbose: bool = True,
) -> VNSResult:
    """Ejecuta VNS completo (todos los epochs) y retorna el VNSResult."""
    epochs_res = []
    mejor_g_v = -float("inf")
    mejor_g_s: list[int] = []

    for e in range(params.epochs):
        res = ejecutar_epoch(inst, params, e, verbose)
        epochs_res.append(res)
        if res.mejor_valor > mejor_g_v:
            mejor_g_v = res.mejor_valor
            mejor_g_s = res.mejor_solucion.copy()

    return VNSResult(
        epochs=epochs_res,
        mejor_valor_global=mejor_g_v,
        mejor_sol_global=mejor_g_s,
        valor_optimo=inst.valor_optimo,
    )
