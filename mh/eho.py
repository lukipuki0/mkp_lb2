"""
mh/eho.py
---------
Elk Herd Optimizer (EHO) para el MKP con binarización LB2.

Adaptado para el pipeline híbrido: incluye detección de estancamiento
por DTW y soporte de inyección de soluciones.
Referencia original: https://doi.org/10.1007/s10462-023-10680-4
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from mkp_core.problem   import MKPInstance
from mkp_core.repair    import reparar_solucion
from dtw_stagnation     import StagnationConfig, StagnationMonitor
from lb2 import binarizar_posicion, interpolar_G


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass
class EHOParams:
    """Hiperparámetros del EHO."""
    pop_size    : int   = 30
    iterations  : int   = 300
    epochs      : int   = 10
    v_max       : float = 6.0
    bull_ratio  : float = 0.2
    # LB2 params
    G1_i : float = 0.5;  G1_f : float = 1.0
    G2_i : float = 0.5;  G2_f : float = 7.2
    G3_i : float = 0.5;  G3_f : float = 0.0
    # Inyección de solución (pipeline híbrido)
    injection_mode : str  = "random"    # "random" | "mutated" | "mixed"
    # Stagnation
    use_stagnation : bool = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class EHOEpochResult:
    """Resultado de un epoch del EHO."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)  # fitness del mejor elk esta iteración
    mejor_solucion   : list[int]   = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)


@dataclass
class EHOResult:
    """Resultado completo del EHO (todos los epochs)."""
    epochs             : list[EHOEpochResult]
    mejor_valor_global : float
    mejor_sol_global   : list[int]
    valor_optimo       : float

    @property
    def gap_pct(self) -> float | None:
        if self.valor_optimo == 0:
            return None
        return 100.0 * (self.valor_optimo - self.mejor_valor_global) / self.valor_optimo

    @property
    def valores_por_epoch(self) -> list[float]:
        return [ep.mejor_valor for ep in self.epochs]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inicializar_poblacion(
    inst: MKPInstance,
    pop_size: int,
    v_max: float,
) -> tuple[np.ndarray, list[list[int]], list[float]]:
    """Genera la población inicial: posiciones continuas + soluciones binarias factibles."""
    n = inst.n
    posiciones = np.random.uniform(-v_max, v_max, size=(pop_size, n))
    poblacion_bin = []
    fitnesses     = []

    for i in range(pop_size):
        sol = [random.randint(0, 1) for _ in range(n)]
        sol, val = reparar_solucion(sol, inst)
        poblacion_bin.append(sol)
        fitnesses.append(val)

    return posiciones, poblacion_bin, fitnesses


def _mutar_solucion(sol: list[int], inst: MKPInstance, n_flips: int = 0) -> tuple[list[int], float]:
    """Crea una copia de `sol` con bits invertidos al azar y la repara."""
    copia = list(sol)
    n = len(copia)
    if n_flips <= 0:
        n_flips = random.randint(1, 3)
    indices = random.sample(range(n), min(n_flips, n))
    for idx in indices:
        copia[idx] = 1 - copia[idx]
    copia, val = reparar_solucion(copia, inst)
    return copia, val


# ── Epoch individual ─────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst      : MKPInstance,
    params    : EHOParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: list[int] | None = None,
) -> EHOEpochResult:
    """Ejecuta un epoch completo del EHO con detección de estancamiento (abort)."""

    pop_size = params.pop_size
    n = inst.n
    num_bulls = max(1, round(pop_size * params.bull_ratio))

    # Inicializar población
    posiciones, poblacion_bin, fitnesses = _inicializar_poblacion(
        inst, pop_size, params.v_max,
    )

    # Inyectar solución del orquestador según el modo de inyección
    if sol_inyectada is not None:
        sol_rep = list(sol_inyectada)
        sol_rep, val_rep = reparar_solucion(sol_rep, inst)
        mode = params.injection_mode

        if mode == "random":
            peor_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            poblacion_bin[peor_idx] = sol_rep
            fitnesses[peor_idx] = val_rep
            posiciones[peor_idx] = np.zeros(n) # Resetear posición para inyectado

        elif mode == "mutated":
            poblacion_bin[0] = sol_rep
            fitnesses[0] = val_rep
            posiciones[0] = np.zeros(n)
            for i in range(1, pop_size):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion_bin[i] = msol
                fitnesses[i] = mval
                posiciones[i] = np.random.uniform(-params.v_max, params.v_max, size=n)

        elif mode == "mixed":
            poblacion_bin[0] = sol_rep
            fitnesses[0] = val_rep
            posiciones[0] = np.zeros(n)
            n_mutados = pop_size // 2
            for i in range(1, n_mutados):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion_bin[i] = msol
                fitnesses[i] = mval
                posiciones[i] = np.random.uniform(-params.v_max, params.v_max, size=n)

    # Convertir fitnesses a numpy array para facilitar manejo de índices
    fitnesses_np = np.array(fitnesses)
    
    # Encontrar mejor inicial
    mejor_idx = np.argmax(fitnesses_np)
    mejor_val = fitnesses_np[mejor_idx]
    mejor_sol = poblacion_bin[mejor_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0

    # Estado dinámico de los parámetros G
    G1 = params.G1_i
    G2 = params.G2_i
    G3 = params.G3_i

    # Inicializar monitor
    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):
        # Ordenar población por fitness (descendente, ya que MKP es maximización)
        sorted_indexes = np.argsort(fitnesses_np)[::-1]
        sorted_fitness = fitnesses_np[sorted_indexes]
        
        bull_indices = sorted_indexes[:num_bulls]
        
        # Selección Ruleta para las Familias (Rutting Season)
        # Adaptado para maximización: probabilidad proporcional al fitness
        bull_fitness = sorted_fitness[:num_bulls]
        total_fit = np.sum(bull_fitness)
        if total_fit > 0:
            selection_probs = bull_fitness / total_fit
        else:
            selection_probs = np.ones(num_bulls) / num_bulls

        Families = np.zeros(pop_size, dtype=int)
        for i in range(num_bulls, pop_size):
            female_index = sorted_indexes[i]
            selected_bull = np.random.choice(bull_indices, p=selection_probs)
            Families[female_index] = selected_bull

        # Reproducción (Calving Season)
        offspring_pos = []
        offspring_bin = []
        offspring_fit = []

        for i in range(pop_size):
            individual = posiciones[i].copy()
            
            if i in bull_indices:
                # Bull (male)
                h = np.random.randint(0, pop_size)
                alpha = np.random.rand()
                new_pos = individual + alpha * (posiciones[h] - individual)
            else:
                # Harem (female)
                male_index = Families[i]
                h = np.random.randint(0, num_bulls)
                random_bull = bull_indices[h]
                
                gamma_vec = np.random.uniform(-2, 2, size=n)
                beta = 1.0
                new_pos = (individual 
                           + beta * (posiciones[male_index] - individual) 
                           + gamma_vec * (posiciones[random_bull] - individual))
            
            # Clip position to [-v_max, v_max] bounds
            new_pos = np.clip(new_pos, -params.v_max, params.v_max)
            
            # Binarizar la nueva posición usando LB2
            nueva_sol, nueva_val = binarizar_posicion(
                new_pos, poblacion_bin[i], inst,
                G1, G2, G3, params.v_max,
            )
            
            offspring_pos.append(new_pos)
            offspring_bin.append(nueva_sol)
            offspring_fit.append(nueva_val)

        # Merge de las poblaciones (padres + hijos)
        merged_pos = np.concatenate([posiciones, np.array(offspring_pos)])
        merged_bin = poblacion_bin + offspring_bin
        merged_fit = np.concatenate([fitnesses_np, np.array(offspring_fit)])

        # Seleccionar la mejor pop_size (descendente)
        merged_sorted_indices = np.argsort(merged_fit)[::-1]
        selected_indices = merged_sorted_indices[:pop_size]

        # Actualizar la población actual
        posiciones = merged_pos[selected_indices]
        poblacion_bin = [merged_bin[idx] for idx in selected_indices]
        fitnesses_np = merged_fit[selected_indices]

        # El mejor de la iteración actual
        mejor_actual_val = fitnesses_np[0]
        mejor_actual_sol = poblacion_bin[0].copy()

        # Actualizar mejor global
        if mejor_actual_val > mejor_val:
            mejor_val = mejor_actual_val
            mejor_sol = mejor_actual_sol

        historial.append(mejor_val)
        historial_inst.append(mejor_actual_val)

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
                    print(f"    [Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break
        else:
            # Sin monitor: transición lineal ciega de G
            G1 = interpolar_G(it, params.iterations, params.G1_i, params.G1_f)
            G2 = interpolar_G(it, params.iterations, params.G2_i, params.G2_f)
            G3 = interpolar_G(it, params.iterations, params.G3_i, params.G3_f)

    return EHOEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = params.iterations,
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
    )


# ── Ejecución multi-epoch ────────────────────────────────────────────────────

def ejecutar_eho(
    inst: MKPInstance,
    params: EHOParams,
    verbose: bool = True,
) -> EHOResult:
    """Ejecuta el EHO completo (todos los epochs) y retorna el EHOResult."""
    epochs_result    = []
    mejor_val_global = -float("inf")
    mejor_sol_global: list[int] = []

    for e in range(params.epochs):
        epoch_res = ejecutar_epoch(inst, params, epoch_idx=e, verbose=verbose)
        epochs_result.append(epoch_res)

        if epoch_res.mejor_valor > mejor_val_global:
            mejor_val_global = epoch_res.mejor_valor
            mejor_sol_global = epoch_res.mejor_solucion.copy()

    return EHOResult(
        epochs             = epochs_result,
        mejor_valor_global = mejor_val_global,
        mejor_sol_global   = mejor_sol_global,
        valor_optimo       = inst.valor_optimo,
    )
