"""
mh/abc.py
---------
Artificial Bee Colony (ABC) Algorithm para el MKP con binarización LB2.

Lógica de 4 Fases:
  1. Inicialización: Genera población inicial de N fuentes de alimento en [-v_max, v_max].
  2. Fase de Abejas Empleadas: v_{i,j} = x_{i,j} + phi * (x_{i,j} - x_{k,j}) con phi en [-1, 1],
     binarización LB2 + reparación greedy, selección codiciosa (si mejora trials[i]=0, si no trials[i]+=1).
  3. Fase de Abejas Observadoras: P_i = 0.9 * (fit_i / max(fit)) + 0.1, selección por ruleta
     y búsqueda en vecindario con misma modificación + selección codiciosa.
  4. Fase de Abejas Exploradoras (Scouts): Si trials[i] >= limit, reinicialización aleatoria.

Versión limpia para el pipeline híbrido: solo usa estrategia "abort" cuando DTW detecta estancamiento.
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
class ABCParams:
    """Hiperparámetros del Artificial Bee Colony (ABC)."""
    pop_size       : int   = 30     # Número de fuentes de alimento / abejas
    iterations     : int   = 300
    epochs         : int   = 10
    v_max          : float = 6.0
    limit          : int | None = None # Límite de intentos para abejas exploradoras
    # LB2 params
    G1_i : float = 0.5;  G1_f : float = 1.0
    G2_i : float = 0.5;  G2_f : float = 7.2
    G3_i : float = 0.5;  G3_f : float = 0.0
    # Inyección de solución (pipeline híbrido)
    injection_mode : str  = "mixed" # "random" | "mutated" | "mixed"
    # Stagnation
    use_stagnation : bool = True
    stag_cfg       : StagnationConfig | None = None


@dataclass
class ABCEpochResult:
    """Resultado de un epoch del ABC."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    historial_inst   : list[float] = field(default_factory=list)
    mejor_solucion   : list[int]   = field(default_factory=list)
    dtw_deltas       : list[float] = field(default_factory=list)


@dataclass
class ABCResult:
    """Resultado completo del ABC (todos los epochs)."""
    epochs             : list[ABCEpochResult]
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

def _inicializar_fuentes(
    inst: MKPInstance,
    pop_size: int,
    v_max: float,
) -> tuple[np.ndarray, list[list[int]], list[float], np.ndarray]:
    """Genera las fuentes de alimento iniciales (posiciones continuas, soluciones binarias, fitnesses y trials)."""
    n = inst.n
    posiciones = np.random.uniform(-v_max, v_max, size=(pop_size, n))
    poblacion_bin = []
    fitnesses     = []

    for i in range(pop_size):
        sol = [random.randint(0, 1) for _ in range(n)]
        sol, val = reparar_solucion(sol, inst)
        poblacion_bin.append(sol)
        fitnesses.append(val)

    trials = np.zeros(pop_size, dtype=int)
    return posiciones, poblacion_bin, fitnesses, trials


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
    inst          : MKPInstance,
    params        : ABCParams,
    epoch_idx     : int = 0,
    verbose       : bool = True,
    sol_inyectada : list[int] | None = None,
) -> ABCEpochResult:
    """Ejecuta un epoch completo del ABC con las 4 fases especificadas y monitor DTW."""

    n = inst.n
    pop_size = params.pop_size
    limit = params.limit if params.limit is not None else int(pop_size * n / 2)

    # 1. Fase de Inicialización
    posiciones, poblacion_bin, fitnesses, trials = _inicializar_fuentes(
        inst, pop_size, params.v_max
    )

    # Inyección de solución si aplica
    if sol_inyectada is not None:
        sol_rep = list(sol_inyectada)
        sol_rep, val_rep = reparar_solucion(sol_rep, inst)
        mode = params.injection_mode

        if mode == "random":
            peor_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            poblacion_bin[peor_idx] = sol_rep
            fitnesses[peor_idx] = val_rep
            trials[peor_idx] = 0

        elif mode == "mutated":
            poblacion_bin[0] = sol_rep
            fitnesses[0] = val_rep
            trials[0] = 0
            for i in range(1, pop_size):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion_bin[i] = msol
                fitnesses[i] = mval
                trials[i] = 0

        elif mode == "mixed":
            poblacion_bin[0] = sol_rep
            fitnesses[0] = val_rep
            trials[0] = 0
            n_mutados = pop_size // 2
            for i in range(1, n_mutados):
                msol, mval = _mutar_solucion(sol_rep, inst)
                poblacion_bin[i] = msol
                fitnesses[i] = mval
                trials[i] = 0

    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    mejor_val = fitnesses[best_idx]
    mejor_sol = poblacion_bin[best_idx].copy()

    historial      = []
    historial_inst = []
    dtw_deltas     = []
    stag_fires     = 0

    # Estado dinámico de los parámetros G (transición lineal)
    G1 = params.G1_i
    G2 = params.G2_i
    G3 = params.G3_i

    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):

        # 2. Fase de Abejas Empleadas
        for i in range(pop_size):
            # Seleccionar vecina k != i
            k_candidates = [idx for idx in range(pop_size) if idx != i]
            k = random.choice(k_candidates)

            # Seleccionar dimensión j
            j = random.randint(0, n - 1)

            phi = random.uniform(-1.0, 1.0)
            v_i = posiciones[i].copy()
            v_i[j] = v_i[j] + phi * (v_i[j] - posiciones[k][j])
            v_i = np.clip(v_i, -params.v_max, params.v_max)

            # Binarización LB2 + Reparación greedy
            sol_prop, val_prop = binarizar_posicion(
                v_i, poblacion_bin[i], inst,
                G1, G2, G3, params.v_max
            )

            # Selección codiciosa
            if val_prop >= fitnesses[i]:
                posiciones[i] = v_i
                poblacion_bin[i] = sol_prop
                fitnesses[i] = val_prop
                trials[i] = 0
            else:
                trials[i] += 1

        # 3. Fase de Abejas Observadoras
        max_fit = max(fitnesses)
        if max_fit > 0:
            probs = np.array([0.9 * (fit / max_fit) + 0.1 for fit in fitnesses])
        else:
            probs = np.full(pop_size, 1.0 / pop_size)
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum

        for _ in range(pop_size):
            # Seleccionar fuente i mediante ruleta basada en P_i
            i = np.random.choice(pop_size, p=probs)

            k_candidates = [idx for idx in range(pop_size) if idx != i]
            k = random.choice(k_candidates)

            j = random.randint(0, n - 1)
            phi = random.uniform(-1.0, 1.0)
            v_i = posiciones[i].copy()
            v_i[j] = v_i[j] + phi * (v_i[j] - posiciones[k][j])
            v_i = np.clip(v_i, -params.v_max, params.v_max)

            sol_prop, val_prop = binarizar_posicion(
                v_i, poblacion_bin[i], inst,
                G1, G2, G3, params.v_max
            )

            if val_prop >= fitnesses[i]:
                posiciones[i] = v_i
                poblacion_bin[i] = sol_prop
                fitnesses[i] = val_prop
                trials[i] = 0
            else:
                trials[i] += 1

        # Actualizar mejor global de la colonia
        best_idx_iter = max(range(pop_size), key=lambda idx: fitnesses[idx])
        fit_iter_best = fitnesses[best_idx_iter]

        if fit_iter_best > mejor_val:
            mejor_val = fit_iter_best
            mejor_sol = poblacion_bin[best_idx_iter].copy()

        # 4. Fase de Abejas Exploradoras (Scouts)
        for i in range(pop_size):
            if trials[i] >= limit:
                # Reinicializar aleatoriamente la fuente desatendida
                posiciones[i] = np.random.uniform(-params.v_max, params.v_max, size=n)
                random_sol = [random.randint(0, 1) for _ in range(n)]
                sol_rep, val_rep = reparar_solucion(random_sol, inst)
                poblacion_bin[i] = sol_rep
                fitnesses[i] = val_rep
                trials[i] = 0

                if val_rep > mejor_val:
                    mejor_val = val_rep
                    mejor_sol = sol_rep.copy()

        historial.append(mejor_val)
        historial_inst.append(fit_iter_best)

        if verbose:
            print(f"  [ABC MKP] Iter {it+1:3d}/{params.iterations} | Mejor: {mejor_val:10.1f} | IterBest: {fit_iter_best:10.1f}")

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
                print(f"i={it:03d} | Estado: {estado:<15} | Delta={dlt:6.1f} | Th_d={td:6.1f} | best={mejor_val:.1f}")

            if status.get("fire"):
                stag_fires += 1
                if verbose:
                    print(f"    [ABC Stagnation] Fire #{stag_fires} @ iter {it} -> ABORT")
                break
        else:
            G1 = interpolar_G(it, params.iterations, params.G1_i, params.G1_f)
            G2 = interpolar_G(it, params.iterations, params.G2_i, params.G2_f)
            G3 = interpolar_G(it, params.iterations, params.G3_i, params.G3_f)

    return ABCEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_val,
        iteraciones      = len(historial),
        stagnation_fires = stag_fires,
        historial        = historial,
        historial_inst   = historial_inst,
        mejor_solucion   = mejor_sol,
        dtw_deltas       = dtw_deltas,
    )


# ── Ejecución multi-epoch ────────────────────────────────────────────────────

def ejecutar_abc(
    inst: MKPInstance,
    params: ABCParams,
    verbose: bool = True,
) -> ABCResult:
    """Ejecuta el ABC completo (todos los epochs) y retorna el ABCResult."""
    epochs_result    = []
    mejor_val_global = -float("inf")
    mejor_sol_global: list[int] = []

    for e in range(params.epochs):
        epoch_res = ejecutar_epoch(inst, params, epoch_idx=e, verbose=verbose)
        epochs_result.append(epoch_res)

        if epoch_res.mejor_valor > mejor_val_global:
            mejor_val_global = epoch_res.mejor_valor
            mejor_sol_global = epoch_res.mejor_solucion.copy()

    return ABCResult(
        epochs             = epochs_result,
        mejor_valor_global = mejor_val_global,
        mejor_sol_global   = mejor_sol_global,
        valor_optimo       = inst.valor_optimo,
    )
