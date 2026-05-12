"""
mh/pso.py
---------
Particle Swarm Optimization (PSO) para el MKP con binarización LB2.

Versión limpia para el pipeline híbrido: solo usa estrategia "abort"
cuando el monitor DTW detecta estancamiento.
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
class PSOParams:
    """Hiperparámetros del PSO."""
    pop_size              : int   = 20
    iterations            : int   = 100
    epochs                : int   = 10
    v_max                 : float = 6.0
    inercia               : float = 0.65
    coeficiente_cognitivo : float = 2.0
    coeficiente_social    : float = 2.0
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
class PSOEpochResult:
    """Resultado de un epoch del PSO."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)
    mejor_solucion   : list[int]  = field(default_factory=list)


@dataclass
class PSOResult:
    """Resultado completo del PSO (todos los epochs)."""
    epochs             : list[PSOEpochResult]
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

def _inicializar_enjambre(inst: MKPInstance, pop_size: int) -> tuple[list[dict], list[int], float]:
    """Genera el enjambre inicial y retorna las partículas y el mejor global."""
    n = inst.n
    particulas = []
    mejor_solucion_global = None
    mejor_valor_global = -float('inf')

    for _ in range(pop_size):
        sol = [random.randint(0, 1) for _ in range(n)]
        sol, val = reparar_solucion(sol, inst)
        particula = {
            'solucion': sol,
            'valor': val,
            'mejor_valor_personal': val,
            'mejor_solucion_personal': sol.copy()
        }
        particulas.append(particula)
        if val > mejor_valor_global:
            mejor_valor_global = val
            mejor_solucion_global = sol.copy()

    return particulas, mejor_solucion_global, mejor_valor_global


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


def _crear_particula(sol: list[int], val: float) -> dict:
    """Crea un dict de partícula a partir de una solución ya reparada."""
    return {
        'solucion': sol,
        'valor': val,
        'mejor_valor_personal': val,
        'mejor_solucion_personal': sol.copy(),
    }


# ── Epoch individual ─────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst      : MKPInstance,
    params    : PSOParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
    sol_inyectada: list[int] | None = None,
) -> PSOEpochResult:
    """Ejecuta un epoch completo del PSO con detección de estancamiento (abort)."""

    n = inst.n

    # Inicializar enjambre
    particulas, mejor_solucion_global, mejor_valor_global = _inicializar_enjambre(
        inst, params.pop_size
    )

    # Inyectar solución del orquestador según el modo de inyección
    if sol_inyectada is not None:
        sol_rep = list(sol_inyectada)
        sol_rep, val_rep = reparar_solucion(sol_rep, inst)
        mode = params.injection_mode

        if mode == "random":
            peor_idx = min(range(len(particulas)), key=lambda i: particulas[i]['valor'])
            particulas[peor_idx] = _crear_particula(sol_rep, val_rep)

        elif mode == "mutated":
            particulas[0] = _crear_particula(sol_rep, val_rep)
            for i in range(1, params.pop_size):
                msol, mval = _mutar_solucion(sol_rep, inst)
                particulas[i] = _crear_particula(msol, mval)

        elif mode == "mixed":
            particulas[0] = _crear_particula(sol_rep, val_rep)
            n_mutados = params.pop_size // 2
            for i in range(1, n_mutados):
                msol, mval = _mutar_solucion(sol_rep, inst)
                particulas[i] = _crear_particula(msol, mval)

        # Recalcular mejor global después de la inyección
        for p in particulas:
            if p['valor'] > mejor_valor_global:
                mejor_valor_global = p['valor']
                mejor_solucion_global = p['solucion'].copy()

    historial  = []
    stag_fires = 0

    # Estado dinámico de los parámetros G (transición lineal)
    G1 = params.G1_i
    G2 = params.G2_i
    G3 = params.G3_i

    # Inicializar monitor
    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    for it in range(params.iterations):

        # Actualizar cada partícula
        for particula in particulas:

            # Actualizar velocidad
            velocidad = (np.array(particula['solucion']) * params.inercia +
                         np.array(particula['mejor_solucion_personal']) * params.coeficiente_cognitivo * random.random() +
                         np.array(mejor_solucion_global) * params.coeficiente_social * random.random())

            # Limitar velocidad
            nueva_solucion_continua = np.clip(velocidad, -params.v_max, params.v_max)

            # Binarización LB2
            nueva_sol, nueva_val = binarizar_posicion(
                nueva_solucion_continua, particula['solucion'], inst,
                G1, G2, G3, params.v_max,
            )

            # Actualizar posición actual si mejora
            if nueva_val > particula['valor']:
                particula['solucion'] = nueva_sol
                particula['valor'] = nueva_val

            # Actualizar la mejor solución personal
            if particula['valor'] > particula['mejor_valor_personal']:
                particula['mejor_valor_personal'] = particula['valor']
                particula['mejor_solucion_personal'] = particula['solucion'].copy()

            # Actualizar el mejor global
            if particula['valor'] > mejor_valor_global:
                mejor_valor_global = particula['valor']
                mejor_solucion_global = particula['solucion'].copy()

        historial.append(mejor_valor_global)

        # ── Stagnation check ──────────────────────────────────────────────
        if monitor is not None:
            status = monitor.update(mejor_valor_global)

            if verbose and status.get("ready"):
                d1  = status.get("D1_vs_ramp", 0.0)
                d2  = status.get("D2_vs_const", 0.0)
                dlt = status.get("delta", 0.0)
                td  = status.get("theta_delta", 0.0)
                tc  = status.get("theta_c", 0.0)
                tr  = status.get("theta_r", 0.0)
                ni  = status.get("no_improve_len", 0)
                fr  = status.get("fire", False)
                ns  = status.get("n", 0)
                print(
                    f"iter={it:03d}",
                    f"n={ns}",
                    f"D1={d1:.3f}",
                    f"D2={d2:.3f}",
                    f"Delta={dlt:.3f}",
                    f"theta={td}",
                    f"theta_c={tc:.2f}",
                    f"theta_r={tr:.2f}",
                    f"no_improve={ni}",
                    f"fire={fr}",
                    f"best={mejor_valor_global:.1f}",
                )

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

    return PSOEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_valor_global,
        iteraciones      = params.iterations,
        stagnation_fires = stag_fires,
        historial        = historial,
        mejor_solucion   = list(mejor_solucion_global) if mejor_solucion_global is not None else [],
    )


# ── Ejecución multi-epoch ────────────────────────────────────────────────────

def ejecutar_pso(
    inst: MKPInstance,
    params: PSOParams,
    verbose: bool = True,
) -> PSOResult:
    """Ejecuta el PSO completo (todos los epochs) y retorna el PSOResult."""
    epochs_result    = []
    mejor_val_global = -float("inf")
    mejor_sol_global: list[int] = []

    for e in range(params.epochs):
        epoch_res = ejecutar_epoch(inst, params, epoch_idx=e, verbose=verbose)
        epochs_result.append(epoch_res)

        if epoch_res.mejor_valor > mejor_val_global:
            mejor_val_global = epoch_res.mejor_valor
            mejor_sol_global = epoch_res.mejor_solucion.copy()

    return PSOResult(
        epochs             = epochs_result,
        mejor_valor_global = mejor_val_global,
        mejor_sol_global   = mejor_sol_global,
        valor_optimo       = inst.valor_optimo,
    )

