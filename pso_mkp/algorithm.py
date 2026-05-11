"""
pso_mkp/algorithm.py
--------------------
Motor principal del Particle Swarm Optimization (PSO) para el MKP.

Implementa el bucle de optimización: inicialización del enjambre,
actualización de velocidades, binarización LB2, adaptación de parámetros G 
vía DTW, y soporte para 8 variantes de rescate de estancamiento.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from mkp_core.problem   import MKPInstance
from mkp_core.repair    import reparar_solucion
from dtw_stagnation     import StagnationConfig, StagnationMonitor
from lb2 import (
    binarizar_posicion,
    interpolar_G,
    adaptar_G_por_dtw,
)
from pso_mkp.rescues import (
    inyectar_heuristicos,
    mutacion_lp,
    ruin_and_recreate_enjambre,
)


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
    # Stagnation
    use_stagnation : bool = True
    stag_cfg       : StagnationConfig | None = None
    stag_strategy  : str  = "adapt_g"
    stag_max_fires : int  = 4


@dataclass
class PSOEpochResult:
    """Resultado de una epoch del PSO."""
    epoch_idx        : int
    mejor_valor      : float
    iteraciones      : int
    stagnation_fires : int
    historial        : list[float] = field(default_factory=list)


@dataclass
class PSOResult:
    """Resultado completo del PSO (todas las epochs)."""
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


# ── Inicialización ────────────────────────────────────────────────────────────

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


def _obtener_mejor_global(particulas: list[dict]) -> tuple[list[int], float]:
    """Recalcula el mejor global iterando por las partículas."""
    mejor_val = -float('inf')
    mejor_sol = None
    for p in particulas:
        if p['valor'] > mejor_val:
            mejor_val = p['valor']
            mejor_sol = p['solucion'].copy()
    return mejor_sol, mejor_val


# ── Una epoch del PSO ─────────────────────────────────────────────────────────

def ejecutar_epoch(
    inst      : MKPInstance,
    params    : PSOParams,
    epoch_idx : int = 0,
    verbose   : bool = True,
) -> PSOEpochResult:
    """Ejecuta una epoch completa del PSO y devuelve el resultado."""

    n = inst.n

    # Inicializar enjambre
    particulas, mejor_solucion_global, mejor_valor_global = _inicializar_enjambre(
        inst, params.pop_size
    )

    historial  = []
    stag_fires = 0

    # Estado dinámico de los parámetros G
    G1 = params.G1_i
    G2 = params.G2_i
    G3 = params.G3_i
    v2_exploring = True   # Para V2 cyclic
    tabu_genes   = []     # Para V7

    # Inicializar monitor
    monitor: StagnationMonitor | None = None
    if params.use_stagnation and params.stag_cfg:
        monitor = StagnationMonitor(cfg=params.stag_cfg)

    # Variable para guardar el estado LB2 (solo adapt_g)
    estado_lb2 = ""

    if verbose:
        print(f"Comienzo Epoch {epoch_idx + 1}")

    for it in range(params.iterations):

        # Actualizar cada partícula
        for particula in particulas:
            
            # Actualizar velocidad
            velocidad = (np.array(particula['solucion']) * params.inercia +
                         np.array(particula['mejor_solucion_personal']) * params.coeficiente_cognitivo * random.random() +
                         np.array(mejor_solucion_global) * params.coeficiente_social * random.random())
            
            # Limitar velocidad
            nueva_solucion_continua = np.clip(velocidad, -params.v_max, params.v_max)

            # Binarización LB2: convertir posición continua -> solución binaria
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

            # Adaptación continua de G vía DTW (siempre, si ready)
            if status.get("ready") and params.stag_strategy == "adapt_g":
                delta_dtw   = status.get("delta", 0.0)
                theta_delta = status.get("theta_delta", 1.0)
                G1, G2, G3, estado_lb2 = adaptar_G_por_dtw(
                    delta_dtw, theta_delta,
                    params.G1_i, params.G1_f,
                    params.G2_i, params.G2_f,
                    params.G3_i, params.G3_f,
                )

            # Verbose: imprimir métricas DTW por iteración (estilo notebook)
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
                if params.stag_strategy == "adapt_g" and estado_lb2:
                    print(
                        f"iter={it:03d} | Estado: {estado_lb2:<16s}"
                        f"| Delta={dlt:7.1f} | theta_d={td:7.1f} "
                        f"| d1={d1:.1f} d2={d2:.1f} "
                        f"best={mejor_valor_global:.1f}"
                    )
                else:
                    print(
                        f"iter={it:03d} n={ns} "
                        f"D1={d1:.3f} D2={d2:.3f} "
                        f"Delta={dlt:.3f} theta={td} "
                        f"theta_c={tc:.2f} theta_r={tr:.2f} "
                        f"no_improve={ni} fire={fr} "
                        f"best={mejor_valor_global:.1f}"
                    )

            if status.get("fire"):
                max_f = params.stag_max_fires
                if max_f > 0 and stag_fires >= max_f:
                    pass  # Límite alcanzado
                else:
                    stag_fires += 1
                    monitor.reset()

                    s = params.stag_strategy

                    if s == "adapt_g":
                        # El Original: fuerza una exploración temporal fuerte
                        G1 = params.G1_i
                        G2 = params.G2_i
                        G3 = params.G3_i
                        # También reiniciar 30% peores partículas
                        n_reinicio = max(1, params.pop_size // 3)
                        peores = sorted(range(len(particulas)), key=lambda i: particulas[i]['valor'])[:n_reinicio]
                        for idx in peores:
                            sol = [random.randint(0, 1) for _ in range(n)]
                            sol, val = reparar_solucion(sol, inst)
                            particulas[idx]['solucion'] = sol
                            particulas[idx]['valor'] = val
                            particulas[idx]['mejor_solucion_personal'] = sol
                            particulas[idx]['mejor_valor_personal'] = val

                    elif s == "v1_exploit":
                        # Forzar G a explotación máxima
                        G1 = params.G1_f
                        G2 = params.G2_f
                        G3 = params.G3_f

                    elif s == "v2_cycle":
                        if v2_exploring:
                            G1 = params.G1_i
                            G2 = params.G2_i
                            G3 = params.G3_i
                            v2_exploring = False
                        else:
                            G1 = params.G1_f
                            G2 = params.G2_f
                            G3 = params.G3_f
                            v2_exploring = True

                    elif s == "v3_explore":
                        G1 = params.G1_i
                        G2 = params.G2_i
                        G3 = min(0.8, params.G3_i + 0.3)   # Piso muy alto
                        # 50% inmigrantes aleatorios
                        n_inmig = params.pop_size // 2
                        peores = sorted(range(len(particulas)), key=lambda i: particulas[i]['valor'])[:n_inmig]
                        for idx in peores:
                            sol = [random.randint(0, 1) for _ in range(n)]
                            sol, val = reparar_solucion(sol, inst)
                            particulas[idx]['solucion'] = sol
                            particulas[idx]['valor'] = val
                            particulas[idx]['mejor_solucion_personal'] = sol
                            particulas[idx]['mejor_valor_personal'] = val

                    elif s == "v4_nonlinear":
                        # G3 crece exponencialmente con los fires
                        G3 = min(0.9, params.G3_i * math.exp(0.5 * stag_fires))

                    elif s == "v5_heuristic":
                        particulas = inyectar_heuristicos(
                            particulas, inst, 0.50,
                        )

                    elif s == "v6_lp":
                        particulas = mutacion_lp(
                            particulas, inst, None,
                        )

                    elif s == "v7_tabu_lp":
                        particulas = mutacion_lp(
                            particulas, inst, tabu_genes,
                        )
                        tabu_genes.extend(random.sample(range(n), min(5, n)))
                        if len(tabu_genes) > 25:
                            tabu_genes = tabu_genes[-25:]

                    elif s == "v8_ruin_recreate":
                        particulas = ruin_and_recreate_enjambre(
                            particulas, inst, 0.30,
                        )

                    # Recalcular mejor global despues de un rescate
                    mejor_solucion_global, mejor_valor_global = _obtener_mejor_global(particulas)

                    # Restaurar G a default para estrategias que no son cíclicas
                    if s not in ("v2_cycle", "v4_nonlinear", "adapt_g"):
                        G1 = interpolar_G(it, params.iterations, params.G1_i, params.G1_f)
                        G2 = interpolar_G(it, params.iterations, params.G2_i, params.G2_f)
                        G3 = interpolar_G(it, params.iterations, params.G3_i, params.G3_f)
        else:
            # Sin monitor: transición lineal ciega de G
            G1 = interpolar_G(it, params.iterations, params.G1_i, params.G1_f)
            G2 = interpolar_G(it, params.iterations, params.G2_i, params.G2_f)
            G3 = interpolar_G(it, params.iterations, params.G3_i, params.G3_f)

    # Fin de la epoch
    if verbose:
        print(f"Fin Epoch {epoch_idx + 1}")

    return PSOEpochResult(
        epoch_idx        = epoch_idx,
        mejor_valor      = mejor_valor_global,
        iteraciones      = params.iterations,
        stagnation_fires = stag_fires,
        historial        = historial,
    )


# ── Ejecución completa (todas las epochs) ────────────────────────────────────

def ejecutar_pso(
    inst    : MKPInstance,
    params  : PSOParams,
    verbose : bool = True,
) -> PSOResult:
    """Ejecuta el PSO completo (todas las epochs) y retorna el PSOResult."""
    epochs_result    = []
    mejor_val_global = -float("inf")
    mejor_sol_global: list[int] = []

    for e in range(params.epochs):
        epoch_res = ejecutar_epoch(inst, params, epoch_idx=e, verbose=verbose)
        epochs_result.append(epoch_res)

        if epoch_res.mejor_valor > mejor_val_global:
            mejor_val_global = epoch_res.mejor_valor
            mejor_sol_global = []   # No guardamos sol entera por consistencia

    resultado = PSOResult(
        epochs             = epochs_result,
        mejor_valor_global = mejor_val_global,
        mejor_sol_global   = mejor_sol_global,
        valor_optimo       = inst.valor_optimo,
    )

    if verbose:
        print(resultado.valores_por_epoch)

    return resultado
