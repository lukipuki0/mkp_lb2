"""
hybrid_mkp/orchestrator.py
--------------------------
Orquestador del Pipeline Híbrido de Rotación de Metaheurísticas.

Alterna dinámicamente entre un pool de MH poblacionales (exploración)
y un pool de MH de trayectoria (explotación). El monitor DTW es el
"gatillo" que decide cuándo cambiar de algoritmo. La condición de
parada global es un tiempo máximo configurable.

Flujo:
    Poblacional (estancada por DTW) → Trayectoria → Poblacional → ...
    hasta agotar tiempo_max.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from mkp_core.problem import MKPInstance
from mkp_core.repair  import reparar_solucion
from dtw_stagnation   import StagnationConfig

from mh.sa  import SAParams,  ejecutar_epoch as _sa_epoch
from mh.ts  import TSParams,  ejecutar_epoch as _ts_epoch
from mh.ga  import GAParams,  ejecutar_epoch as _ga_epoch
from mh.pso import PSOParams, ejecutar_epoch as _pso_epoch
from mh.gwo import GWOParams, ejecutar_epoch as _gwo_epoch


# ── Estructuras de datos ──────────────────────────────────────────────────────

POOL_POBLACIONAL = ["GA", "PSO", "GWO"]
POOL_TRAYECTORIA = ["SA", "TS"]

COLORES_MH = {
    "GA" : "#4CAF50",
    "PSO": "#2196F3",
    "GWO": "#9C27B0",
    "SA" : "#FF5722",
    "TS" : "#FF9800",
}


@dataclass
class SwitchLog:
    """Registro de un turno de ejecución de una MH."""
    mh_nombre    : str
    tipo         : str    # "poblacional" | "trayectoria"
    mejor_valor  : float
    t_inicio     : float  # segundos desde el inicio del pipeline
    t_fin        : float
    n_iters      : int    # iteraciones/generaciones ejecutadas
    dtw_deltas   : list   = None  # historial de deltas DTW de este turno


@dataclass
class PipelineResult:
    """Resultado completo del pipeline híbrido."""
    mejor_valor_global   : float
    mejor_solucion_global: list[int]
    historial_global     : list[float]   # mejor valor por iteración acumulada
    historial_inst_global: list[float]   # fitness instantáneo
    dtw_deltas_global    : list[float]   # delta DTW por iteración (donde hay datos)
    log_switches         : list[SwitchLog]
    valor_optimo         : float

    @property
    def gap_pct(self) -> float | None:
        if self.valor_optimo == 0:
            return None
        return 100.0 * (self.valor_optimo - self.mejor_valor_global) / self.valor_optimo

    @property
    def n_switches(self) -> int:
        return len(self.log_switches)


# ── Función pública principal ─────────────────────────────────────────────────

def ejecutar_pipeline(
    inst               : MKPInstance,
    tiempo_max         : float = 120.0,
    stag_cfg           : StagnationConfig | None = None,
    pop_injection_mode : str = "mixed",
    verbose            : bool = True,
) -> PipelineResult:
    """Ejecuta el pipeline híbrido rotando MH hasta agotar tiempo_max.

    Parameters
    ----------
    inst               : Instancia del MKP.
    tiempo_max         : Tiempo máximo en segundos.
    stag_cfg           : Configuración del monitor DTW. Si None usa valores por defecto.
    pop_injection_mode : Estrategia de inyección para MH poblacionales.
                         "random"  → 1 inyectada + resto aleatorio.
                         "mutated" → toda la población mutada de la inyectada.
                         "mixed"   → 50% mutaciones + 50% aleatorios.
    verbose            : Imprime log de cada switch en consola.

    Returns
    -------
    PipelineResult con la mejor solución global y el log de switches.
    """
    if stag_cfg is None:
        stag_cfg = StagnationConfig()

    # Estado global
    solucion_global  : list[int] | None = None
    valor_global     : float = float("-inf")
    historial_global : list[float] = []
    historial_inst_global : list[float] = []
    dtw_deltas_global: list[float] = []
    log_switches     : list[SwitchLog] = []

    turno        = "poblacional"
    epoch_ctr    = 0
    t_inicio     = time.time()

    if verbose:
        print("\n" + "=" * 62)
        print("  PIPELINE HIBRIDO DTW -- INICIO")
        print(f"  Tiempo max : {tiempo_max}s")
        print(f"  Inyeccion  : {pop_injection_mode}")
        print(f"  Poblacional: {POOL_POBLACIONAL}")
        print(f"  Trayectoria: {POOL_TRAYECTORIA}")
        print("=" * 62)

    while (time.time() - t_inicio) < tiempo_max:
        t_mh_inicio = time.time() - t_inicio

        # Elegir MH del pool correspondiente
        if turno == "poblacional":
            mh = random.choice(POOL_POBLACIONAL)
            tipo = "poblacional"
        else:
            mh = random.choice(POOL_TRAYECTORIA)
            tipo = "trayectoria"

        if verbose:
            elapsed = time.time() - t_inicio
            print(f"\n  [{elapsed:06.1f}s] > {mh:4s} ({tipo}) | global = {valor_global:.1f}")

        # Ejecutar la MH con stag_strategy="abort" (termina al estancarse)
        resultado = _ejecutar_mh(
            mh_nombre          = mh,
            inst               = inst,
            solucion_global    = solucion_global,
            stag_cfg           = stag_cfg,
            pop_injection_mode = pop_injection_mode,
            epoch_idx          = epoch_ctr,
            verbose            = verbose,
        )
        epoch_ctr += 1

        # Actualizar mejor global si mejoró
        if resultado.mejor_valor > valor_global:
            valor_global    = resultado.mejor_valor
            solucion_global = list(resultado.mejor_solucion)

        # Acumular historial y deltas DTW, registrar switch
        historial_global.extend(resultado.historial)
        historial_inst_global.extend(getattr(resultado, 'historial_inst', []) or [])

        # Alinear dtw_deltas con historial: rellenar NaN al inicio (ventana no lista)
        mh_deltas = getattr(resultado, 'dtw_deltas', []) or []
        n_hist    = len(resultado.historial)
        n_deltas  = len(mh_deltas)
        padded    = [float('nan')] * (n_hist - n_deltas) + list(mh_deltas)
        dtw_deltas_global.extend(padded)

        t_mh_fin = time.time() - t_inicio
        n_iters  = len(resultado.historial)

        log_switches.append(SwitchLog(
            mh_nombre   = mh,
            tipo        = tipo,
            mejor_valor = resultado.mejor_valor,
            t_inicio    = t_mh_inicio,
            t_fin       = t_mh_fin,
            n_iters     = n_iters,
            dtw_deltas  = mh_deltas,
        ))

        if verbose:
            dur = t_mh_fin - t_mh_inicio
            print(f"          Duracion: {dur:.1f}s | iters: {n_iters} "
                  f"| mejor MH: {resultado.mejor_valor:.1f} "
                  f"| global: {valor_global:.1f}")

        # Alternar turno
        turno = "trayectoria" if turno == "poblacional" else "poblacional"

    elapsed_total = time.time() - t_inicio
    if verbose:
        print(f"\n  [{elapsed_total:.1f}s] TIEMPO AGOTADO")
        print(f"  Mejor global  : {valor_global:.1f}")
        print(f"  Total switches: {len(log_switches)}")
        if inst.valor_optimo > 0:
            gap = 100.0 * (inst.valor_optimo - valor_global) / inst.valor_optimo
            print(f"  Gap vs optimo : {gap:.2f}%")
        print("=" * 62)

    return PipelineResult(
        mejor_valor_global    = valor_global,
        mejor_solucion_global = solucion_global or [],
        historial_global      = historial_global,
        historial_inst_global = historial_inst_global,
        dtw_deltas_global     = dtw_deltas_global,
        log_switches          = log_switches,
        valor_optimo          = inst.valor_optimo,
    )


# ── Dispatcher interno ────────────────────────────────────────────────────────

def _ejecutar_mh(
    mh_nombre          : str,
    inst               : MKPInstance,
    solucion_global    : list[int] | None,
    stag_cfg           : StagnationConfig,
    pop_injection_mode : str,
    epoch_idx          : int,
    verbose            : bool,
):
    """Ejecuta una MH específica con abort activado."""

    if mh_nombre == "GA":
        params = GAParams(
            pop_size=50, generations=500, epochs=1,
            injection_mode=pop_injection_mode,
            use_stagnation=True, stag_cfg=stag_cfg,
        )
        return _ga_epoch(inst, params, epoch_idx=epoch_idx, verbose=verbose,
                         sol_inyectada=solucion_global)

    elif mh_nombre == "PSO":
        params = PSOParams(
            pop_size=30, iterations=300, epochs=1,
            injection_mode=pop_injection_mode,
            use_stagnation=True, stag_cfg=stag_cfg,
        )
        return _pso_epoch(inst, params, epoch_idx=epoch_idx, verbose=verbose,
                          sol_inyectada=solucion_global)

    elif mh_nombre == "GWO":
        params = GWOParams(
            pop_size=30, iterations=300, epochs=1,
            injection_mode=pop_injection_mode,
            use_stagnation=True, stag_cfg=stag_cfg,
        )
        return _gwo_epoch(inst, params, epoch_idx=epoch_idx, verbose=verbose,
                          sol_inyectada=solucion_global)

    elif mh_nombre == "SA":
        params = SAParams(
            T_inicial=5_000.0, T_final=1.0, alpha=0.97, iter_por_T=50,
            epochs=1, use_stagnation=True, stag_cfg=stag_cfg,
        )
        return _sa_epoch(inst, params, epoch_idx=epoch_idx, verbose=verbose,
                         sol_inicial=solucion_global)

    elif mh_nombre == "TS":
        params = TSParams(
            epochs=1, iterations=2_000,
            use_stagnation=True, stag_cfg=stag_cfg,
        )
        return _ts_epoch(inst, params, epoch_idx=epoch_idx, verbose=verbose,
                         sol_inicial=solucion_global)

    else:
        raise ValueError(f"MH desconocida: '{mh_nombre}'")
