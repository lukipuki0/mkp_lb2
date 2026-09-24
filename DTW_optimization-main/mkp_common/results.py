"""
Resultados: persistencia de datos, carga y diagnóstico en consola para MKP.
Especializado en gestionar métricas y estadísticas de múltiples metaheurísticas.
Las tareas de ploteo avanzado y para paper se delegan a los scripts de análisis dedicados.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# =============================================================================
# DIAGNOSTICO DTW (Texto / Consola)
# =============================================================================


def diagnosticar_dtw(historial_dtw: List[Dict]) -> None:
    """Imprime resumen del comportamiento del DTW durante una corrida."""
    ready = [h for h in historial_dtw if h.get("ready")]
    fires = [h for h in ready if h.get("fire")]

    if not ready:
        print("  [DTW] No alcanzó el warm-up")
        return

    deltas = [h["delta"] for h in ready]
    d1s = [h["D1_vs_ramp"] for h in ready]
    d2s = [h["D2_vs_const"] for h in ready]

    print(f"  [DTW] Iteraciones activas: {len(ready)}")
    print(f"  [DTW] Fires disparados:   {len(fires)}")
    print(f"  [DTW] Delta promedio:      {np.mean(deltas):.1f}")
    print(f"  [DTW] Delta min/max:       {np.min(deltas):.1f} / {np.max(deltas):.1f}")
    print(f"  [DTW] D1 promedio:         {np.mean(d1s):.1f}")
    print(f"  [DTW] D2 promedio:         {np.mean(d2s):.1f}")


def print_summary(resultados: List[dict], inst: dict) -> None:
    """Imprime resumen final de múltiples epochs en consola."""
    fits = [r["mejor_fitness"] for r in resultados]
    optimo = inst["optimo"]

    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  Mejor:      {np.max(fits):.1f}")
    print(f"  Promedio:   {np.mean(fits):.1f}")
    print(f"  Peor:       {np.min(fits):.1f}")
    print(f"  Desv. Est.: {np.std(fits):.1f}")
    if optimo > 0:
        print(f"  Optimo:     {optimo:.0f}")
        print(f"  Gap:        {100 - np.max(fits) / optimo * 100:.2f}%")
    print(f"  Epochs:     {len(resultados)}")


# =============================================================================
# GUARDADO / CARGA JSON (Lógica de Persistencia Compartida)
# =============================================================================


def save_results(
    resultados: List[dict],
    path: str,
    mh_name: str = "PSO",
    extra_info: Optional[dict] = None,
    optimo_conocido: Optional[float] = None,
) -> None:
    """
    Guarda resultados a JSON de forma limpia (removiendo tipos de numpy
    para evitar problemas de serialización).

    Args:
        resultados:       Lista de dicts obtenidos de una tanda de ejecución.
        path:             Ruta del archivo .json a escribir.
        mh_name:          Nombre de la metaheurística para identificación.
        extra_info:       Configuraciones adicionales del experimento.
        optimo_conocido:  Óptimo teórico de la instancia si existe.
    """
    fits = [r["mejor_fitness"] for r in resultados]
    tiempos = [r.get("tiempo", 0) for r in resultados]
    mejor = float(np.max(fits))

    stats = {
        "mejor": mejor,
        "promedio": float(np.mean(fits)),
        "peor": float(np.min(fits)),
        "std": float(np.std(fits)),
        "tiempo_promedio": float(np.mean(tiempos)),
        "tiempo_std": float(np.std(tiempos)),
    }

    if optimo_conocido and optimo_conocido > 0:
        stats["optimo_conocido"] = optimo_conocido
        stats["gap_al_optimo"] = round(100 - mejor / optimo_conocido * 100, 4)

    salida = {
        "mh": mh_name,
        "epochs": len(resultados),
        "optimo_conocido": optimo_conocido if optimo_conocido and optimo_conocido > 0 else None,
        "fitness": fits,
        "fire_counts": [r["fire_count"] for r in resultados],
        "ganancias": [r["ganancia"] for r in resultados],
        "tiempos": tiempos,
        "stats": stats,
    }
    if extra_info:
        salida["info"] = extra_info

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(salida, f, indent=2)
    print(f"  Resultados guardados: {path}")


def load_results(path: str) -> dict:
    """Carga resultados previamente persistidos desde un JSON."""
    with open(path) as f:
        return json.load(f)
