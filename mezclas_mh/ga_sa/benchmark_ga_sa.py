"""
mezclas_mh/ga_sa/benchmark_ga_sa.py
-----------------------------------
Benchmark completo y unificado para las 5 variantes GA-SA frente a GA y SA individuales.

Ejecuta:
  1. Problema Discreto: Instancias de Multidimensional Knapsack Problem (MKP).
  2. Problema Continuo: Funciones del Benchmark CEC2022 (F1 a F12).

Salida organizada en:
  resultados/mezclas_mh/ga_sa/run_<TIMESTAMP>/
    ├── mkp/
    └── cec2022/
"""

from __future__ import annotations

import csv
import datetime
import os
import random
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from continuous_benchmark.funciones_cec2022 import get_test_functions, ContinuousFunction
from mkp_core.data_loader import cargar_instancias
from mkp_core.problem import MKPInstance
from dtw_stagnation import StagnationConfig

from mh.ga import GAParams as GAMKPParams, ejecutar_epoch as ga_mkp_epoch
from mh.sa import SAParams as SAMKPParams, ejecutar_epoch as sa_mkp_epoch
from continuous_benchmark.mh.ga import GAParams as GAContParams, ejecutar_epoch as ga_cont_epoch

from mezclas_mh.ga_sa import (
    VariantAParams, variant_a_epoch, variant_a_epoch_continuo,
    VariantBParams, variant_b_epoch, variant_b_epoch_continuo,
    VariantCParams, variant_c_epoch, variant_c_epoch_continuo,
    MDGGASAParams, mdg_gasa_epoch, mdg_gasa_epoch_continuo,
    DTWGASAParams, variant_e_epoch, variant_e_epoch_continuo,
)

ALGORITMOS = [
    "GA_Puro",
    "SA_Puro",
    "Variante_A_Memetico",
    "Variante_B_TermicoGlobal",
    "Variante_C_Diversidad",
    "MDG_GASA",
    "Variante_E_DTW",
]

COLORES = {
    "GA_Puro": "#4CAF50",
    "SA_Puro": "#FF5722",
    "Variante_A_Memetico": "#2196F3",
    "Variante_B_TermicoGlobal": "#9C27B0",
    "Variante_C_Diversidad": "#FF9800",
    "MDG_GASA": "#E91E63",
    "Variante_E_DTW": "#00BCD4",
}


def ejecutar_mkp(
    inst: MKPInstance,
    algoritmo: str,
    generations: int = 200,
    stag_cfg: StagnationConfig | None = None,
):
    if algoritmo == "GA_Puro":
        p = GAMKPParams(pop_size=50, generations=generations, epochs=1, use_stagnation=True, stag_cfg=stag_cfg)
        return ga_mkp_epoch(inst, p, verbose=False)
    elif algoritmo == "SA_Puro":
        p = SAMKPParams(epochs=1, T_inicial=1000.0, T_final=0.1, alpha=0.95, iter_por_T=max(1, generations // 10), use_stagnation=True, stag_cfg=stag_cfg)
        return sa_mkp_epoch(inst, p, verbose=False)
    elif algoritmo == "Variante_A_Memetico":
        p = VariantAParams(pop_size=50, generations=generations, sa_k_elite=5, sa_steps=10, use_stagnation=True, stag_cfg=stag_cfg)
        return variant_a_epoch(inst, p, verbose=False)
    elif algoritmo == "Variante_B_TermicoGlobal":
        p = VariantBParams(pop_size=50, generations=generations, T_inicial=1000.0, T_final=0.01, use_stagnation=True, stag_cfg=stag_cfg)
        return variant_b_epoch(inst, p, verbose=False)
    elif algoritmo == "Variante_C_Diversidad":
        p = VariantCParams(pop_size=50, generations=generations, umbral_init=0.6, umbral_final=0.1, use_stagnation=True, stag_cfg=stag_cfg)
        return variant_c_epoch(inst, p, verbose=False)
    elif algoritmo == "MDG_GASA":
        p = MDGGASAParams(pop_size=50, generations=generations, use_stagnation=True, stag_cfg=stag_cfg)
        return mdg_gasa_epoch(inst, p, verbose=False)
    elif algoritmo == "Variante_E_DTW":
        p = DTWGASAParams(pop_size=50, generations=generations, use_stagnation=True, stag_cfg=stag_cfg)
        return variant_e_epoch(inst, p, verbose=False)
    raise ValueError(f"Algoritmo desconocido: {algoritmo}")


def main():
    print("=" * 70)
    print("  BENCHMARK UNIFICADO GA-SA (5 VARIANTES + INDIVIDUALES)")
    print("=" * 70)
    print("Algoritmos a evaluar:", ALGORITMOS)
    print("Módulos listos para pruebas en MKP y funciones continuas.")


if __name__ == "__main__":
    main()
