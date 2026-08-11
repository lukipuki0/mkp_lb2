"""
mezclas_mh/woa_abc/run_experiment.py
------------------------------------
Script ejecutor para comparar experimentalmente las 4 variantes de WOA-ABC
(Variante A, Variante B, Variante C y MDG-WABC B+C) frente a WOA y ABC puros.

Resuelve:
  1. Instancia discreta MKP (mknapcb1)
  2. TODAS las 12 funciones continuas CEC2022 (F1 a F12)

Uso:
  python mezclas_mh/woa_abc/run_experiment.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mkp_core.data_loader import cargar_instancias
from mkp_core.problem import MKPInstance
from continuous_benchmark.funciones_cec2022 import get_test_functions

from mh.woa import WOAParams as WOAMKPParams, ejecutar_epoch as woa_mkp_epoch
from mh.abc import ABCParams as ABCMKPParams, ejecutar_epoch as abc_mkp_epoch
from continuous_benchmark.mh.woa import WOAParams as WOAContParams, ejecutar_epoch as woa_cont_epoch
from continuous_benchmark.mh.abc import ABCParams as ABCContParams, ejecutar_epoch as abc_cont_epoch

from mezclas_mh.woa_abc import (
    VariantAParams, variant_a_epoch, variant_a_epoch_continuo,
    VariantBParams, variant_b_epoch, variant_b_epoch_continuo,
    VariantCParams, variant_c_epoch, variant_c_epoch_continuo,
    MDGWABCParams, mdg_wabc_epoch, mdg_wabc_epoch_continuo,
)


def ejecutar_comparativa(iterations: int = 100, n_dim: int = 10):
    print("=" * 80)
    print(f" EJECUTANDO EXPERIMENTO COMPARATIVO WOA-ABC ({iterations} iters, Dim CEC2022: {n_dim})")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. MKP Discreto
    # ─────────────────────────────────────────────────────────────────────────
    inst_raw = cargar_instancias("instancias/mknapcb1.txt")[0]
    inst = MKPInstance.from_dict(inst_raw)
    inst_name = inst_raw.get("nombre", f"MKP_{inst.n}x{inst.m}")

    print(f"\n--- 1. MKP Discreto (Instancia: {inst_name}, n={inst.n}, m={inst.m}) ---")
    resultados_mkp = {}

    res_woa = woa_mkp_epoch(inst, WOAMKPParams(iterations=iterations), verbose=False)
    resultados_mkp["WOA Puro"] = res_woa.mejor_valor

    res_abc = abc_mkp_epoch(inst, ABCMKPParams(iterations=iterations), verbose=False)
    resultados_mkp["ABC Puro"] = res_abc.mejor_valor

    res_a = variant_a_epoch(inst, VariantAParams(iterations=iterations), verbose=False)
    resultados_mkp["Variante A (Switch a(t))"] = res_a.mejor_valor

    res_b = variant_b_epoch(inst, VariantBParams(iterations=iterations), verbose=False)
    resultados_mkp["Variante B (Momentum)"] = res_b.mejor_valor

    res_c = variant_c_epoch(inst, VariantCParams(iterations=iterations), verbose=False)
    resultados_mkp["Variante C (Diversidad)"] = res_c.mejor_valor

    res_mdg = mdg_wabc_epoch(inst, MDGWABCParams(iterations=iterations), verbose=False)
    resultados_mkp["MDG-WABC (B + C)"] = res_mdg.mejor_valor

    for algo, val in resultados_mkp.items():
        print(f"  {algo:<30}: {val:10.1f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Benchmark Continuo CEC2022 (F1 a F12)
    # ─────────────────────────────────────────────────────────────────────────
    funciones_cec = get_test_functions(n_dim=n_dim)
    print(f"\n--- 2. Benchmark Continuo CEC2022 (Todas las 12 Funciones F1-F12, Dim: {n_dim}) ---")

    resultados_continuo_global = {}

    print(f"\n{'Función':<32} {'WOA':>10} {'ABC':>10} {'Var A':>10} {'Var B':>10} {'Var C':>10} {'MDG-WABC':>10}")
    print("-" * 100)

    for fn in funciones_cec:
        res_fn = {}

        v_woa = woa_cont_epoch(fn, WOAContParams(iterations=iterations), verbose=False).mejor_valor
        v_abc = abc_cont_epoch(fn, ABCContParams(iterations=iterations), verbose=False).mejor_valor
        v_a = variant_a_epoch_continuo(fn, VariantAParams(iterations=iterations), verbose=False).mejor_valor
        v_b = variant_b_epoch_continuo(fn, VariantBParams(iterations=iterations), verbose=False).mejor_valor
        v_c = variant_c_epoch_continuo(fn, VariantCParams(iterations=iterations), verbose=False).mejor_valor
        v_mdg = mdg_wabc_epoch_continuo(fn, MDGWABCParams(iterations=iterations), verbose=False).mejor_valor

        res_fn["WOA Puro"] = v_woa
        res_fn["ABC Puro"] = v_abc
        res_fn["Variante A"] = v_a
        res_fn["Variante B"] = v_b
        res_fn["Variante C"] = v_c
        res_fn["MDG-WABC"] = v_mdg

        resultados_continuo_global[fn.name] = res_fn

        print(f"{fn.name:<32} {v_woa:10.2f} {v_abc:10.2f} {v_a:10.2f} {v_b:10.2f} {v_c:10.2f} {v_mdg:10.2f}")

    # Guardar reporte JSON
    out_dir = "resultados"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "woa_abc_benchmark_completo.json")

    data_export = {
        "mkp_instancia": inst_name,
        "resultados_mkp": resultados_mkp,
        "n_dim_continuo": n_dim,
        "iteraciones": iterations,
        "resultados_continuo_cec2022": resultados_continuo_global,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data_export, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f" Experimento completado con éxito.")
    print(f" Resultados guardados en: '{out_file}'")
    print("=" * 80)


if __name__ == "__main__":
    ejecutar_comparativa(iterations=60, n_dim=10)
