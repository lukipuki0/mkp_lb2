"""
================================================================================
Test de Verificación CEC 2022 Benchmark Suite
================================================================================
Este script evalúa las 12 funciones de benchmark CEC 2022 en:
 1. Su punto óptimo conocido (x = shift vector o_1), verificando que f(x_opt) == bias
    con una tolerancia menor a 1e-6.
 2. El punto de origen [0, 0, ..., 0].

Verifica la corrección para nD = 20 y nD = 10.
================================================================================
"""

import sys
import os
import numpy as np

# Agregar directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from continuous_benchmark.funciones_cec2022 import (
    cec2022_func,
    get_cec2022_optimum_point,
    OFFICIAL_BIASES,
)


def run_verification(n_dim: int = 20) -> bool:
    print("\n" + "=" * 80)
    print(f"  VERIFICACIÓN DE FUNCIONES BENCHMARK CEC 2022 (Dimensión nD = {n_dim})")
    print("=" * 80)
    print(f"{'Función':<8} {'Bias Oficial':<14} {'f(x_opt)':<14} {'Diferencia (Error)':<20} {'f(x=0)':<14} {'Estado':<8}")
    print("-" * 80)

    all_passed = True
    tol = 1e-6

    for func_num in range(1, 13):
        bias = OFFICIAL_BIASES[func_num]
        
        # 1. Evaluar en punto óptimo (x = shift vector o_1)
        x_opt = get_cec2022_optimum_point(func_num, n_dim)
        f_opt = cec2022_func(x_opt, func_num, n_dim=n_dim)
        diff = abs(f_opt - bias)

        # 2. Evaluar en el origen x = [0, ..., 0]
        x_zero = np.zeros(n_dim)
        f_zero = cec2022_func(x_zero, func_num, n_dim=n_dim)

        passed = diff <= tol
        if not passed:
            all_passed = False

        status = "PASSED" if passed else "FAILED"
        print(f"F{func_num:<7} {bias:<14.2f} {f_opt:<14.6f} {diff:<20.2e} {f_zero:<14.2f} {status:<8}")

    print("=" * 80)
    if all_passed:
        print(f" [SUCCESS] Todas las 12 funciones verificaron correctamente su valor óptimo (nD={n_dim}).")
    else:
        print(f" [FAILURE] Una o más funciones no alcanzaron el bias en el punto óptimo (nD={n_dim}).")
    return all_passed


def main():
    print("Iniciando suite de verificación CEC 2022...")
    passed_20 = run_verification(n_dim=20)
    passed_10 = run_verification(n_dim=10)

    if passed_20 and passed_10:
        print("\n>>> PRUEBA COMPLETA: TODOS LOS TESTS PASARON EXITOSAMENTE. <<<")
        sys.exit(0)
    else:
        print("\n>>> PRUEBA COMPLETA: HUBO ERRORES EN LA VERIFICACIÓN. <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
