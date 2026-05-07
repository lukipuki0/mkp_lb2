"""
ts_benchmark_variants.py
------------------------
Script maestro para evaluar y comparar estadisticamente las 9 variantes de
manejo de estancamiento en Tabu Search (TS) para el MKP.
"""

import copy
import datetime
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from ts_mkp import config
from ts_mkp.algorithm     import TSParams, ejecutar_ts
from mkp_core.data_loader import cargar_instancias, seleccionar_instancia
from mkp_core.problem     import MKPInstance
from dtw_stagnation       import StagnationConfig


def guardar_resumen_txt(resultado, label: str, output_dir: str) -> None:
    ruta = os.path.join(output_dir, f"resultado_{label}.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        sep = "=" * 68
        f.write(f"{sep}\n")
        f.write(f"  RESULTADOS TS-MKP - VARIANTE: {label}\n")
        f.write(f"{sep}\n")
        f.write(f"  {'Epoch':>6}  {'Mejor valor':>14}  {'Iteraciones':>13}  {'Stag. fires':>12}\n")
        f.write(f"  {'------':>6}  {'------------':>14}  {'-------------':>13}  {'------------':>12}\n")
        for ep in resultado.epochs:
            f.write(
                f"  {ep.epoch_idx + 1:>6}  "
                f"{ep.mejor_valor:>14.1f}  "
                f"{ep.iteraciones:>13}  "
                f"{ep.stagnation_fires:>12}\n"
            )
        f.write(f"{sep}\n")
        f.write(f"  Mejor valor global    : {resultado.mejor_valor_global:.1f}\n")
        f.write(f"  Valor optimo conocido : {resultado.valor_optimo:.1f}\n")
        gap = resultado.gap_pct
        if gap is not None:
            f.write(f"  Gap relativo          : {gap:.2f}%\n")
        else:
            f.write("  Gap relativo          : N/A\n")
        f.write(f"{sep}\n")
    print(f"  [txt]  Saved '{ruta}'")


def graficar_convergencia_variante(resultado, label: str, output_dir: str, dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for ep in resultado.epochs:
        ax.plot(ep.historial, alpha=0.65, linewidth=1.2,
                label=f"Epoch {ep.epoch_idx + 1} (fires={ep.stagnation_fires})")
    if resultado.valor_optimo != 0:
        ax.axhline(y=resultado.valor_optimo, color="red", linestyle="--",
                   linewidth=1.4, label=f"Known Optimal ({resultado.valor_optimo:.0f})")
    ax.set_title(f"TS Convergence - MKP [{label}]", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Value", fontsize=11)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(output_dir, f"convergencia_{label}.png")
    fig.savefig(ruta, dpi=dpi)
    print(f"  [plot] Saved '{ruta}'")
    plt.close(fig)


def main() -> None:
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("resultados", "resultados_ts", f"benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    if config.RANDOM_SEED is not None:
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

    instancias = cargar_instancias(config.INSTANCE_URL)
    data       = seleccionar_instancia(instancias, config.INSTANCE_INDEX)
    inst       = MKPInstance.from_dict(data)

    stag_cfg = StagnationConfig(
        window           = config.STAG_WINDOW,
        band             = config.STAG_BAND,
        min_slope        = config.STAG_MIN_SLOPE,
        plateau_max      = config.STAG_PLATEAU_MAX,
        patience         = config.STAG_PATIENCE,
        use_ddtw         = config.STAG_USE_DDTW,
        adapt_thresholds = config.STAG_ADAPT,
        p_low            = config.STAG_P_LOW,
        p_high           = config.STAG_P_HIGH,
    )

    base_params = TSParams(
        epochs          = config.EPOCHS,
        iterations      = config.ITERATIONS,
        tabu_tenure     = config.TABU_TENURE,
        neighborhood_sz = config.NEIGHBORHOOD_SZ,
        use_stagnation  = config.USE_STAGNATION,
        stag_cfg        = stag_cfg,
        stag_max_fires  = config.STAG_MAX_FIRES,
    )

    variantes = [
        ("Original", "random_restart"),
        ("V1",       "v1_exploit"),
        ("V2",       "v2_cycle"),
        ("V3",       "v3_explore"),
        ("V4",       "v4_nonlinear"),
        ("V5",       "v5_heuristic"),
        ("V6",       "v6_lp"),
        ("V7",       "v7_tabu_lp"),
        ("V8",       "v8_ruin_recreate"),
    ]

    resultados_brutos = {}
    
    print("\n" + "=" * 60)
    print(" STARTING TS-MKP VARIANT BENCHMARK")
    print(f" Output: {output_dir}")
    print("=" * 60)

    for label, strategy in variantes:
        print(f"\n{'#' * 60}\n# VARIANT: {label} (strategy={strategy})\n{'#' * 60}")
        params = copy.deepcopy(base_params)
        params.stag_strategy = strategy
        resultado = ejecutar_ts(inst, params, verbose=True)
        resultados_brutos[label] = resultado.valores_por_epoch
        print(f"\n  >> {label} DONE | Best = {resultado.mejor_valor_global:.1f}")
        guardar_resumen_txt(resultado, label, output_dir)
        graficar_convergencia_variante(resultado, label, output_dir)

    report = []
    def log(text: str):
        print(text)
        report.append(text)

    log("\n" + "=" * 60 + "\n FINAL RESULTS & STATISTICS\n" + "=" * 60)
    log(f"\n  {'Variant':<12} | {'Mean':>12} | {'Std Dev':>10} | {'Best':>10}\n  " + "-" * 52)
    for label, _ in variantes:
        datos = resultados_brutos[label]
        log(f"  {label:<12} | {np.mean(datos):>12.1f} | {np.std(datos):>10.2f} | {np.max(datos):>10.1f}")

    log("\n  --- Wilcoxon Test (vs Original) ---")
    data_orig = resultados_brutos["Original"]
    for label, _ in variantes:
        if label != "Original":
            try:
                _, p = wilcoxon(resultados_brutos[label], data_orig, alternative="greater")
                sig  = " *" if p < 0.05 else ""
                log(f"  {label:<12} vs Original -> p-value = {p:.5f}{sig}")
            except ValueError:
                log(f"  {label:<12} vs Original -> p-value = N/A")

    with open(os.path.join(output_dir, "estadisticas_benchmark.txt"), "w") as f:
        f.write("\n".join(report))

    nombres = [l for l, _ in variantes]
    datos   = [resultados_brutos[l] for l in nombres]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(datos, labels=nombres, showfliers=True)
    ax.set_ylabel("Best Value Obtained")
    ax.set_title("Result Distribution by Variant (TS-MKP)", fontweight="bold")
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "distribucion_variantes.png"), dpi=150)
    plt.close(fig)

    print("\n  BENCHMARK COMPLETE.")

if __name__ == "__main__":
    main()
