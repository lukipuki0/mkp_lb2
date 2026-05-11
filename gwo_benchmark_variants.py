"""
gwo_benchmark_variants.py
-------------------------
Script maestro para evaluar y comparar estadisticamente las 9 variantes de
manejo de estancamiento en el Grey Wolf Optimizer (GWO) para el MKP.

Genera:
  - Salida verbose en consola (epochs, stagnation fires, etc.)
  - Un grafico de convergencia por variante (convergencia_<label>.png)
  - Un boxplot comparativo global (distribucion_variantes.png)
  - Estadisticas (Media, Std, Wilcoxon) por consola
"""

import copy
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from gwo_mkp import config
from gwo_mkp.algorithm import GWOParams, ejecutar_gwo
from mkp_core.data_loader import cargar_instancias, seleccionar_instancia
from mkp_core.problem import MKPInstance
from dtw_stagnation import StagnationConfig


# -- Utilidades de exportacion -------------------------------------------------

def guardar_resumen_txt(resultado, label: str, output_dir: str) -> None:
    """Guarda un TXT con la tabla de resultados para la variante."""
    ruta = os.path.join(output_dir, f"resultado_{label}.txt")

    with open(ruta, "w", encoding="utf-8") as f:
        sep = "=" * 65
        f.write(f"{sep}\n")
        f.write(f"  RESULTADOS GWO-MKP - VARIANTE: {label}\n")
        f.write(f"{sep}\n")
        f.write(
            f"  {'Epoch':>6}  {'Mejor valor':>14}  "
            f"{'Iteraciones':>12}  {'Stag. fires':>12}\n"
        )
        f.write(
            f"  {'------':>6}  {'------------':>14}  "
            f"{'------------':>12}  {'------------':>12}\n"
        )
        for ep in resultado.epochs:
            f.write(
                f"  {ep.epoch_idx + 1:>6}  "
                f"{ep.mejor_valor:>14.1f}  "
                f"{ep.iteraciones:>12}  "
                f"{ep.stagnation_fires:>12}\n"
            )

        f.write(f"{sep}\n")
        f.write(f"  Mejor valor global       : {resultado.mejor_valor_global:.1f}\n")
        f.write(f"  Valor optimo conocido    : {resultado.valor_optimo:.1f}\n")

        gap = resultado.gap_pct
        if gap is not None:
            f.write(f"  Gap relativo             : {gap:.2f}%\n")
        else:
            f.write("  Gap relativo             : N/A (optimo no disponible en esta instancia)\n")
        f.write(f"{sep}\n")

    print(f"  [txt] Saved '{ruta}'")


# -- Grafico de convergencia individual ----------------------------------------

def graficar_convergencia_variante(resultado, label: str, output_dir: str, dpi: int = 150) -> None:
    """Genera y guarda el grafico de convergencia de una variante."""
    fig, ax = plt.subplots(figsize=(11, 5))

    for ep in resultado.epochs:
        ax.plot(ep.historial, alpha=0.65, linewidth=1.2,
                label=f"Epoch {ep.epoch_idx + 1} (fires={ep.stagnation_fires})")

    if resultado.valor_optimo != 0:
        ax.axhline(
            y=resultado.valor_optimo,
            color="red", linestyle="--", linewidth=1.4,
            label=f"Known Optimal ({resultado.valor_optimo:.0f})",
        )

    ax.set_title(f"GWO Convergence - MKP [{label}]", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Value (Epoch)", fontsize=11)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    ruta = os.path.join(output_dir, f"convergencia_{label}.png")
    fig.savefig(ruta, dpi=dpi)
    print(f"  [plot] Saved '{ruta}'")
    plt.close(fig)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    # -- Configuracion de directorio de salida ---------------------------------
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("resultados", "resultados_gwo", f"benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # -- Configuracion inicial -------------------------------------------------
    if config.RANDOM_SEED is not None:
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

    # Cargar instancia
    instancias = cargar_instancias(config.INSTANCE_URL)
    data       = seleccionar_instancia(instancias, config.INSTANCE_INDEX)
    inst       = MKPInstance.from_dict(data)

    # Configuracion de estancamiento base
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

    # Parametros base del GWO
    base_params = GWOParams(
        pop_size       = config.POP_SIZE,
        iterations     = config.ITERATIONS,
        epochs         = config.EPOCHS,
        v_max          = config.V_MAX,
        G1_i           = config.G1_INITIAL,
        G1_f           = config.G1_FINAL,
        G2_i           = config.G2_INITIAL,
        G2_f           = config.G2_FINAL,
        G3_i           = config.G3_INITIAL,
        G3_f           = config.G3_FINAL,
        use_stagnation = config.USE_STAGNATION,
        stag_cfg       = stag_cfg,
        stag_max_fires = config.STAG_MAX_FIRES,
    )

    # -- Variantes a evaluar ---------------------------------------------------
    variantes = [
        ("Original", "adapt_g"),
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
    resultados_obj    = {}

    # -- Ejecucion del Benchmark -----------------------------------------------
    print("\n" + "=" * 60)
    print(" STARTING GWO-MKP VARIANT BENCHMARK")
    print("=" * 60)

    for label, strategy in variantes:
        print(f"\n{'#' * 60}")
        print(f"# VARIANT: {label} (strategy={strategy})")
        print(f"{'#' * 60}")

        params = copy.deepcopy(base_params)
        params.stag_strategy = strategy

        resultado = ejecutar_gwo(inst, params, verbose=True)

        resultados_brutos[label] = resultado.valores_por_epoch
        resultados_obj[label]    = resultado

        print(f"\n  >> {label} DONE | Best global = {resultado.mejor_valor_global:.1f}")

        guardar_resumen_txt(resultado, label, output_dir)
        graficar_convergencia_variante(resultado, label, output_dir)

    # -- Analisis y Reporte ----------------------------------------------------
    report_text = []

    def print_and_log(text: str):
        print(text)
        report_text.append(text)

    print_and_log("\n" + "=" * 60)
    print_and_log(" FINAL RESULTS & STATISTICS")
    print_and_log("=" * 60)

    # 1. Medias y Desviaciones Estandar
    print_and_log(f"\n  {'Variant':<10} | {'Mean':<12} | {'Std Dev':<10} | {'Best':<10}")
    print_and_log("  " + "-" * 50)
    for label, _ in variantes:
        datos = resultados_brutos[label]
        print_and_log(f"  {label:<10} | {np.mean(datos):<12.1f} | {np.std(datos):<10.2f} | {np.max(datos):<10.1f}")

    # 2. Test de Wilcoxon (Comparado con Original)
    print_and_log("\n  --- Wilcoxon Test (vs Original, alternative='greater') ---")
    data_original = resultados_brutos["Original"]
    for label, _ in variantes:
        if label == "Original":
            continue
        data_variante = resultados_brutos[label]
        try:
            stat, p = wilcoxon(data_variante, data_original, alternative="greater")
            sig = " *" if p < 0.05 else ""
            print_and_log(f"  {label:<10} vs Original -> p-value = {p:.5f}{sig}")
        except ValueError:
            print_and_log(f"  {label:<10} vs Original -> p-value = N/A (identical data)")

    # 3. Guardar reporte en texto
    report_path = os.path.join(output_dir, "estadisticas_benchmark.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_text))
    print(f"\n  [benchmark] Reporte guardado en '{report_path}'")

    # 4. Boxplot comparativo global
    nombres_labels = [label for label, _ in variantes]
    datos_plot     = [resultados_brutos[label] for label in nombres_labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(datos_plot, labels=nombres_labels, showfliers=True)
    ax.set_ylabel("Best Value Obtained", fontsize=11)
    ax.set_title("Result Distribution by Variant (GWO-MKP)", fontsize=14, fontweight="bold")
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "distribucion_variantes.png")
    fig.savefig(plot_path, dpi=150)
    print(f"  [benchmark] Boxplot saved as '{plot_path}'")
    plt.close(fig)

    print("\n  BENCHMARK COMPLETE.")


if __name__ == "__main__":
    main()
