"""
results.py
----------
Reporte de resultados y graficos de convergencia del SA para MKP.

Funciones:
  - imprimir_resumen()      : muestra tabla de resultados en consola.
  - graficar_convergencia() : genera y guarda el grafico de convergencia.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from sa_mkp.algorithm import SAResult


# -- Reporte en consola -------------------------------------------------------

def imprimir_resumen(resultado: SAResult) -> None:
    """Imprime un resumen completo de la ejecucion SA."""
    sep = "=" * 65

    print(f"\n{sep}")
    print("  RESULTADOS  SIMULATED ANNEALING  -  MKP")
    print(sep)

    # Tabla por epoch
    print(
        f"  {'Epoch':>6}  {'Mejor valor':>14}  "
        f"{'Iteraciones':>12}  {'Stag. fires':>12}"
    )
    print(
        f"  {'------':>6}  {'------------':>14}  "
        f"{'------------':>12}  {'------------':>12}"
    )
    for ep in resultado.epochs:
        print(
            f"  {ep.epoch_idx + 1:>6}  "
            f"{ep.mejor_valor:>14.1f}  "
            f"{ep.iteraciones:>12}  "
            f"{ep.stagnation_fires:>12}"
        )

    print(sep)
    print(f"  Mejor valor global       : {resultado.mejor_valor_global:.1f}")
    print(f"  Valor optimo conocido    : {resultado.valor_optimo:.1f}")

    gap = resultado.gap_pct
    if gap is not None:
        print(f"  Gap relativo             : {gap:.2f}%")
    else:
        print("  Gap relativo             : N/A (optimo no disponible en esta instancia)")

    print(sep)


# -- Grafico ------------------------------------------------------------------

def graficar_convergencia(
    resultado: SAResult,
    ruta: str = "convergencia_SA_MKP.png",
    dpi: int = 150,
    mostrar: bool = True,
) -> None:
    """Genera el grafico de convergencia y lo guarda en *ruta*.

    Parameters
    ----------
    resultado : SAResult
        Resultado devuelto por `ejecutar_sa`.
    ruta : str
        Ruta del archivo de imagen de salida.
    dpi : int
        Resolucion del grafico.
    mostrar : bool
        Si True, llama a plt.show() al final.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    for ep in resultado.epochs:
        ax.plot(ep.historial, alpha=0.65, linewidth=1.2,
                label=f"Epoch {ep.epoch_idx + 1}")

    # Linea de referencia: optimo conocido (solo si es distinto de 0)
    if resultado.valor_optimo != 0:
        ax.axhline(
            y=resultado.valor_optimo,
            color="red", linestyle="--", linewidth=1.4,
            label=f"Known Optimal ({resultado.valor_optimo:.0f})",
        )

    ax.set_title("SA Convergence - MKP", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Value (Epoch)", fontsize=11)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta, dpi=dpi)
    print(f"[results] Grafico guardado en '{ruta}'")

    if mostrar:
        plt.show()

    plt.close(fig)
