"""
HRES2-H2/plots/convergencia.py
-------------------------------
Gráfico de convergencia LCOE del pipeline HRES2-H2, coloreado por metaheurística.
NO dibuja línea de óptimo externo (LCOE óptimo es desconocido en HRES2).

Genera: fitness_convergence.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def grafico_convergencia_hres2(
    historial_global: list,
    log_switches: list,
    colores_mh: dict,
    output_dir: str,
    ylabel: str = "Best LCOE (CNY/kWh)",
) -> str:
    """
    Genera y guarda el gráfico de convergencia LCOE del pipeline HRES2-H2.
    No dibuja ninguna línea de referencia de óptimo conocido.

    Parameters
    ----------
    historial_global : Lista de LCOE acumulado iteración a iteración.
    log_switches     : Lista de SwitchLog del orquestador.
    colores_mh       : Dict {nombre_mh: color_hex}.
    output_dir       : Carpeta donde se guarda el PNG.
    ylabel           : Etiqueta del eje Y.

    Returns
    -------
    str : Ruta absoluta del archivo generado.
    """
    if not historial_global:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))

    offset = 0
    legend_patches = []
    seen = set()

    dibujar_lineas = len(log_switches) < 50

    for sw in log_switches:
        n_seg = sw.n_iters
        if n_seg == 0:
            continue
        seg = historial_global[offset: offset + n_seg]
        xs  = range(offset, offset + len(seg))
        col = colores_mh.get(sw.mh_nombre, "gray")

        ax.plot(xs, seg, color=col, linewidth=3.0, alpha=0.85)
        if dibujar_lineas:
            ax.axvline(x=offset, color=col, linestyle="--", linewidth=1.5, alpha=0.5)
        offset += n_seg

        if sw.mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=sw.mh_nombre))
            seen.add(sw.mh_nombre)

    ax.set_title("HRES2-H2 Hybrid DTW Pipeline - LCOE Convergence",
                 fontsize=20, fontweight="bold")
    ax.set_xlabel("Accumulated Iterations", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "fitness_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] fitness_convergence.png   -> '{path}'")
    return path
