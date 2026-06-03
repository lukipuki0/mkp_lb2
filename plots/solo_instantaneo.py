"""
plots/solo_instantaneo.py
-------------------------
Gráfico de convergencia que muestra ÚNICAMENTE el fitness instantáneo
evaluado en cada iteración (el "ruido" de exploración).

Genera: solo_instantaneo.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def grafico_solo_instantaneo(
    historial_inst_global: list,
    log_switches: list,
    colores_mh: dict,
    valor_optimo: float,
    output_dir: str,
) -> str:
    """
    Genera y guarda el gráfico que muestra solo el fitness instantáneo.

    Parameters
    ----------
    historial_inst_global : Lista del fitness instantáneo evaluado en cada iteración.
    log_switches          : Lista de SwitchLog del orquestador.
    colores_mh            : Dict {nombre_mh: color_hex}.
    valor_optimo          : Valor óptimo conocido de la instancia (0 = desconocido).
    output_dir            : Carpeta donde se guarda el PNG.

    Returns
    -------
    str : Ruta absoluta del archivo generado.
    """
    if not historial_inst_global:
        return ""

    fig, ax = plt.subplots(figsize=(14, 7))

    offset = 0
    legend_patches = []
    seen = set()

    for sw in log_switches:
        n_seg = sw.n_iters
        if n_seg == 0:
            continue
            
        seg_inst = historial_inst_global[offset: offset + n_seg]
        xs       = range(offset, offset + len(seg_inst))
        col      = colores_mh.get(sw.mh_nombre, "gray")

        # Línea de ruido (instantáneo) - sólida
        ax.plot(xs, seg_inst, color=col, linewidth=2.5, alpha=0.9)
        
        # Separador vertical
        ax.axvline(x=offset, color=col, linestyle="--", linewidth=1.5, alpha=0.5)
        
        offset += n_seg

        if sw.mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=sw.mh_nombre))
            seen.add(sw.mh_nombre)

    if valor_optimo > 0:
        ax.axhline(y=valor_optimo, color="red", linestyle="--",
                   linewidth=2.0)
        legend_patches.append(
            mpatches.Patch(color="red", label=f"Optimum ({valor_optimo:.0f})")
        )

    ax.set_title("Hybrid DTW Pipeline - Instantaneous Fitness per Iteration", fontsize=20, fontweight="bold")
    ax.set_xlabel("Accumulated Iterations", fontsize=18)
    ax.set_ylabel("Current Iteration Fitness", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(handles=legend_patches, loc="lower right", fontsize=15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "solo_instantaneo.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] solo_instantaneo.png      -> '{path}'")
    return path
