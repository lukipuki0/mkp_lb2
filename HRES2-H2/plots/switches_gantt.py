"""
HRES2-H2/plots/switches_gantt.py
---------------------------------
Diagrama de Gantt que muestra el turno de ejecución de cada MH — módulo HRES2-H2.

Genera: switches_gantt.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def grafico_switches(
    log_switches: list,
    colores_mh: dict,
    output_dir: str,
) -> str:
    if not log_switches:
        return ""

    fig, ax = plt.subplots(figsize=(14, 4))

    legend_patches = []
    seen = set()

    for sw in log_switches:
        col      = colores_mh.get(sw.mh_nombre, "gray")
        y_pos    = 1 if sw.tipo == "poblacional" else 0
        duracion = sw.t_fin - sw.t_inicio

        ax.barh(
            y=y_pos,
            width=duracion,
            left=sw.t_inicio,
            color=col,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.6,
        )
        if duracion > 1.5:
            ax.text(
                sw.t_inicio + duracion / 2,
                y_pos,
                sw.mh_nombre,
                ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
            )

        if sw.mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=sw.mh_nombre))
            seen.add(sw.mh_nombre)

    pob_usadas = sorted({sw.mh_nombre for sw in log_switches if sw.tipo == "poblacional"})
    tra_usadas = sorted({sw.mh_nombre for sw in log_switches if sw.tipo == "trayectoria"})
    lbl_pob = f"Population-based ({'/'.join(pob_usadas)})" if pob_usadas else "Population-based"
    lbl_tra = f"Trajectory-based ({'/'.join(tra_usadas)})" if tra_usadas else "Trajectory-based"

    if tra_usadas:
        ax.set_yticks([0, 1])
        ax.set_yticklabels([lbl_tra, lbl_pob], fontsize=12)
    else:
        ax.set_yticks([1])
        ax.set_yticklabels([lbl_pob], fontsize=12)
    ax.set_xlabel("Real Time (seconds)", fontsize=14)
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.set_title("HRES2-H2 Hybrid DTW Pipeline - MH Execution Turns Gantt Chart",
                 fontsize=16, fontweight="bold")
    ax.legend(handles=legend_patches, loc="upper right", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "switches_gantt.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] switches_gantt.png        -> '{path}'")
    return path
