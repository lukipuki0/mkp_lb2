"""
plots/switches_gantt.py
-----------------------
Diagrama de Gantt que muestra el turno de ejecución de cada MH en el tiempo.

Genera: switches_gantt.png

Cada barra horizontal representa un turno de ejecución de una metaheurística.
La posición en el eje X es el tiempo real (segundos) y la barra está coloreada
según el algoritmo. Es útil para ver si el pipeline está rotando de forma
equilibrada o si una MH domina el tiempo total.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def grafico_switches(
    log_switches: list,
    colores_mh: dict,
    output_dir: str,
) -> str:
    """
    Genera y guarda el diagrama de Gantt de switches del pipeline.

    Parameters
    ----------
    log_switches : Lista de SwitchLog del orquestador.
    colores_mh   : Dict {nombre_mh: color_hex}.
    output_dir   : Carpeta donde se guarda el PNG.

    Returns
    -------
    str : Ruta absoluta del archivo generado.
    """
    if not log_switches:
        return ""

    fig, ax = plt.subplots(figsize=(14, 4))

    POOL_POBB = {"GA", "PSO", "GWO"}
    legend_patches = []
    seen = set()
    yticks = []
    ylabels = []

    for i, sw in enumerate(log_switches):
        col   = colores_mh.get(sw.mh_nombre, "gray")
        y_pos = 1 if sw.mh_nombre in POOL_POBB else 0   # 2 filas: poblacional / trayectoria
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
        # Etiqueta con el nombre dentro de la barra si cabe
        if duracion > 1.5:
            ax.text(
                sw.t_inicio + duracion / 2,
                y_pos,
                sw.mh_nombre,
                ha="center", va="center",
                fontsize=7, color="white", fontweight="bold",
            )

        if sw.mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=sw.mh_nombre))
            seen.add(sw.mh_nombre)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Trayectoria (SA/TS)", "Poblacional (GA/PSO/GWO)"], fontsize=10)
    ax.set_xlabel("Tiempo real (segundos)", fontsize=11)
    ax.set_title("Pipeline Hibrido DTW - Diagrama de Turnos por MH", fontsize=13, fontweight="bold")
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "switches_gantt.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] switches_gantt.png        -> '{path}'")
    return path
