"""
plots/instantaneo.py
--------------------
Gráfico de convergencia que superpone el fitness instantáneo (ruidoso)
con el mejor fitness histórico (escalera).

Genera: convergencia_instantanea.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def grafico_instantaneo(
    historial_global: list,
    historial_inst_global: list,
    log_switches: list,
    colores_mh: dict,
    valor_optimo: float,
    output_dir: str,
) -> str:
    """
    Genera y guarda el gráfico superponiendo fitness histórico e instantáneo.

    Parameters
    ----------
    historial_global      : Lista de mejor fitness acumulado iteración a iteración.
    historial_inst_global : Lista del fitness instantáneo evaluado en cada iteración.
    log_switches          : Lista de SwitchLog del orquestador.
    colores_mh            : Dict {nombre_mh: color_hex}.
    valor_optimo          : Valor óptimo conocido de la instancia (0 = desconocido).
    output_dir            : Carpeta donde se guarda el PNG.

    Returns
    -------
    str : Ruta absoluta del archivo generado.
    """
    if not historial_global or not historial_inst_global:
        return ""

    fig, ax = plt.subplots(figsize=(14, 7))

    offset = 0
    legend_patches = []
    seen = set()

    for sw in log_switches:
        n_seg = sw.n_iters
        if n_seg == 0:
            continue
            
        seg_hist = historial_global[offset: offset + n_seg]
        seg_inst = historial_inst_global[offset: offset + n_seg]
        xs       = range(offset, offset + len(seg_hist))
        col      = colores_mh.get(sw.mh_nombre, "gray")

        # Línea de ruido (instantáneo) - semitransparente
        ax.plot(xs, seg_inst, color=col, linewidth=0.8, alpha=0.35)
        
        # Línea sólida (mejor histórico)
        ax.plot(xs, seg_hist, color=col, linewidth=1.8, alpha=0.95)
        
        # Separador vertical
        ax.axvline(x=offset, color=col, linestyle="--", linewidth=0.8, alpha=0.4)
        
        offset += n_seg

        if sw.mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=sw.mh_nombre))
            seen.add(sw.mh_nombre)

    if valor_optimo > 0:
        ax.axhline(y=valor_optimo, color="red", linestyle="--",
                   linewidth=1.4)
        legend_patches.append(
            mpatches.Patch(color="red", label=f"Optimo ({valor_optimo:.0f})")
        )

    ax.set_title("Pipeline Hibrido DTW - Exploración vs Explotación", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteracion acumulada", fontsize=11)
    ax.set_ylabel("Fitness (Sólido=Mejor Histórico | Transparente=Instantáneo)", fontsize=11)
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "convergencia_instantanea.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] convergencia_instantanea.png -> '{path}'")
    return path
