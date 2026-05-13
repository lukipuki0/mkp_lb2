"""
plots/dtw_delta.py
------------------
Gráfico del Delta DTW a lo largo de toda la ejecución del pipeline.

Genera: dtw_delta.png

El Delta DTW mide la diferencia entre la distancia a la curva rampa (D1)
y la distancia a la curva plana (D2):
  - Delta > 0  => el historial se parece más a una meseta => ESTANCAMIENTO
  - Delta < 0  => el historial sigue mejorando => EXPLOTACIÓN activa
"""

import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def grafico_dtw_delta(
    dtw_deltas_global: list,
    log_switches: list,
    colores_mh: dict,
    output_dir: str,
) -> str:
    """
    Genera y guarda el gráfico del Delta DTW.

    Parameters
    ----------
    dtw_deltas_global : Lista de deltas DTW (con NaN donde la ventana no estaba lista).
    log_switches      : Lista de SwitchLog del orquestador.
    colores_mh        : Dict {nombre_mh: color_hex}.
    output_dir        : Carpeta donde se guarda el PNG.

    Returns
    -------
    str : Ruta absoluta del archivo generado.
    """
    # Filtrar si no hay ningún dato real
    datos_reales = [d for d in dtw_deltas_global
                    if d != "" and not (isinstance(d, float) and math.isnan(d))]
    if not datos_reales:
        print("  [plot] dtw_delta.png  -> sin datos suficientes, omitido.")
        return ""

    fig, ax = plt.subplots(figsize=(14, 5))

    offset = 0
    legend_patches = []
    seen = set()

    for sw in log_switches:
        n_seg = sw.n_iters
        if n_seg == 0:
            continue
        seg_d = dtw_deltas_global[offset: offset + n_seg]
        xs_d  = range(offset, offset + len(seg_d))
        col   = colores_mh.get(sw.mh_nombre, "gray")

        ax.plot(xs_d, seg_d, color=col, linewidth=1.2, alpha=0.85)
        ax.axvline(x=offset, color=col, linestyle="--", linewidth=0.8, alpha=0.3)
        offset += n_seg

        if sw.mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=sw.mh_nombre))
            seen.add(sw.mh_nombre)

    # Línea en cero: separación exploración/explotación
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.6,
               label="Umbral (Delta=0)")
    legend_patches.append(mpatches.Patch(color="black", label="Umbral (Delta=0)"))

    ax.set_title("Delta DTW por Iteracion  [+ = estancamiento | - = mejora activa]",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteracion acumulada", fontsize=11)
    ax.set_ylabel("Delta DTW", fontsize=11)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "dtw_delta.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] dtw_delta.png             -> '{path}'")
    return path
