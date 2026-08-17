"""
HRES2-H2/plots/dtw_delta.py
----------------------------
Gráfico del Delta DTW a lo largo de toda la ejecución — módulo HRES2-H2.

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
    xs_validos = []
    ys_validos = []
    offset = 0
    for sw in log_switches:
        n_seg = sw.n_iters
        seg_d = dtw_deltas_global[offset: offset + n_seg]
        for idx, val in enumerate(seg_d):
            if val != "" and not (isinstance(val, float) and math.isnan(val)):
                xs_validos.append(offset + idx)
                ys_validos.append(val)
        offset += n_seg

    if not ys_validos:
        print("  [plot] dtw_delta.png  -> sin datos suficientes, omitido.")
        return ""

    fig, ax = plt.subplots(figsize=(14, 5))

    legend_patches = []
    seen = set()
    dibujar_lineas = len(log_switches) < 50

    offset = 0
    for sw in log_switches:
        n_seg = sw.n_iters
        if n_seg == 0:
            continue
        col = colores_mh.get(sw.mh_nombre, "gray")

        # Puntos del segmento actual
        seg_xs = [x for x in range(offset, offset + n_seg) if x in xs_validos]
        seg_ys = [ys_validos[xs_validos.index(x)] for x in seg_xs]

        if seg_xs:
            # Para conectar sin huecos con el punto válido anterior
            prev_indices = [i for i, x in enumerate(xs_validos) if x < seg_xs[0]]
            if prev_indices:
                last_prev_idx = prev_indices[-1]
                plot_xs = [xs_validos[last_prev_idx]] + seg_xs
                plot_ys = [ys_validos[last_prev_idx]] + seg_ys
            else:
                plot_xs = seg_xs
                plot_ys = seg_ys

            ax.plot(plot_xs, plot_ys, color=col, linewidth=2.5, alpha=0.85)

        if dibujar_lineas:
            ax.axvline(x=offset, color=col, linestyle="--", linewidth=1.5, alpha=0.5)

        offset += n_seg

        if sw.mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=sw.mh_nombre))
            seen.add(sw.mh_nombre)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=2.0, alpha=0.6,
               label="Threshold (Delta=0)")
    legend_patches.append(mpatches.Patch(color="black", label="Threshold (Delta=0)"))


    ax.set_title("HRES2-H2 DTW Delta per Iteration  [+ = stagnation | - = active improvement]",
                 fontsize=18, fontweight="bold")
    ax.set_xlabel("Accumulated Iterations", fontsize=18)
    ax.set_ylabel("DTW Delta", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(handles=legend_patches, loc="lower right", fontsize=15)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "dtw_delta.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] dtw_delta.png             -> '{path}'")
    return path
