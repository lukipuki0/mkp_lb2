"""
DTW/DDTW delta of the monitor over the whole run, colored by profile.

Generates: dtw_delta.png

Same plot as hybrid_mkp/plots/dtw_delta.py::grafico_dtw_delta (copied, not
imported); hybrid_mkp's MH turns are the profile turns here.

The delta is the distance to the ramp curve (D1) minus the distance to the
flat curve (D2):
  - Delta > 0  => the window looks more like a plateau => stagnation
  - Delta < 0  => the window is still improving        => active improvement
"""

import math
import os

from .. import config
from ._common import mpatches, plt, profile_turns


def delta_series(res: dict) -> list:
    """Monitor delta per generation, NaN where the window was not ready."""
    return [out["delta"] if out["ready"] else float("nan") for out in res["monitor"]]


def plot_dtw_delta(res: dict, output_dir) -> str:
    """Saves the delta plot and returns its path ("" when no window was ready)."""
    dtw_deltas_global = delta_series(res)
    log_switches = profile_turns(res["profile_series"])
    colores_mh = config.PROFILE_COLORS

    xs_validos = []
    ys_validos = []
    offset = 0
    for mh_nombre, n_seg in log_switches:
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
    for mh_nombre, n_seg in log_switches:
        if n_seg == 0:
            continue
        col = colores_mh.get(mh_nombre, "gray")

        # Points of the current segment
        seg_xs = [x for x in range(offset, offset + n_seg) if x in xs_validos]
        seg_ys = [ys_validos[xs_validos.index(x)] for x in seg_xs]

        if seg_xs:
            # Connect without gaps to the previous valid point
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

        if mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=mh_nombre))
            seen.add(mh_nombre)

    ax.set_title("DTW/DDTW Delta per Iteration  [+ = stagnation | - = active improvement]",
                 fontsize=20, fontweight="bold")
    ax.set_xlabel("Accumulated Iterations", fontsize=18)
    ax.set_ylabel("DTW Delta", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # Keep the legend inside the axes, anchored at the lower-right corner.
    ax.legend(handles=legend_patches, loc="lower right", fontsize=15,
              borderaxespad=0.25, frameon=True, framealpha=0.88)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "dtw_delta.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] dtw_delta.png             -> '{path}'")
    return path
