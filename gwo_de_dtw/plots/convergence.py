"""
Global fitness convergence of the run, colored by profile.

Generates: fitness_convergence.png

Same plot as hybrid_mkp/plots/convergencia.py::grafico_convergencia (copied,
not imported); hybrid_mkp's MH turns are the profile turns here.
"""

import os

from .. import config
from ._common import mpatches, plt, profile_turns


def plot_convergence(res: dict, output_dir) -> str:
    """Saves the convergence plot and returns its path ("" when there are no generations)."""
    historial_global = res["best_series"]
    log_switches = profile_turns(res["profile_series"])
    colores_mh = config.PROFILE_COLORS
    valor_optimo = res["known_optimum"]

    if not historial_global:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))

    offset = 0
    legend_patches = []
    seen = set()

    dibujar_lineas = len(log_switches) < 50

    for mh_nombre, n_seg in log_switches:
        if n_seg == 0:
            continue
        seg = historial_global[offset: offset + n_seg]
        col = colores_mh.get(mh_nombre, "gray")

        # Connect with the last point of the previous profile to draw one continuous line
        if offset > 0:
            xs_plot = list(range(offset - 1, offset + len(seg)))
            ys_plot = [historial_global[offset - 1]] + seg
        else:
            xs_plot = list(range(offset, offset + len(seg)))
            ys_plot = seg

        ax.plot(xs_plot, ys_plot, color=col, linewidth=3.0, alpha=0.85)
        if dibujar_lineas:
            ax.axvline(x=offset, color=col, linestyle="--", linewidth=1.5, alpha=0.5)
        offset += n_seg

        if mh_nombre not in seen:
            legend_patches.append(mpatches.Patch(color=col, label=mh_nombre))
            seen.add(mh_nombre)

    if valor_optimo > 0:
        ax.axhline(y=valor_optimo, color="red", linestyle="--",
                   linewidth=2.0)
        legend_patches.append(
            mpatches.Patch(color="red", label=f"Optimum ({valor_optimo:.0f})")
        )

    ax.set_title("Hybrid DTW/DDTW Pipeline - Fitness Convergence",
                 fontsize=20, fontweight="bold")
    ax.set_xlabel("Accumulated Iterations", fontsize=18)
    ax.set_ylabel("Best Fitness Value", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # Keep the legend inside the axes, anchored at the lower-right corner.
    ax.legend(handles=legend_patches, loc="lower right", fontsize=15,
              borderaxespad=0.25, frameon=True, framealpha=0.88)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "fitness_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] fitness_convergence.png   -> '{path}'")
    return path
