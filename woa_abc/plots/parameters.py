"""
Parameters used in every cycle: a (WOA) and rho (ABC).

Generates: parameters.png

One panel per parameter, drawn as a step line over a background shaded by
the profile of each cycle (res["parameter_series"]). G3 is recorded in
the JSON but not plotted. Copied from gwo_de_dtw/plots/parameters.py; only
the panels, the axis label and the title differ.
"""

from ._common import (add_legend, draw_change_lines, plt, profile_color, profile_legend,
                      profile_segments, save, style_axes)

FILENAME = "parameters.png"
PANELS = (("a", "a (WOA)"), ("rho", "rho (ABC)"))


def plot_parameters(res: dict, output_dir) -> str:
    """Saves the parameter plot and returns its path ("" when there are no cycles)."""
    profiles = res["profile_series"]
    if not profiles:
        return ""

    panels = PANELS
    total = len(profiles)
    xs = list(range(1, total + 1))
    segments = profile_segments(profiles)
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 7), sharex=True)
    for ax, (key, label) in zip(axes, panels):
        for name, start, end in segments:
            ax.axvspan(start + 0.5, end + 0.5, color=profile_color(name), alpha=0.18,
                       linewidth=0)
        ax.step(xs, res["parameter_series"][key], where="mid", color="black", linewidth=2.5)
        draw_change_lines(ax, segments)
        style_axes(ax, "", label)
        ax.margins(y=0.3)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Cycle", fontsize=18)
    axes[-1].set_xlim(0.5, total + 0.5)
    add_legend(axes[0], profile_legend(profiles))

    fig.suptitle("WOA-ABC Hybrid - Parameters per Cycle", fontsize=20, fontweight="bold")
    return save(fig, output_dir, FILENAME)
