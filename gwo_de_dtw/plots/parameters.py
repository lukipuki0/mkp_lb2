"""
Parameters used in every generation: a (GWO), F and CR (DE).

Generates: parameters.png

One panel per parameter, drawn as a step line over a background shaded by
the profile of each generation (res["parameter_series"]). G3 is recorded in
the JSON but not plotted.
"""

from ._common import (add_legend, draw_change_lines, plt, profile_color, profile_legend,
                      profile_segments, save, style_axes)

FILENAME = "parameters.png"
PANELS = (("a", "a (GWO)"), ("F", "F (DE)"), ("CR", "CR (DE)"))


def plot_parameters(res: dict, output_dir) -> str:
    """Saves the parameter plot and returns its path ("" when there are no generations)."""
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
    axes[-1].set_xlabel("Generation", fontsize=18)
    axes[-1].set_xlim(0.5, total + 0.5)
    add_legend(axes[0], profile_legend(profiles))

    fig.suptitle("GWO-DE Hybrid - Parameters per Generation", fontsize=20, fontweight="bold")
    return save(fig, output_dir, FILENAME)
