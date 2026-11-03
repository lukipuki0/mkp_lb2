"""
Shared pieces of the GWO-DE plots: profile segments, colors, style and save.

A segment (turn) is a run of consecutive generations with the same profile
(res["profile_series"]). parameters.png plots generations as 1..T; the
convergence and delta plots copy hybrid_mkp and use 0-based iterations.
"""

import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from .. import config  # noqa: E402

# Dashed change lines are drawn only below this many profile changes (as hybrid_mkp).
MAX_CHANGE_LINES = 50

Segment = Tuple[str, int, int]  # (profile, first generation index, end index exclusive)


def profile_color(name: str) -> str:
    return config.PROFILE_COLORS.get(name, "gray")


def profile_segments(profile_series: list) -> List[Segment]:
    segments: List[Segment] = []
    start = 0
    for g in range(1, len(profile_series) + 1):
        if g == len(profile_series) or profile_series[g] != profile_series[start]:
            segments.append((profile_series[start], start, g))
            start = g
    return segments


def profile_turns(profile_series: list) -> List[Tuple[str, int]]:
    """Profile turns as (profile, n_iters), the log_switches of hybrid_mkp."""
    return [(name, end - start) for name, start, end in profile_segments(profile_series)]


def profile_legend(names: list) -> list:
    """One patch per profile, in order of first appearance."""
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return [mpatches.Patch(color=profile_color(name), label=name) for name in seen]


def draw_change_lines(ax, segments: List[Segment]) -> None:
    """Dashed vertical line where every new profile starts (between generations)."""
    if len(segments) - 1 >= MAX_CHANGE_LINES:
        return
    for name, start, _ in segments[1:]:
        ax.axvline(x=start + 0.5, color=profile_color(name), linestyle="--",
                   linewidth=1.5, alpha=0.5)


def style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)


def add_legend(ax, handles: list) -> None:
    # Keep the legend inside the axes, anchored at the lower-right corner.
    ax.legend(handles=handles, loc="lower right", fontsize=15,
              borderaxespad=0.25, frameon=True, framealpha=0.88)


def save(fig, output_dir, filename: str) -> str:
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {filename} -> '{path}'")
    return path
