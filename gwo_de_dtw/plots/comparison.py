"""
Best-so-far convergence of the four policies of one run group, on one axes.

Generates: convergence_comparison.png and convergence_comparison.pdf

Each record is a run record (run.py's result.json or run_hybrid's dict). Every
line is its raw `best_series` (maximization) against generation 1..T; nothing is
normalized. The red dashed best-known line of fitness_convergence.png is drawn
only when every record has the same positive `known_optimum`.

With seed_labels=True (campaign.py, best run of each policy) every legend label
adds the record's seed and the title reads "Best run of each policy".
"""

import os
import re
from typing import List, Sequence

from ._common import plt

# Fixed color per policy (Okabe-Ito, colorblind safe; red is kept for the best known value).
POLICY_COLORS = {
    "binary_simple": "#0072B2",
    "binary_complex": "#E69F00",
    "continuous_simple": "#009E73",
    "continuous_complex": "#CC79A7",
    "vanilla_exploitation": "#56B4E9",
    "vanilla_exploration": "#000000",
}


def convergence_comparison_figure(records: Sequence[dict], seed_labels: bool = False):
    """Returns the comparison figure (caller closes it); one line per record, in order.

    seed_labels=True labels each line `<policy> (seed <seed>)`.
    """
    for i, res in enumerate(records):
        if not res.get("policy"):
            raise ValueError(f"record {i} has no policy")
        if seed_labels and res.get("seed") is None:
            raise ValueError(f"record {i} has no seed")

    fig, ax = plt.subplots(figsize=(14, 6))
    for res in records:
        series = res["best_series"]
        label = f"{res['policy']} (seed {res['seed']})" if seed_labels else res["policy"]
        ax.plot(range(1, len(series) + 1), series, label=label,
                color=POLICY_COLORS.get(res["policy"], "gray"), linewidth=2.5, alpha=0.9)

    optima = {res.get("known_optimum") for res in records}
    if len(optima) == 1:
        optimum = optima.pop()
        if optimum is not None and optimum > 0:
            ax.axhline(y=optimum, color="red", linestyle="--", linewidth=2.0,
                       label=f"Best known ({optimum:.0f})")

    counted = {1: "one policy", 4: "the four policies", 6: "the six policies"}.get(
        len(records), f"the {len(records)} policies")
    title = "Best run of each policy" if seed_labels else f"Convergence of {counted}"
    instances = {res.get("instance") for res in records}
    if len(instances) == 1 and None not in instances:
        instance_label = re.sub(r"\.txt(?=\[|$)", "", instances.pop(), flags=re.IGNORECASE)
        title += f" - {instance_label}"
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=18)
    ax.set_ylabel("Best-so-far fitness (maximization)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.legend(loc="lower right", fontsize=15, borderaxespad=0.25, frameon=True,
              framealpha=0.88)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_convergence_comparison(records: Sequence[dict], output_dir,
                                seed_labels: bool = False) -> List[str]:
    """Saves the comparison PNG and PDF in `output_dir`; returns both paths."""
    fig = convergence_comparison_figure(records, seed_labels=seed_labels)
    png = os.path.join(output_dir, "convergence_comparison.png")
    pdf = os.path.join(output_dir, "convergence_comparison.pdf")
    try:
        fig.savefig(png, dpi=150)
        fig.savefig(pdf)
    finally:
        plt.close(fig)
    print(f"  [plot] convergence_comparison.png -> '{png}'")
    return [png, pdf]
