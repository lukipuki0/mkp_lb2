"""Seed-labelled comparison and the best-run fitness trace."""

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICY_COLORS = {
    "binary_simple": "#0072B2",
    "binary_complex": "#E69F00",
    "continuous_simple": "#009E73",
    "continuous_complex": "#CC79A7",
}


def comparison_figure(records, seed_labels=False):
    """One best-so-far line per record. Seed labels read ``policy (seed N)``."""
    fig, ax = plt.subplots(figsize=(14, 6))
    for record in records:
        series = record["best_series"]
        if seed_labels:
            label = f"{record['policy']} (seed {record['seed']})"
        else:
            label = record["policy"]
        ax.plot(range(1, len(series) + 1), series, label=label,
                color=POLICY_COLORS.get(record["policy"], "gray"), linewidth=2.5, alpha=0.9)
    optima = {record.get("known_optimum") for record in records}
    if len(optima) == 1:
        optimum = optima.pop()
        if optimum is not None and optimum > 0:
            ax.axhline(optimum, color="red", linestyle="--", linewidth=2.0,
                       label=f"Best known ({optimum:.0f})")
    title = "Best run of each policy" if seed_labels else "Convergence of the four policies"
    instances = {record.get("instance") for record in records}
    if len(instances) == 1 and None not in instances:
        label = re.sub(r"\.txt(?=\[|$)", "", str(instances.pop()), flags=re.IGNORECASE)
        title += f" - {label}"
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=18)
    ax.set_ylabel("Best-so-far fitness (maximization)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.legend(loc="lower right", fontsize=15, frameon=True, framealpha=0.88)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_comparison(records, output_dir, seed_labels=True):
    fig = comparison_figure(records, seed_labels=seed_labels)
    png = os.path.join(output_dir, "convergence_comparison.png")
    pdf = os.path.join(output_dir, "convergence_comparison.pdf")
    try:
        fig.savefig(png, dpi=150)
        fig.savefig(pdf)
    finally:
        plt.close(fig)
    return [png, pdf]


def save_best_trace(record, output_dir):
    """Write result.json, trace.csv and fitness.png for one kept run."""
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "result.json")
    trace_path = os.path.join(output_dir, "trace.csv")
    figure_path = os.path.join(output_dir, "fitness.png")
    with open(trace_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "best_fitness", "profile"])
        profiles = record.get("profile_series") or []
        for generation, fitness in enumerate(record["best_series"], start=1):
            profile = profiles[generation - 1] if generation - 1 < len(profiles) else ""
            writer.writerow([generation, fitness, profile])
    fig, ax = plt.subplots(figsize=(10, 4))
    try:
        ax.plot(range(1, len(record["best_series"]) + 1), record["best_series"],
                color=POLICY_COLORS.get(record.get("policy"), "#0072B2"), linewidth=2.0)
        ax.set_title(f"{record.get('policy')} (seed {record.get('seed')})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best-so-far fitness")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
    finally:
        plt.close(fig)
    return [result_path, trace_path, figure_path]
