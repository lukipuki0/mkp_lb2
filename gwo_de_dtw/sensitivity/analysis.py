"""Deterministic descriptive summaries. No inferential tests and no pooled instances."""

import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import CONFIGURATIONS, POLICIES

PLOT_DPI = 150
PARAMETERS = ("window", "band", "p_low", "p_high", "plateau_max", "patience")
_COLORS = ("#0072B2", "#E69F00", "#009E73")


def better_run(candidate, incumbent) -> bool:
    """Fitness is maximized. Equal fitness keeps the lower seed."""
    if incumbent is None:
        return True
    if candidate["best_fitness"] != incumbent["best_fitness"]:
        return candidate["best_fitness"] > incumbent["best_fitness"]
    return candidate["seed"] < incumbent["seed"]


def _known(value) -> bool:
    return value is not None and value != ""


def summarize(rows) -> dict:
    """Mean, median, best, worst and sample SD. SD and unknown gaps stay blank at n=1."""
    fitness = [float(row["best_fitness"]) for row in rows]
    best = None
    for row in rows:
        if better_run(row, best):
            best = row
    gaps = [row["gap_pct"] for row in rows]
    known = all(_known(gap) for gap in gaps)
    return {
        "best_fitness": max(fitness),
        "worst_fitness": min(fitness),
        "mean_fitness": statistics.mean(fitness),
        "median_fitness": statistics.median(fitness),
        "std_fitness": statistics.stdev(fitness) if len(fitness) > 1 else "",
        "best_seed": best["seed"],
        "best_gap_pct": best["gap_pct"] if known and _known(best["gap_pct"]) else "",
        "mean_gap_pct": statistics.mean(float(gap) for gap in gaps) if known else "",
        "mean_elapsed_s": statistics.mean(float(row["elapsed_s"]) for row in rows),
    }


def comparison_table(stat_rows) -> list:
    """Mean fitness minus the same instance/policy baseline. Missing baseline stays blank."""
    order = {spec.name: i for i, spec in enumerate(CONFIGURATIONS)}
    policy_order = {name: i for i, name in enumerate(POLICIES)}
    grouped = {}
    for row in stat_rows:
        grouped.setdefault((row["index"], row["policy"]), []).append(row)
    table = []
    for key in sorted(grouped, key=lambda item: (item[0], policy_order.get(item[1], 99))):
        rows = sorted(grouped[key], key=lambda row: order.get(row["configuration"], 99))
        base = next((row for row in rows if row["configuration"] == "baseline"), None)
        base_mean = None if base is None else row_mean(base)
        for row in rows:
            delta = "" if base_mean is None else row_mean(row) - base_mean
            table.append({
                "instance": row["instance"],
                "index": row["index"],
                "policy": row["policy"],
                "parameter": row["parameter"],
                "configuration": row["configuration"],
                "parameter_value": row["parameter_value"],
                "mean_fitness": row_mean(row),
                "baseline_mean_fitness": "" if base_mean is None else base_mean,
                "mean_fitness_delta_vs_baseline": delta,
            })
    return table


def row_mean(row) -> float:
    return float(row["mean_fitness"])


def sensitivity_figure(rows, policy: str):
    """One panel per varied parameter. The zero line is that instance's baseline."""
    selected = [row for row in rows if row.get("policy", policy) == policy]
    indices = sorted({row["index"] for row in selected})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(
        f"Descriptive OAT mean fitness delta vs baseline ({policy}). "
        "Baseline is zero. Not a global-robustness claim."
    )
    for ax, parameter in zip(axes.ravel(), PARAMETERS):
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(parameter)
        ax.set_ylabel("mean fitness delta vs baseline")
        for color_i, index in enumerate(indices):
            points = [
                row for row in selected
                if row["parameter"] == parameter and row["index"] == index
                and row["mean_fitness_delta_vs_baseline"] != ""
            ]
            points.sort(key=lambda row: float(row["parameter_value"]))
            if not points:
                continue
            ax.plot(
                [float(row["parameter_value"]) for row in points],
                [float(row["mean_fitness_delta_vs_baseline"]) for row in points],
                marker="o", color=_COLORS[color_i % len(_COLORS)],
                label=f"mknapcb1.txt[{index}]",
            )
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_sensitivity_plot(rows, policy: str, output_dir) -> str:
    """Write one PNG at the same DPI as the hybrid plots and always close this figure."""
    from pathlib import Path

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig = sensitivity_figure(rows, policy)
    path = output / f"sensitivity_{policy}.png"
    try:
        fig.savefig(path, dpi=PLOT_DPI)
    finally:
        plt.close(fig)
    return str(path)
