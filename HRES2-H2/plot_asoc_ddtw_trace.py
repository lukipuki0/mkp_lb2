"""Render DDTW-labeled figures from saved historial_dtw.csv; never run a solver.

Usage: py -3.11 -B HRES2-H2/plot_asoc_ddtw_trace.py --source BEST_RUN_DIRECTORY
Creates only ddtw_convergence.png and ddtw_delta.png; refuses existing targets.
CSV iteration 1 maps to x=0, as in the original exporters. Blank deltas remain
unavailable. Delta segments connect to the previous valid point, matching the
legacy exporter; no intermediate values are synthesized.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Same population colors as continuous_benchmark/orchestrator.py, without importing solvers.
COLORS = {"PSO": "#2196F3", "GWO": "#9C27B0", "WOA": "#E040FB", "EHO": "#00BCD4", "ACO": "#8D6E63"}
METHODS = set(COLORS) | {"ILS", "SA", "TS", "VNS"}
OUTPUTS = ("ddtw_convergence.png", "ddtw_delta.png")


def render(source):
    if any((source / name).exists() or (source / name).is_symlink() for name in OUTPUTS):
        raise FileExistsError("refusing to overwrite existing DDTW figures")
    with (source / "historial_dtw.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    segments = []
    for index, row in enumerate(rows):
        epoch, iteration = int(row["epoch"]), int(row["iteracion"])
        method = row["mh"]
        value = float(row["lcoe_best"])
        delta = None if row["dtw_delta"] == "" else float(row["dtw_delta"])
        if (iteration != index + 1 or method not in METHODS or not math.isfinite(value)
                or (delta is not None and not math.isfinite(delta))):
            raise ValueError
        if not segments or epoch != segments[-1]["epoch"]:
            if epoch != len(segments) + 1:
                raise ValueError
            segments.append({"epoch": epoch, "method": method, "points": []})
        if segments[-1]["method"] != method:
            raise ValueError
        segments[-1]["points"].append((index, value, delta))
    if not rows or not any(point[2] is not None for seg in segments for point in seg["points"]):
        raise ValueError

    # Build and validate both figures before writing either target.
    figures = []
    try:
        for is_delta in (False, True):
            fig, ax = plt.subplots(figsize=(14, 5 if is_delta else 6))
            figures.append(fig)
            previous = None
            legends, seen = [], set()
            for segment in segments:
                method, points = segment["method"], segment["points"]
                color = COLORS.get(method, "gray")
                values = [(x, delta if is_delta else value) for x, value, delta in points
                          if not is_delta or delta is not None]
                if values:
                    # Match legacy cross-segment delta connection using real saved endpoints.
                    plotted = ([previous] if is_delta and previous is not None else []) + values
                    ax.plot([p[0] for p in plotted], [p[1] for p in plotted], color=color,
                            linewidth=2.5 if is_delta else 3.0, alpha=0.85)
                    previous = values[-1]
                if len(segments) < 50:
                    ax.axvline(points[0][0], color=color, linestyle="--", linewidth=1.5, alpha=0.5)
                if method not in seen:
                    legends.append(Patch(color=color, label=method))
                    seen.add(method)
            ax.set_title("HRES2-H2 Hybrid DDTW - " + ("Delta per Iteration [+ = stagnation | - = active improvement]" if is_delta else "LCOE Convergence"), fontsize=16, fontweight="bold")
            ax.set_xlabel("Accumulated Iterations (zero-based)", fontsize=14)
            ax.set_ylabel("DDTW Delta" if is_delta else "Best LCOE (CNY/kWh)", fontsize=14)
            ax.legend(handles=legends, loc="lower right" if is_delta else "upper right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
        for figure, name in zip(figures, OUTPUTS):
            # Exclusive creation protects originals and refuses concurrent overwrites.
            with (source / name).open("xb") as handle:
                figure.savefig(handle, format="png", dpi=150)
    finally:
        for figure in figures:
            plt.close(figure)
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing saved DDTW historial_dtw.csv; no optimization.")
    args = parser.parse_args(argv)
    try:
        count = render(args.source)
    except (ValueError, KeyError, TypeError):
        print("error: invalid or insufficient saved telemetry", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Rendered ddtw_convergence.png and ddtw_delta.png from {count} saved iterations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
