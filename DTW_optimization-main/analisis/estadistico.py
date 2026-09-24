"""
Statistical comparison of DTW versions vs V-Exploración baseline.
Uses Wilcoxon signed-rank test (paired by seed) with Bonferroni correction.

Usage:
    python -m analisis.estadistico
    python -m analisis.estadistico --instancia instances/mknapcb1.txt
    python -m analisis.estadistico --instancia instances/mknapcb1.txt --indice 0
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from mkp_common.stats import compare_versions, format_table, format_math_table


try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False


BASE = Path(__file__).resolve().parent.parent


def find_latest(directory, subdir: str = None):
    """
    Return the most recent ``comparacion_mhs_*`` directory under *directory*.

    If *subdir* is given (e.g. ``mknapcb1_0``), looks only inside that
    instance subdirectory. Otherwise scans all instance subdirectories.
    """
    if subdir:
        target = Path(directory) / subdir
        dirs = sorted(target.glob("comparacion_mhs_*")) if target.is_dir() else []
    else:
        # New structure: results/{version}/todos/{instance}/comparacion_mhs_*
        dirs = sorted(Path(directory).glob("*/comparacion_mhs_*"))
        if not dirs:
            # Old structure: results/{version}/todos/comparacion_mhs_*
            dirs = sorted(Path(directory).glob("comparacion_mhs_*"))
    return str(dirs[-1]) if dirs else None


def _ensure_scipy() -> bool:
    """Verify scipy is installed; print a friendly message otherwise."""
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        print("scipy is not installed. Please run: pip install scipy")
        return False


def _plot_box(results: dict, output_path: Path, instance_label: str = "",
              optimo: Optional[float] = None) -> None:
    """Optional boxplot comparing all versions (requires matplotlib)."""
    if not MATPLOTLIB_AVAILABLE:
        return

    mhs = results.get("mhs", {})
    if not mhs:
        return

    version_names = []
    for mh_entry in mhs.values():
        version_names.extend(mh_entry.get("versions", {}).keys())
    version_names = list(dict.fromkeys(version_names))

    n_mhs = len(mhs)
    n_versions = len(version_names) + 1  # +1 for V-Exploración (baseline)
    fig_width = max(8, 2.5 * n_versions)
    fig, axes = plt.subplots(1, n_mhs, figsize=(fig_width, 5), sharey=False)
    if n_mhs == 1:
        axes = [axes]

    for ax, (mh, mh_entry) in zip(axes, mhs.items()):
        data = [mh_entry.get("baseline_fitness", [])]
        labels = ["Exploration-only"]

        for v_name in version_names:
            v = mh_entry.get("versions", {}).get(v_name)
            if v is None:
                data.append([])
            else:
                data.append(v.get("version_fitness", []))
            labels.append(v_name)

        ax.boxplot([d for d in data if d], tick_labels=[labels[i] for i, d in enumerate(data) if d])
        ax.set_title(mh)
        ax.set_ylabel("Fitness")
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")

        if optimo and optimo > 0:
            ax.axhline(
                optimo,
                color="#2ecc71",
                linestyle="--",
                linewidth=2,
            )

    if optimo and optimo > 0:
        opt_part = f" (opt={optimo:.0f})"
    else:
        opt_part = ""
    suptitle = f"Fitness distribution across MH variants — {instance_label}{opt_part}"
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Boxplot saved to: {output_path}")
    # also save PNG for quick preview
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Boxplot saved to: {png_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of DTW versions vs V-Exploración baseline"
    )
    parser.add_argument(
        "--instancia",
        default=None,
        help="Filtrar por instancia (ej: instances/mknapcb1.txt). "
             "Si no se especifica, usa los resultados más recientes de cualquier instancia.",
    )
    parser.add_argument(
        "--indice",
        type=int,
        default=0,
        help="Índice de la instancia (default: 0). Solo se usa con --instancia.",
    )
    parser.add_argument(
        "--one-sided",
        action="store_true",
        default=False,
        help="Usar test one-sided (version > baseline) en vez de two-sided (default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not _ensure_scipy():
        return 1

    # Build instance subdirectory filter
    subdir = None
    instance_label = "?"
    instance_dir = "unknown"

    if args.instancia:
        inst_name = Path(args.instancia).stem
        subdir = f"{inst_name}_{args.indice}"
        instance_label = f"{inst_name}[{args.indice}]"
        instance_dir = subdir
        print(f"  Instancia seleccionada: {instance_label}")
        print(f"  Buscando en: results/*/todos/{subdir}/")
        print()

    baseline = find_latest(BASE / "results" / "vanilla_exploracion" / "todos", subdir=subdir)
    versions = {
        "Exploitation-only": find_latest(BASE / "results" / "vanilla_explotacion" / "todos", subdir=subdir),
        "Binary-Simple": find_latest(BASE / "results" / "binary_simple" / "todos", subdir=subdir),
        "Binary-Complex": find_latest(BASE / "results" / "binary_complex" / "todos", subdir=subdir),
        "Continuous-Simple": find_latest(BASE / "results" / "continuous_simple" / "todos", subdir=subdir),
        "Continuous-Complex": find_latest(BASE / "results" / "continuous_complex" / "todos", subdir=subdir),
    }

    # Drop versions whose directories are missing.
    versions = {k: v for k, v in versions.items() if v is not None}
    if not versions:
        print("No result directories found. Run the experiments first.")
        return 1

    if baseline is None:
        print("Baseline vanilla_exploracion results not found. Run vanilla_exploracion first.")
        return 1

    mhs = ["PSO", "GA", "GWO", "DE"]

    alternative = "greater" if args.one_sided else "two-sided"
    results = compare_versions(baseline, versions, mhs, alpha=0.05, alternative=alternative)

    # Extract instance info from baseline JSON if not explicitly set
    if instance_label == "?":
        inst_name = "?"
        inst_idx = "?"
        try:
            first_mh = mhs[0]
            candidates = sorted(Path(baseline).glob(f"{first_mh}_*.json"))
            if candidates:
                with open(candidates[0], encoding="utf-8") as f:
                    info = json.load(f).get("info", {})
                    inst_path = info.get("instancia", "")
                    inst_idx = info.get("idx", "?")
                    inst_name = Path(inst_path).stem if inst_path else "?"
        except Exception:
            pass
        instance_label = f"{inst_name}[{inst_idx}]" if inst_name != "?" else "mknapcb4[0]"
        instance_dir = f"{inst_name}_{inst_idx}" if inst_name != "?" else "unknown"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE / "results" / "estadistico" / instance_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    alt_label = "(one-tailed >)" if args.one_sided else "(two-sided)"

    # Formatted table (console + file)
    title1 = f"DTW Adaptation vs Exploration-only — {instance_label} — Wilcoxon Signed-Rank {alt_label}"
    table = format_table(results, title=title1)
    print(table)

    # Math table (console + file)
    print()
    title2 = f"Numerical Results — {instance_label} — Exploration-only vs Variants"
    math_table = format_math_table(results, title=title2)
    print(math_table)

    # Save both to files
    table_path = output_dir / f"tabla_estadistica_{stamp}.txt"
    math_path = output_dir / f"tabla_matematica_{stamp}.txt"
    table_path.write_text(table + "\n", encoding="utf-8")
    math_path.write_text(math_table + "\n", encoding="utf-8")
    print(f"\nSaved: {table_path}")
    print(f"Saved: {math_path}")

    if MATPLOTLIB_AVAILABLE:
        plot_path = output_dir / f"comparacion_{stamp}.pdf"
        optimo = results.get("summary", {}).get("optimo_conocido")
        _plot_box(results, plot_path, instance_label=instance_label, optimo=optimo)

    return 0


if __name__ == "__main__":
    sys.exit(main())
