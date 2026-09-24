"""Genera o regenera la estadística de una campaña ya ejecutada.

Uso::

    python3 -m new_woa_abc.runners.analyze_results \
        new_woa_abc/resultados/cec/run_D10_ddtw_...
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

from new_woa_abc.domains.mkp import instance_group_name
from new_woa_abc.reporting import (
    save_global_statistical_summary,
    save_mkp_group_summary,
    save_paired_statistical_analysis,
)


def _read_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    path = experiment_dir / "todos_los_runs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no existe {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} no contiene resultados")
    return rows


def _groups(
    experiment_dir: Path,
    rows: Sequence[dict[str, Any]],
) -> tuple[str, list[tuple[str, Path, list[dict[str, Any]]]], str, str, bool]:
    domain = str(rows[0].get("domain", "")).lower()
    if any(str(row.get("domain", "")).lower() != domain for row in rows):
        raise ValueError("una campaña no puede mezclar dominios")

    grouped: dict[tuple, list[dict[str, Any]]] = {}
    if domain == "cec":
        for row in rows:
            key = (int(row["function_id"]),)
            grouped.setdefault(key, []).append(row)
        units = []
        for (function_id,), unit_rows in sorted(grouped.items()):
            candidates = sorted(experiment_dir.glob(f"CEC_{function_id:02d}_*"))
            if len(candidates) != 1:
                raise ValueError(f"no se encontró una carpeta única para CEC F{function_id}")
            units.append((f"CEC F{function_id:02d}", candidates[0], unit_rows))
        return domain, units, "absolute_error", "Error absoluto respecto al óptimo", True

    if domain == "hres2":
        problem = str(rows[0]["problem"])
        return domain, [
            (problem, experiment_dir / problem, list(rows))
        ], "best_cost", "LCOE / costo objetivo", True

    if domain == "mkp":
        for row in rows:
            key = (str(row["family"]), int(row["instance"]), int(row["items"]))
            grouped.setdefault(key, []).append(row)
        units = [
            (
                str(unit_rows[0]["problem"]),
                experiment_dir / instance_group_name(item_count) / family / f"inst_{instance:02d}",
                unit_rows,
            )
            for (family, instance, item_count), unit_rows in sorted(grouped.items())
        ]
        return domain, units, "best_profit", "Beneficio", False

    raise ValueError(f"dominio no reconocido en todos_los_runs.csv: {domain!r}")


def analyze(
    experiment_dir: Path,
    reference_variant: str = "M0_no_dtw",
    alpha: float = 0.05,
) -> Path:
    experiment_dir = experiment_dir.expanduser().resolve()
    rows = _read_rows(experiment_dir)
    domain, units, metric, metric_label, minimize = _groups(experiment_dir, rows)
    analyses: list[tuple[str, dict[str, Any]]] = []
    mkp_rows_by_size: dict[int, list[dict[str, Any]]] = {}
    mkp_analyses_by_size: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for problem, directory, unit_rows in units:
        title = problem if domain != "mkp" else f"MKP {problem}"
        analysis = save_paired_statistical_analysis(
            directory,
            unit_rows,
            metric=metric,
            metric_label=metric_label,
            title=title,
            reference_variant=reference_variant,
            minimize=minimize,
            alpha=alpha,
        )
        analyses.append((problem, analysis))
        if domain == "mkp":
            item_count = int(unit_rows[0]["items"])
            mkp_rows_by_size.setdefault(item_count, []).extend(unit_rows)
            mkp_analyses_by_size.setdefault(item_count, []).append((problem, analysis))

    if domain == "mkp":
        variant_order = {
            variant: index
            for index, variant in enumerate(dict.fromkeys(str(row["variant"]) for row in rows))
        }
        for item_count, group_rows in sorted(mkp_rows_by_size.items()):
            group_rows.sort(key=lambda row: (
                str(row["family"]),
                int(row["instance"]),
                variant_order[str(row["variant"])],
                int(row["run"]),
            ))
            save_mkp_group_summary(
                experiment_dir / instance_group_name(item_count),
                group_rows,
                sorted(mkp_analyses_by_size[item_count], key=lambda item: item[0]),
                item_count,
            )

    global_title = {
        "cec": "CEC2022",
        "hres2": "HRES2-H2/WPEB",
        "mkp": "MKP Chu--Beasley",
    }[domain]
    save_global_statistical_summary(
        experiment_dir,
        analyses,
        title=global_title,
    )
    return experiment_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("experiment_dir", type=Path)
    result.add_argument("--reference-variant", default="M0_no_dtw")
    result.add_argument("--alpha", type=float, default=0.05)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    output = analyze(args.experiment_dir, args.reference_variant, args.alpha)
    print(f"Análisis estadístico guardado en: {output}")


if __name__ == "__main__":
    main()
