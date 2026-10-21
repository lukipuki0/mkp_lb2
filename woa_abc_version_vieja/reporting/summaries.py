"""Resúmenes descriptivos por problema y variante."""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any

from .io import write_csv


def save_summary(directory: Path, rows: list[dict[str, Any]], title: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "resultados.csv", rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for variant, variant_rows in grouped.items():
        values = [float(row["best_cost"]) for row in variant_rows]
        summary_rows.append({
            "variant": variant,
            "runs": len(values),
            "best": min(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "worst": max(values),
        })
    write_csv(directory / "resumen.csv", summary_rows)
    lines = [
        f"# {title}",
        "",
        "| Variante | Runs | Mejor | Media | Mediana | Desv. | Peor |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['runs']} | {row['best']:.12g} | "
            f"{row['mean']:.12g} | {row['median']:.12g} | "
            f"{row['std']:.12g} | {row['worst']:.12g} |"
        )
    lines.extend([
        "",
        "Con dos o más seeds, los archivos `analisis_estadistico_*` contienen las pruebas inferenciales pareadas.",
    ])
    (directory / "resumen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["save_summary"]
