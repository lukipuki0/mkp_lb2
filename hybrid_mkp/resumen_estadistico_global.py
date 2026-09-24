"""Resumen estadístico global válido para campañas MKP.

No compara los valores objetivos crudos de instancias diferentes. Cada fila
resume los contrastes del híbrido contra los algoritmos base dentro de la
misma instancia y aplica Holm a esa familia de comparaciones.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def _holm(pvalues: list[float]) -> list[float]:
    """Corrección step-down de Holm, conservando el orden de entrada."""
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=pvalues.__getitem__)
    adjusted = [1.0] * n
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (n - rank) * float(pvalues[original_index]))
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def _instance_dir(batch_dir: Path, instance_name: str) -> Path:
    match = re.fullmatch(r"(.+)_inst(\d+)", instance_name)
    if match is None:
        raise ValueError(f"Nombre de instancia MKP no reconocido: {instance_name}")
    family, index = match.groups()
    return batch_dir / family / f"inst_{int(index):02d}"


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Falta la columna o el valor '{key}'")
    return float(value)


def _label(wins: int, ties: int, losses: int) -> str:
    counts = f"+{wins}/={ties}/-{losses}"
    if wins and losses:
        return f"Mixta significativa ({counts})"
    if wins:
        return f"Favorable significativa ({counts})"
    if losses:
        return f"Desfavorable significativa ({counts})"
    return f"Sin diferencia significativa ({counts})"


def generar_resumen_estadistico_global(
    batch_dir: Path | str,
    resumen_global: list[dict[str, object]],
    alpha: float = 0.05,
) -> tuple[Path, Path]:
    """Genera CSV/Markdown sin mezclar objetivos de instancias distintas."""
    batch_dir = Path(batch_dir)
    output_rows: list[dict[str, object]] = []

    for summary in resumen_global:
        instance_name = str(summary["nombre"])
        analysis_path = _instance_dir(batch_dir, instance_name) / "analisis_estadistico_pvalues.csv"
        if not analysis_path.exists():
            raise FileNotFoundError(f"Falta el análisis por instancia: {analysis_path}")

        with analysis_path.open(newline="", encoding="utf-8") as f:
            analysis_rows = list(csv.DictReader(f))
        hybrid_rows = [
            row for row in analysis_rows
            if row.get("algoritmo", "").startswith("Hybrid")
        ]
        if len(hybrid_rows) != 1:
            raise ValueError(f"Se esperaba un único híbrido en {analysis_path}")

        hybrid_mean = _float(hybrid_rows[0], "media")
        comparators = [row for row in analysis_rows if row not in hybrid_rows]
        p_key = (
            "wilcoxon_pvalue_raw"
            if comparators and "wilcoxon_pvalue_raw" in comparators[0]
            else "wilcoxon_pvalue"
        )
        raw_pvalues = [_float(row, p_key) for row in comparators]
        adjusted_pvalues = _holm(raw_pvalues)

        wins = ties = losses = 0
        for comparator, adjusted in zip(comparators, adjusted_pvalues):
            comparator_mean = _float(comparator, "media")
            if adjusted >= alpha or comparator_mean == hybrid_mean:
                ties += 1
            elif hybrid_mean > comparator_mean:  # MKP es maximización
                wins += 1
            else:
                losses += 1

        output_rows.append({
            "instancia": instance_name,
            "n_runs": int(summary.get("n_runs", len(summary.get("valores_runs", [])))),
            "media_hibrido": float(summary["media"]),
            "mejor_hibrido": float(summary["mejor"]),
            "valor_optimo_bks": float(summary["valor_optimo"]),
            "gap_medio_pct": summary.get("gap_medio", ""),
            "gap_mejor_pct": summary.get("gap_mejor", ""),
            "comparadores": len(comparators),
            "victorias_significativas": wins,
            "similares": ties,
            "derrotas_significativas": losses,
            "significancia_estadistica": _label(wins, ties, losses),
        })

    fields = [
        "instancia", "n_runs", "media_hibrido", "mejor_hibrido",
        "valor_optimo_bks", "gap_medio_pct", "gap_mejor_pct", "comparadores",
        "victorias_significativas", "similares", "derrotas_significativas",
        "significancia_estadistica",
    ]
    csv_path = batch_dir / "analisis_estadistico_global.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    total_wins = sum(int(row["victorias_significativas"]) for row in output_rows)
    total_ties = sum(int(row["similares"]) for row in output_rows)
    total_losses = sum(int(row["derrotas_significativas"]) for row in output_rows)

    md_path = batch_dir / "analisis_estadistico_global.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Análisis estadístico global válido — MKP\n\n")
        f.write(
            "> No se comparan valores objetivos crudos entre instancias MKP, "
            "porque tienen escalas y BKS diferentes. Cada fila resume pruebas "
            "Wilcoxon híbrido–algoritmo base dentro de la misma instancia.\n\n"
        )
        f.write(f"- **Corrección:** Holm dentro de cada instancia ($\\alpha={alpha}$)\n")
        f.write(
            f"- **Total híbrido:** +{total_wins} / ={total_ties} / -{total_losses} "
            "(victorias/similares/derrotas significativas)\n\n"
        )
        f.write(
            "| Instancia | Media híbrido | Mejor | BKS | Gap medio (%) | "
            "Gap mejor (%) | + | = | - | Significancia estadística |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in output_rows:
            f.write(
                f"| `{row['instancia']}` | {float(row['media_hibrido']):.3f} | "
                f"{float(row['mejor_hibrido']):.3f} | {float(row['valor_optimo_bks']):.3f} | "
                f"{float(row['gap_medio_pct']):.3f} | {float(row['gap_mejor_pct']):.3f} | "
                f"{row['victorias_significativas']} | {row['similares']} | "
                f"{row['derrotas_significativas']} | {row['significancia_estadistica']} |\n"
            )

    return csv_path, md_path


def cargar_resumen_batch(batch_dir: Path) -> list[dict[str, object]]:
    path = batch_dir / "resumen_batch.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = []
    for row in rows:
        instance_dir = _instance_dir(batch_dir, row["instancia"])
        with (instance_dir / "runs_resultados.csv").open(newline="", encoding="utf-8") as f:
            n_runs = sum(1 for _ in csv.DictReader(f))
        result.append({
            "nombre": row["instancia"],
            "n_runs": n_runs,
            "media": float(row["media"]),
            "mejor": float(row["mejor"]),
            "valor_optimo": float(row["valor_optimo"]),
            "gap_medio": float(row["gap_medio_pct"]),
            "gap_mejor": float(row["gap_mejor_pct"]),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for batch_dir in args.batch_dirs:
        csv_path, md_path = generar_resumen_estadistico_global(
            batch_dir, cargar_resumen_batch(batch_dir)
        )
        print(f"[OK] {csv_path}")
        print(f"[OK] {md_path}")


if __name__ == "__main__":
    main()
