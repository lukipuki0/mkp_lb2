"""Comparación global Hybrid DTW vs Hybrid DDTW en CEC2022.

La comparación se hace dentro de cada función, porque las funciones CEC2022
tienen escalas distintas. El reporte incluye el error normalizado respecto al
óptimo conocido y una decisión basada en Wilcoxon pareado con corrección de
Holm sobre las 12 funciones.

Uso desde la raíz del proyecto::

    python -m continuous_benchmark.analisis_global_dtw_ddtw

También se pueden indicar dos carpetas concretas::

    python -m continuous_benchmark.analisis_global_dtw_ddtw \
        --dtw-dir continuous_benchmark/resultados/run_dtw_... \
        --ddtw-dir continuous_benchmark/resultados/run_ddtw_...
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy import stats

from continuous_benchmark.analisis_estadistico import ajustar_pvalues_holm
from continuous_benchmark.funciones_cec2022 import get_test_functions


CEC_RE = re.compile(r"CEC_(\d+)_?(.*)")


def _cec_index(path: Path) -> int:
    match = CEC_RE.match(path.name)
    if match is None:
        raise ValueError(f"No se pudo identificar la función CEC: {path}")
    return int(match.group(1))


def _find_latest(root: Path, prefix: str) -> Path:
    candidates = sorted(
        (p for p in root.glob(f"{prefix}*" ) if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No se encontró una carpeta {prefix}* en {root}")
    return candidates[0]


def _load_values(function_dir: Path) -> list[float]:
    path = function_dir / "runs_resultados.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    values = [float(row["mejor_valor"]) for row in rows]
    if len(values) < 2:
        raise ValueError(f"Se requieren al menos 2 runs en {path}")
    return values


def _optima_by_index(run_dir: Path) -> dict[int, float]:
    summary_path = run_dir / "resumen_global.csv"
    if summary_path.exists():
        result = {}
        with summary_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                match = re.search(r"CEC\s*(\d+)", row["funcion"])
                if match:
                    result[int(match.group(1))] = float(row["valor_optimo"])
        if result:
            return result

    return {i + 1: float(func.optimum) for i, func in enumerate(get_test_functions(10))}


def _wilcoxon_p(dtw: list[float], ddtw: list[float]) -> float:
    if np.array_equal(dtw, ddtw):
        return 1.0
    try:
        return float(stats.wilcoxon(dtw, ddtw, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _estado_optimo(valor: float, optimo: float) -> str:
    if np.isclose(valor, optimo, rtol=1e-9, atol=1e-8):
        return "Óptimo alcanzado"
    if valor > optimo:
        return "No alcanzado"
    return "Mejor que la referencia (revisar)"


def comparar_dtw_ddtw(
    dtw_dir: Path,
    ddtw_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    dtw_functions = {_cec_index(p): p for p in dtw_dir.glob("CEC_*") if p.is_dir()}
    ddtw_functions = {_cec_index(p): p for p in ddtw_dir.glob("CEC_*") if p.is_dir()}
    indices = sorted(set(dtw_functions) & set(ddtw_functions))
    if not indices:
        raise ValueError("No hay funciones CEC comunes entre DTW y DDTW")

    optima = _optima_by_index(dtw_dir)
    raw_rows = []
    for index in indices:
        dtw = _load_values(dtw_functions[index])
        ddtw = _load_values(ddtw_functions[index])
        if len(dtw) != len(ddtw):
            raise ValueError(f"CEC {index}: DTW y DDTW no tienen el mismo número de runs")

        optimum = optima[index]
        denominator = abs(optimum) if optimum != 0 else 1.0
        mean_dtw = float(np.mean(dtw))
        mean_ddtw = float(np.mean(ddtw))
        best_dtw = float(np.min(dtw))
        best_ddtw = float(np.min(ddtw))
        raw_rows.append({
            "funcion": f"CEC {index}",
            "nombre_funcion": dtw_functions[index].name,
            "indice": index,
            "n_runs": len(dtw),
            "optimo": optimum,
            "dtw_media": mean_dtw,
            "dtw_std": float(np.std(dtw, ddof=1)),
            "dtw_mejor": best_dtw,
            "dtw_mejor_run": int(np.argmin(dtw)) + 1,
            "dtw_gap_mejor_pct": 100.0 * (best_dtw - optimum) / denominator,
            "dtw_estado_optimo": _estado_optimo(best_dtw, optimum),
            "ddtw_media": mean_ddtw,
            "ddtw_std": float(np.std(ddtw, ddof=1)),
            "ddtw_mejor": best_ddtw,
            "ddtw_mejor_run": int(np.argmin(ddtw)) + 1,
            "ddtw_gap_mejor_pct": 100.0 * (best_ddtw - optimum) / denominator,
            "ddtw_estado_optimo": _estado_optimo(best_ddtw, optimum),
            "dtw_gap_pct": 100.0 * (mean_dtw - optimum) / denominator,
            "ddtw_gap_pct": 100.0 * (mean_ddtw - optimum) / denominator,
            "diferencia_ddtw_menos_dtw": mean_ddtw - mean_dtw,
            "p_wilcoxon_raw": _wilcoxon_p(dtw, ddtw),
        })

    p_holm = ajustar_pvalues_holm([row["p_wilcoxon_raw"] for row in raw_rows])
    for row, adjusted in zip(raw_rows, p_holm):
        row["p_wilcoxon_holm"] = adjusted
        if row["dtw_media"] < row["ddtw_media"]:
            observed = "DTW"
        elif row["ddtw_media"] < row["dtw_media"]:
            observed = "DDTW"
        else:
            observed = "Empate"
        row["ganador_observado"] = observed

        if adjusted < 0.05 and observed != "Empate":
            row["decision"] = f"{observed} significativamente mejor"
            row["significancia"] = (
                "DTW mejor / DDTW peor (significativo)"
                if observed == "DTW"
                else "DDTW mejor / DTW peor (significativo)"
            )
        else:
            row["decision"] = "Sin diferencia significativa"
            row["significancia"] = "No significativa"

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "analisis_global_dtw_ddtw.csv"
    fields = [
        "funcion", "nombre_funcion", "n_runs", "optimo",
        "dtw_media", "dtw_std", "dtw_gap_pct", "dtw_mejor", "dtw_mejor_run",
        "dtw_gap_mejor_pct", "dtw_estado_optimo",
        "ddtw_media", "ddtw_std", "ddtw_gap_pct", "ddtw_mejor", "ddtw_mejor_run",
        "ddtw_gap_mejor_pct", "ddtw_estado_optimo", "diferencia_ddtw_menos_dtw",
        "ganador_observado", "p_wilcoxon_raw", "p_wilcoxon_holm", "decision", "significancia",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({field: row[field] for field in fields})

    dtw_sig = sum(row["decision"].startswith("DTW ") for row in raw_rows)
    ddtw_sig = sum(row["decision"].startswith("DDTW ") for row in raw_rows)
    ties = len(raw_rows) - dtw_sig - ddtw_sig
    mean_dtw_gap = float(np.mean([row["dtw_gap_pct"] for row in raw_rows]))
    mean_ddtw_gap = float(np.mean([row["ddtw_gap_pct"] for row in raw_rows]))

    md_path = output_dir / "analisis_global_dtw_ddtw.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Comparación global Hybrid DTW vs Hybrid DDTW — CEC2022\n\n")
        f.write("> La comparación se realiza dentro de cada función. Los valores "
                "globales se resumen mediante el gap porcentual respecto al "
                "óptimo; no se mezclan los valores crudos de F1–F12.\n\n")
        f.write(f"- **Funciones analizadas:** {len(raw_rows)}\n")
        f.write(f"- **Runs por función:** {raw_rows[0]['n_runs']}\n")
        f.write("- **Prueba:** Wilcoxon pareado por función\n")
        f.write(f"- **Corrección:** Holm sobre {len(raw_rows)} funciones\n")
        f.write(f"- **Gap medio DTW:** {mean_dtw_gap:.4f}%\n")
        f.write(f"- **Gap medio DDTW:** {mean_ddtw_gap:.4f}%\n")
        f.write(f"- **Resultado significativo:** DTW {dtw_sig} | DDTW {ddtw_sig} | "
                f"sin diferencia {ties}\n\n")
        f.write("| Función | Nombre | DTW media | DTW mejor | DTW mejor run | DTW óptimo | "
                "DDTW media | DDTW mejor | DDTW mejor run | DDTW óptimo | "
                "Ganador observado | p bruto | p Holm | Decisión | Significancia |\n")
        f.write("|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---:|---|---|\n")
        for row in raw_rows:
            f.write(
                f"| {row['funcion']} | {row['nombre_funcion']} | {row['dtw_media']:.6f} | "
                f"{row['dtw_mejor']:.6f} | {row['dtw_mejor_run']} | {row['dtw_estado_optimo']} | "
                f"{row['ddtw_media']:.6f} | {row['ddtw_mejor']:.6f} | "
                f"{row['ddtw_mejor_run']} | {row['ddtw_estado_optimo']} | "
                f"{row['ganador_observado']} | {row['p_wilcoxon_raw']:.4e} | "
                f"{row['p_wilcoxon_holm']:.4e} | {row['decision']} | "
                f"{row['significancia']} |\n"
            )

    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent / "resultados"
    parser.add_argument("--dtw-dir", type=Path)
    parser.add_argument("--ddtw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=default_root)
    args = parser.parse_args()

    dtw_dir = args.dtw_dir or _find_latest(default_root, "run_dtw_")
    ddtw_dir = args.ddtw_dir or _find_latest(default_root, "run_ddtw_")
    csv_path, md_path = comparar_dtw_ddtw(dtw_dir, ddtw_dir, args.output_dir)
    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] MD : {md_path}")


if __name__ == "__main__":
    main()
