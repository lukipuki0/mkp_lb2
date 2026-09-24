"""Consolida los resultados CEC2022 producidos en paralelo.

Uso:
    python -m continuous_benchmark.resumir_cec2022 \
        --batch-dir continuous_benchmark/resultados/run_ddtw_job_19058
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
from pathlib import Path

import numpy as np

from continuous_benchmark.benchmark_continuo import (
    escribir_resumen_global_descriptivo,
    grafico_boxplot_global,
    resultado_frente_optimo,
)
from continuous_benchmark.funciones_cec2022 import get_test_functions


CEC_DIR_RE = re.compile(r"CEC_(\d{2})_")


def cargar_resultados(batch_dir: Path, dimension: int) -> list[dict]:
    funciones = get_test_functions(dimension)
    carpetas: dict[int, Path] = {}

    for path in batch_dir.glob("CEC_*"):
        if not path.is_dir():
            continue
        match = CEC_DIR_RE.match(path.name)
        if match:
            carpetas[int(match.group(1))] = path

    esperadas = set(range(1, len(funciones) + 1))
    faltantes = sorted(esperadas - set(carpetas))
    if faltantes:
        raise FileNotFoundError(
            "Faltan carpetas de funciones CEC: "
            + ", ".join(f"F{i}" for i in faltantes)
        )

    resumen: list[dict] = []
    cantidad_runs: int | None = None

    for indice, func in enumerate(funciones, 1):
        csv_path = carpetas[indice] / "runs_resultados.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Falta el archivo {csv_path}")

        with csv_path.open(newline="", encoding="utf-8") as archivo:
            filas = list(csv.DictReader(archivo))
        valores = [float(fila["mejor_valor"]) for fila in filas]
        if not valores:
            raise ValueError(f"No hay runs en {csv_path}")

        if cantidad_runs is None:
            cantidad_runs = len(valores)
        elif len(valores) != cantidad_runs:
            raise ValueError(
                f"F{indice} tiene {len(valores)} runs; se esperaban {cantidad_runs}"
            )

        arreglo = np.asarray(valores, dtype=float)
        mejor_run = int(np.argmin(arreglo)) + 1
        resumen.append({
            "nombre": func.name,
            "cec_label": f"CEC {indice}",
            "n_dim": func.n_dim,
            "valor_optimo": float(func.optimum),
            "media": float(np.mean(arreglo)),
            "std": float(np.std(arreglo, ddof=1)) if len(arreglo) > 1 else 0.0,
            "mediana": float(np.median(arreglo)),
            "mejor": float(np.min(arreglo)),
            "mejor_run": mejor_run,
            "peor": float(np.max(arreglo)),
            "n_runs": len(valores),
            "valores_runs": valores,
        })

    return resumen


def escribir_resumenes(
    resumen: list[dict],
    batch_dir: Path,
    max_iters: int,
) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    n_runs = resumen[0]["n_runs"]

    grafico_boxplot_global(resumen, str(batch_dir))

    resumen_txt = batch_dir / "resumen_global.txt"
    with resumen_txt.open("w", encoding="utf-8") as archivo:
        archivo.write("RESUMEN GLOBAL DEL BENCHMARK CONTINUO\n")
        archivo.write(f"Fecha       : {timestamp}\n")
        archivo.write(f"Funciones   : {len(resumen)}\n")
        archivo.write(f"Runs/func   : {n_runs}\n")
        archivo.write(f"Iters/run   : {max_iters}\n\n")
        archivo.write(
            f"{'#':<3} {'Funcion':<46} {'Dim':>4} {'Media':>14} {'Std':>12} "
            f"{'Mediana':>12} {'Mejor':>12} {'Optimo':>10}\n"
        )
        archivo.write("-" * 123 + "\n")
        for indice, fila in enumerate(resumen, 1):
            archivo.write(
                f"{indice:<3} {fila['nombre']:<46} {fila['n_dim']:>4} "
                f"{fila['media']:>14.4f} {fila['std']:>12.4f} "
                f"{fila['mediana']:>12.4f} {fila['mejor']:>12.4f} "
                f"{fila['valor_optimo']:>10.4f}\n"
            )

    resumen_csv = batch_dir / "resumen_global.csv"
    with resumen_csv.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow([
            "funcion", "n_dim", "n_runs", "media", "std", "mediana",
            "mejor", "mejor_run", "peor", "valor_optimo",
            "resultado_mejor_frente_optimo",
        ])
        for fila in resumen:
            writer.writerow([
                fila["nombre"], fila["n_dim"], fila["n_runs"], fila["media"],
                fila["std"], fila["mediana"], fila["mejor"],
                fila["mejor_run"], fila["peor"], fila["valor_optimo"],
                resultado_frente_optimo(fila["mejor"], fila["valor_optimo"]),
            ])

    runs_csv = batch_dir / "todos_los_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["funcion", "run", "mejor_valor", "valor_optimo"])
        for fila in resumen:
            for run, valor in enumerate(fila["valores_runs"], 1):
                writer.writerow([
                    fila["nombre"], run, valor, fila["valor_optimo"]
                ])

    resumen_md = batch_dir / "resumen_global.md"
    with resumen_md.open("w", encoding="utf-8") as archivo:
        archivo.write(f"# Resumen Global — Benchmark Continuo CEC2022 ({timestamp})\n\n")
        archivo.write(f"- **Total funciones:** {len(resumen)}\n")
        archivo.write(f"- **Runs por función:** {n_runs}\n")
        archivo.write(f"- **Máximo de iteraciones por run:** {max_iters}\n\n")
        archivo.write(
            "| # | Función | Dim | Media | Std | Mediana | Mejor | "
            "Mejor run | Óptimo | Estado |\n"
        )
        archivo.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
        )
        for indice, fila in enumerate(resumen, 1):
            estado = resultado_frente_optimo(
                fila["mejor"], fila["valor_optimo"]
            )
            archivo.write(
                f"| {indice} | `{fila['nombre']}` | {fila['n_dim']} | "
                f"{fila['media']:.6f} | {fila['std']:.6f} | "
                f"{fila['mediana']:.6f} | {fila['mejor']:.6f} | "
                f"{fila['mejor_run']} | {fila['valor_optimo']:.6f} | "
                f"{estado} |\n"
            )

    escribir_resumen_global_descriptivo(resumen, str(batch_dir), timestamp)

    print(f"[OK] Resumen TXT: {resumen_txt}")
    print(f"[OK] Resumen CSV: {resumen_csv}")
    print(f"[OK] Runs CSV   : {runs_csv}")
    print(f"[OK] Resumen MD : {resumen_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument(
        "--dimension",
        type=int,
        default=int(os.environ.get("CEC_DIMENSION", "10")),
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=int(os.environ.get("CEC_MAX_ITERS", "1000")),
    )
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    if not batch_dir.is_dir():
        raise NotADirectoryError(f"No existe el batch: {batch_dir}")

    resumen = cargar_resultados(batch_dir, args.dimension)
    escribir_resumenes(resumen, batch_dir, args.max_iters)


if __name__ == "__main__":
    main()
