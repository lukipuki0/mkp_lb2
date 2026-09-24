"""Ejecuta WOA--ABC + DTW sobre todas las instancias y configuraciones MCDP.

El protocolo replica el lote de ``MCDP-DL-DRL``:

* 10 matrices de incidencia;
* 2 celdas con capacidades 8, 9, 10, 11 y 12;
* 3 celdas con capacidades 6, 7, 8 y 9.

    Por defecto se genera una carpeta ``woa_abc/results/run_TIMESTAMP``
con la tabla, el CSV de la corrida, un JSON detallado y la configuración
utilizada.

Ejemplo:
    python -m woa_abc.run_cooperative_mcdp_batch --iterations 300
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from dtw_stagnation import StagnationConfig
from mcdp_core.data import load_mcdp_instances
from woa_abc.cooperativo_mcdp_dtw import (
    CooperativeMCDPParams,
    ejecutar_mcdp_cooperativo,
)


CONFIGS: tuple[tuple[int, tuple[int, ...]], ...] = (
    (2, (8, 9, 10, 11, 12)),
    (3, (6, 7, 8, 9)),
)
METHOD_NAME = "CooperativeMCDP"
RUN_HEADERS = [
    "Instance_ID",
    "Max_Cells",
    "Machines_Per_Cell",
    "Seed",
    "Cost",
    "Optimum",
    "Gap_Percent",
    "Solution_Vector",
]
def _default_paths() -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    input_file = repo_root / "mcdp_core" / "instances" / "instancias.txt"
    output_root = repo_root / "woa_abc" / "results"
    optima_file = repo_root / "woa_abc" / "mcdp_optimos.csv"
    return input_file, output_root, optima_file


def _make_parser() -> argparse.ArgumentParser:
    input_file, output_root, optima_file = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(input_file), help="Archivo con las matrices MCDP")
    parser.add_argument("--output-root", default=str(output_root), help="Raíz de carpetas de resultados")
    parser.add_argument("--optima-file", default=str(optima_file), help="CSV con los óptimos publicados")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="Límite de scouts ABC")
    parser.add_argument("--window", type=int, default=75, help="Ventana del monitor DTW")
    parser.add_argument("--band", type=int, default=0, help="Banda DTW; 0 = automática")
    parser.add_argument("--min-slope", type=float, default=0.1)
    parser.add_argument("--plateau-max", type=int, default=15)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--ddtw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapt", dest="adapt_thresholds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Mostrar el detalle de cada iteración")
    return parser


def _make_params(args: argparse.Namespace, seed: int) -> CooperativeMCDPParams:
    stag_config = StagnationConfig(
        window=args.window,
        band=args.band,
        patience=args.patience,
        plateau_max=args.plateau_max,
        min_slope=args.min_slope,
        use_ddtw=args.ddtw,
        adapt_thresholds=args.adapt_thresholds,
        p_low=30.0,
        p_high=70.0,
    )
    return CooperativeMCDPParams(
        pop_size=args.pop_size,
        iterations=args.iterations,
        epochs=args.epochs,
        limit=args.limit,
        abc_guide_strength=0.20,
        abc_phi_scale=1.0,
        exploration_probability=0.50,
        step_initial=1.0,
        step_final=0.05,
        momentum_factor=0.20,
        stag_cfg=stag_config,
        seed=seed,
    )


def _load_optima(path: Path) -> dict[tuple[int, int, int], float]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de óptimos: {path}")
    optima: dict[tuple[int, int, int], float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (int(row["Instance_ID"]), int(row["Max_Cells"]), int(row["Machines_Per_Cell"]))
                optimum = float(row["Optimum"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Fila inválida en el archivo de óptimos: {row}") from exc
            if optimum <= 0:
                raise ValueError(f"El óptimo debe ser positivo: {key}={optimum}")
            optima[key] = optimum
    expected = len(CONFIGS[0][1]) * 10 + len(CONFIGS[1][1]) * 10
    if len(optima) != expected:
        raise ValueError(f"Se esperaban {expected} óptimos y se cargaron {len(optima)}")
    return optima


def _table_header() -> list[str]:
    cells_2 = CONFIGS[0][1]
    cells_3 = CONFIGS[1][1]
    header_1 = (
        "Prob Coop".center(8)
        + " | "
        + "N=2".center(len(cells_2) * 4)
        + " | "
        + "N=3".center(len(cells_3) * 4)
    )
    header_2 = (
        "".center(8)
        + " | "
        + "".join(str(cap).center(4) for cap in cells_2)
        + " | "
        + "".join(str(cap).center(4) for cap in cells_3)
    )
    separator = "-" * len(header_2)
    return [separator, header_1, header_2, separator]


def _format_table(results: dict[int, dict[tuple[int, int], float]], count: int) -> str:
    lines = _table_header()
    for instance_id in range(1, count + 1):
        row = f"{instance_id}".center(8) + " | "
        for cells, capacities in CONFIGS:
            for capacity in capacities:
                value = results[instance_id].get((cells, capacity), "-")
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                row += str(value).center(4)
            if cells == 2:
                row += " | "
        lines.append(row)
    lines.append(lines[0])
    return "\n".join(lines)


def _format_gap_table(gaps: dict[int, dict[tuple[int, int], float]], count: int) -> str:
    width = 8
    cells_2 = CONFIGS[0][1]
    cells_3 = CONFIGS[1][1]
    header_1 = "Gap (%)".center(8) + " | " + "N=2".center(len(cells_2) * width) + " | " + "N=3".center(len(cells_3) * width)
    header_2 = "".center(8) + " | " + "".join(str(cap).center(width) for cap in cells_2) + " | " + "".join(str(cap).center(width) for cap in cells_3)
    separator = "-" * len(header_2)
    lines = [separator, header_1, header_2, separator]
    for instance_id in range(1, count + 1):
        line = f"{instance_id}".center(8) + " | "
        for cells, capacities in CONFIGS:
            for capacity in capacities:
                line += f"{gaps[instance_id][(cells, capacity)]:.2f}%".center(width)
            if cells == 2:
                line += " | "
        lines.append(line)
    lines.append(separator)
    return "\n".join(lines)


def _write_run_csv(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RUN_HEADERS)
        for record in records:
            writer.writerow(
                [
                    record["Instance_ID"],
                    record["Max_Cells"],
                    record["Machines_Per_Cell"],
                    record.get("Seed", ""),
                    record["Cost"],
                    record["Optimum"],
                    record["Gap_Percent"],
                    json.dumps(record["Solution_Vector"], separators=(",", ":")),
                ]
            )


def _write_configuration_files(configuration: dict[str, Any], run_dir: Path) -> None:
    """Guarda la configuración completa en formatos JSON y TXT."""
    serialized = json.dumps(configuration, ensure_ascii=False, indent=2)
    (run_dir / "configuracion.json").write_text(serialized, encoding="utf-8")
    (run_dir / "configuracion.txt").write_text(
        "CONFIGURACIÓN COMPLETA — WOA--ABC + DTW para MCDP\n"
        + "=" * 58
        + "\n\n"
        + serialized
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _make_parser().parse_args()
    input_file = Path(args.file).resolve()
    output_root = Path(args.output_root).resolve()
    optima_file = Path(args.optima_file).resolve()

    matrices = load_mcdp_instances(input_file, max_cells=3, max_machines_per_cell=12)
    if len(matrices) != 10:
        raise ValueError(f"Se esperaban 10 instancias y se cargaron {len(matrices)} desde {input_file}")
    optima = _load_optima(optima_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    run_id = run_dir.name

    records: list[dict[str, Any]] = []
    table_results: dict[int, dict[tuple[int, int], float]] = {i: {} for i in range(1, len(matrices) + 1)}
    table_gaps: dict[int, dict[tuple[int, int], float]] = {i: {} for i in range(1, len(matrices) + 1)}
    started = time.perf_counter()

    for instance_id, matrix_instance in enumerate(matrices, start=1):
        print(f"\nInstancia {instance_id}/{len(matrices)} ({matrix_instance.num_machines}x{matrix_instance.num_parts})", flush=True)
        config_number = 0
        for cells, capacities in CONFIGS:
            for capacity in capacities:
                config_number += 1
                # Semilla distinta por instancia/configuración, pero reproducible.
                seed = args.seed + (instance_id - 1) * 100 + config_number
                instance = type(matrix_instance)(
                    matrix_instance.incidence_matrix,
                    max_cells=cells,
                    max_machines_per_cell=capacity,
                )
                params = _make_params(args, seed)
                started_config = time.perf_counter()
                result = ejecutar_mcdp_cooperativo(instance, params, verbose=args.verbose)
                elapsed = time.perf_counter() - started_config
                evaluation = result.evaluacion_global
                cost = float(evaluation.cost)
                optimum = optima[(instance_id, cells, capacity)]
                gap_percent = (cost - optimum) / optimum * 100.0
                table_results[instance_id][(cells, capacity)] = cost
                table_gaps[instance_id][(cells, capacity)] = gap_percent
                records.append({
                    "Instance_ID": instance_id,
                    "Max_Cells": cells,
                    "Machines_Per_Cell": capacity,
                    "Seed": seed,
                    "Cost": cost,
                    "Optimum": optimum,
                    "Gap_Percent": gap_percent,
                    "Feasible": bool(evaluation.feasible),
                    "Exceptional_Elements": int(evaluation.exceptional_elements),
                    "Solution_Vector": list(result.mejor_sol_global),
                    "Part_Assignments": list(evaluation.part_assignments),
                    "Runtime_Seconds": elapsed,
                    "Iterations": sum(epoch.iteraciones for epoch in result.epochs),
                    "Stagnation_Fires": sum(epoch.stagnation_fires for epoch in result.epochs),
                    "Run_ID": run_id,
                    "Method": METHOD_NAME,
                    "Reason": evaluation.reason,
                })
                print(
                    f"  {cells} celdas / capacidad {capacity}: "
                    f"costo={cost:.0f}, tiempo={elapsed:.2f}s",
                    flush=True,
                )

    table = _format_table(table_results, len(matrices))
    gap_table = _format_gap_table(table_gaps, len(matrices))
    (run_dir / "resultados_tabla_CooperativeMCDP.txt").write_text(table + "\n", encoding="utf-8")
    (run_dir / "resultados_tabla_gap_CooperativeMCDP.txt").write_text(gap_table + "\n", encoding="utf-8")
    _write_run_csv(records, run_dir / "results_cooperative_mcdp.csv")
    (run_dir / "results_cooperative_mcdp.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base_params = _make_params(args, args.seed)
    configuration = {
        "method": METHOD_NAME,
        "input_file": str(input_file),
        "optima_file": str(optima_file),
        "instances": len(matrices),
        "configurations_per_instance": sum(len(caps) for _, caps in CONFIGS),
        "configurations": {str(cells): list(caps) for cells, caps in CONFIGS},
        "parameters": asdict(base_params),
        "seed_strategy": "seed = seed_base + (instance_id - 1) * 100 + configuration_number",
        "seed_base": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_configuration_files(configuration, run_dir)

    print("\n" + table)
    print(f"\nResultados guardados en: {run_dir}")


if __name__ == "__main__":
    main()
