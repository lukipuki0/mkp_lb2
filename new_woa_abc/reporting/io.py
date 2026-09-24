"""Escritura segura de resultados JSON, CSV y TXT."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

from new_woa_abc.core.woa_abc import OptimizationResult


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(jsonable(value), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    normalized = [dict(jsonable(row)) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            if normalized:
                fields: list[str] = []
                for row in normalized:
                    for key in row:
                        if key not in fields:
                            fields.append(key)
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in normalized:
                    writer.writerow({
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in row.items()
                    })
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_experiment_dir(base: Path, label: str) -> Path:
    base = base.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = base / f"{label}_{timestamp}"
    destination.mkdir(parents=False, exist_ok=False)
    return destination


def save_run_artifacts(
    run_dir: Path,
    result: OptimizationResult,
    summary: dict[str, Any],
    configuration: dict[str, Any],
    decoded: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    convergence_rows = []
    for iteration, best in enumerate(result.convergence_history):
        convergence_rows.append({
            "iteration": iteration,
            "best_global": best,
            "best_after_woa": result.woa_history[iteration],
            "best_after_abc": result.abc_history[iteration],
            "mode": result.mode_history[iteration],
            "intensity": result.intensity_history[iteration],
        })
    write_csv(run_dir / "historial_convergencia.csv", convergence_rows)
    write_csv(run_dir / "historial_dtw.csv", result.dtw_history)
    write_csv(run_dir / "historial_parametros.csv", result.parameter_history)
    write_json(run_dir / "eventos_control.json", result.control_events)
    write_json(
        run_dir / "resultado.json",
        {
            "summary": summary,
            "configuration": configuration,
            "best_solution": result.best_position,
            "decoded_solution": decoded or {},
            "domain_metrics": metrics or {},
        },
    )
    lines = [
        "RESULTADO NEW WOA--ABC",
        f"Problema              : {result.problem}",
        f"Variante              : {result.variant}",
        f"Semilla               : {result.seed}",
        f"Mejor costo           : {result.best_cost:.12g}",
        f"Evaluaciones objetivo : {result.objective_evaluations}",
        f"Entradas a explore    : {result.fire_count}",
        f"Transiciones de modo  : {result.transition_count}",
        f"Cambios de parámetros : {result.parameter_update_count}",
    ]
    (run_dir / "resumen.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "create_experiment_dir",
    "jsonable",
    "save_run_artifacts",
    "write_csv",
    "write_json",
]
