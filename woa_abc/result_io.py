"""Persistencia común de ejecutores WOA--ABC.

Solo contiene serialización y estructura de carpetas; el análisis estadístico
inferencial se implementará en una etapa posterior.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RESULT_ROOT = Path(__file__).resolve().parent / "resultados"


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


def create_output_dir(
    domain: str,
    output_dir: str | Path | None = None,
    *,
    experiment_label: str | None = None,
) -> Path:
    """Crea una carpeta de resultados sin sobrescribir ejecuciones previas.

    Cuando ``output_dir`` apunta a una carpeta que ya existe (por ejemplo, la
    carpeta base usada por un script HPC), se crea dentro una subcarpeta
    ``run_<marca_de_tiempo>``. Así es seguro relanzar un experimento sin
    perder los CSV/JSON de la ejecución anterior.
    """
    if domain not in {"cec", "hres2"}:
        raise ValueError("domain debe ser 'cec' o 'hres2'")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if experiment_label is not None:
        base_path = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else RESULT_ROOT / domain
        )
        base_path.mkdir(parents=True, exist_ok=True)
        path = base_path / f"{experiment_label}_{timestamp}"
        path.mkdir(parents=False, exist_ok=False)
        return path
    base_path = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else RESULT_ROOT / domain / f"run_{timestamp}"
    )
    path = base_path
    if path.exists():
        path = path / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    normalized = [dict(jsonable(row)) for row in rows]
    if not normalized:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in normalized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized:
            flat = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(flat)


def save_run_details(path: Path, epoch: Any) -> None:
    path.mkdir(parents=True, exist_ok=True)
    convergence = []
    for index, best in enumerate(epoch.historial):
        convergence.append({
            "iteration": index,
            "best_global": best,
            "instantaneous": epoch.historial_inst[index],
            "woa_best": epoch.historial_woa[index],
            "abc_best": epoch.historial_abc[index],
        })
    write_csv(path / "historial_convergencia.csv", convergence)
    write_csv(path / "historial_dtw.csv", epoch.dtw_info_hist)
    write_csv(path / "historial_parametros.csv", epoch.parametros_historial)
    write_json(
        path / "eventos_control.json",
        {
            "variant": epoch.variant,
            "dtw_alarm_count": epoch.stagnation_fires,
            "mode_transition_count": epoch.mode_transitions,
            "parameter_update_count": epoch.parameter_updates,
            "adaptation_events": epoch.eventos_adaptacion,
            "cooperation_events": epoch.eventos_cooperacion,
        },
    )


def save_configuration(output_dir: Path, config: dict[str, Any]) -> None:
    write_json(output_dir / "configuracion.json", config)
    lines = [f"{key}: {jsonable(value)}" for key, value in config.items()]
    (output_dir / "configuracion.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "RESULT_ROOT",
    "create_output_dir",
    "jsonable",
    "save_configuration",
    "save_run_details",
    "write_csv",
    "write_json",
]

# Compatibilidad interna con ejecuciones antiguas; la salida activa ya no
# crea carpetas epoch_*.
save_epoch_details = save_run_details
