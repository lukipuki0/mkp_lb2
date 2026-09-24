"""Utilidades compartidas para reanudar campañas interrumpidas."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable


def load_typed_rows(
    path: Path,
    *,
    integer_fields: Iterable[str] = (),
    float_fields: Iterable[str] = (),
    boolean_fields: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Carga un CSV de checkpoint conservando los tipos usados por los runners."""

    if not path.is_file() or path.stat().st_size == 0:
        return []
    integer_fields = set(integer_fields)
    float_fields = set(float_fields)
    boolean_fields = set(boolean_fields)
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    row[key] = None
                elif key in integer_fields:
                    row[key] = int(value)
                elif key in float_fields:
                    row[key] = float(value)
                elif key in boolean_fields:
                    normalized = value.strip().lower()
                    if normalized not in {"true", "false"}:
                        raise ValueError(
                            f"booleano inválido en {path}, columna {key}: {value!r}"
                        )
                    row[key] = normalized == "true"
                else:
                    row[key] = value
            rows.append(row)
    return rows


def resume_directory(path: Path, expected_domain: str) -> tuple[Path, dict[str, Any]]:
    """Resuelve una campaña existente y carga su configuración."""

    output = path.expanduser().resolve()
    config_path = output / "configuracion.json"
    if not output.is_dir() or not config_path.is_file():
        raise ValueError(
            f"--resume-dir debe contener configuracion.json: {output}"
        )
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    if existing.get("domain") != expected_domain:
        raise ValueError(
            f"la campaña es de dominio {existing.get('domain')!r}, "
            f"no {expected_domain!r}"
        )
    return output, existing


def validate_resume_config(
    existing: dict[str, Any],
    expected: dict[str, Any],
    fields: Iterable[str],
) -> None:
    """Impide reanudar con semillas, presupuesto o variantes diferentes."""

    mismatches = [
        field
        for field in fields
        if existing.get(field) != expected.get(field)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: guardado={existing.get(field)!r}, solicitado={expected.get(field)!r}"
            for field in mismatches
        )
        raise ValueError(f"configuración incompatible al reanudar ({details})")


def archive_existing_directory(path: Path) -> Path | None:
    """Aparta un artefacto incompatible sin destruirlo."""

    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived = path.with_name(f"{path.name}_anterior_{timestamp}")
    path.rename(archived)
    return archived


__all__ = [
    "archive_existing_directory",
    "load_typed_rows",
    "resume_directory",
    "validate_resume_config",
]
