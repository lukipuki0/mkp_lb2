"""Result objects and persistence helpers for MCDP experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .environment import MCDPEvaluation, MCDP_State


@dataclass
class MCDPResult:
    """Auditable result record for one method/configuration/run."""

    method: str
    instance_id: int | str
    max_cells: int
    max_machines_per_cell: int
    cost: float
    feasible: bool
    assignments: list[int] = field(default_factory=list)
    part_assignments: list[int] = field(default_factory=list)
    exceptional_elements: int = 0
    runtime_seconds: float | None = None
    iterations: int | None = None
    switches: int | None = None
    run_id: int | str | None = None
    reason: str = ""

    @classmethod
    def from_state(
        cls,
        state: MCDP_State,
        method: str,
        instance_id: int | str,
        runtime_seconds: float | None = None,
        iterations: int | None = None,
        switches: int | None = None,
        run_id: int | str | None = None,
    ) -> "MCDPResult":
        """Create a record from a core state and its validated evaluation."""
        evaluation = state.evaluation()
        return cls.from_evaluation(
            evaluation=evaluation,
            method=method,
            instance_id=instance_id,
            max_cells=state.inst_info.max_cells,
            max_machines_per_cell=state.inst_info.max_machines_per_cell,
            runtime_seconds=runtime_seconds,
            iterations=iterations,
            switches=switches,
            run_id=run_id,
        )

    @classmethod
    def from_evaluation(
        cls,
        evaluation: MCDPEvaluation,
        method: str,
        instance_id: int | str,
        max_cells: int,
        max_machines_per_cell: int,
        runtime_seconds: float | None = None,
        iterations: int | None = None,
        switches: int | None = None,
        run_id: int | str | None = None,
    ) -> "MCDPResult":
        return cls(
            method=method,
            instance_id=instance_id,
            max_cells=max_cells,
            max_machines_per_cell=max_machines_per_cell,
            cost=float(evaluation.cost),
            feasible=bool(evaluation.feasible),
            assignments=list(evaluation.assignments),
            part_assignments=list(evaluation.part_assignments),
            exceptional_elements=int(evaluation.exceptional_elements),
            runtime_seconds=runtime_seconds,
            iterations=iterations,
            switches=switches,
            run_id=run_id,
            reason=evaluation.reason,
        )

    def to_record(self) -> dict[str, object]:
        """Return a flat, CSV-friendly representation."""
        record = asdict(self)
        record["assignments"] = json.dumps(self.assignments, separators=(",", ":"))
        record["part_assignments"] = json.dumps(self.part_assignments, separators=(",", ":"))
        return record


def save_results_csv(results: Iterable[MCDPResult], filepath: str | Path) -> Path:
    """Save complete per-run records to CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_record() for result in results]
    fields = [
        "method", "instance_id", "run_id", "max_cells", "max_machines_per_cell",
        "cost", "feasible", "exceptional_elements", "runtime_seconds",
        "iterations", "switches", "assignments", "part_assignments", "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_results_json(results: Iterable[MCDPResult], filepath: str | Path) -> Path:
    """Save complete per-run records to JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(result) for result in results], handle, ensure_ascii=False, indent=2)
    return path


def save_best_result(result: MCDPResult, filepath: str | Path) -> Path:
    """Save one best-result record as JSON for convenient inspection."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, ensure_ascii=False, indent=2)
    return path
