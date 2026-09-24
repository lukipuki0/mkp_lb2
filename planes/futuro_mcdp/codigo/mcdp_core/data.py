"""Data and feature utilities for the MCDP core model."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np

from .environment import MCDP_Environment, MCDP_Instance, MCDP_State


def get_features(state: MCDP_State, machine_id: int, cell_id: int) -> list[float]:
    """Return the five source-project features for one candidate cell.

    The features are normalized cell load, full-cell flag, mean Jaccard
    affinity, maximum Jaccard affinity, and empty-cell flag.  The fifth value
    is included because the original DRL reward uses ``features[4]``.
    """
    inst = state.inst_info
    if not 0 <= machine_id < inst.num_machines:
        raise ValueError("machine_id is outside the instance")
    if not 0 <= cell_id < inst.max_cells:
        raise ValueError("cell_id is outside the instance")

    load = state.cell_loads[cell_id]
    capacity = inst.max_machines_per_cell
    matrix = inst.incidence_matrix
    parts_machine = np.flatnonzero(matrix[machine_id] == 1)
    affinities: list[float] = []
    for other_machine, assigned_cell in enumerate(state.assignments):
        if assigned_cell != cell_id:
            continue
        parts_other = np.flatnonzero(matrix[other_machine] == 1)
        union = np.union1d(parts_machine, parts_other).size
        if union:
            intersection = np.intersect1d(parts_machine, parts_other).size
            affinities.append(float(intersection / union))

    mean_affinity = float(np.mean(affinities)) if affinities else 0.0
    max_affinity = float(np.max(affinities)) if affinities else 0.0
    return [
        float(load / capacity),
        float(load >= capacity),
        mean_affinity,
        max_affinity,
        float(load == 0),
    ]


# Name retained for compatibility with the DRL source project.
get_rl_features = get_features


class ZodiacGenerator:
    """Generate shuffled block-structured synthetic MCDP matrices."""

    def __init__(self, min_m: int = 15, max_m: int = 25, min_p: int = 20, max_p: int = 40, n_cells: int = 3, noise_level: float = 0.1) -> None:
        if not 0 <= noise_level <= 1:
            raise ValueError("noise_level must be between 0 and 1")
        self.m_range = (int(min_m), int(max_m))
        self.p_range = (int(min_p), int(max_p))
        self.n_cells = int(n_cells)
        self.noise = float(noise_level)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(matrix, ideal_machine_assignments)``."""
        machines = random.randint(*self.m_range)
        parts = random.randint(*self.p_range)
        if self.n_cells < 1 or self.n_cells > min(machines, parts):
            raise ValueError("n_cells must not exceed the generated dimensions")

        machine_cuts = sorted(random.sample(range(1, machines), self.n_cells - 1))
        part_cuts = sorted(random.sample(range(1, parts), self.n_cells - 1))
        machine_ranges = [0] + machine_cuts + [machines]
        part_ranges = [0] + part_cuts + [parts]

        matrix = np.zeros((machines, parts), dtype=np.int8)
        ideal_assignments = np.zeros(machines, dtype=int)
        for cell_id in range(self.n_cells):
            m_start, m_end = machine_ranges[cell_id], machine_ranges[cell_id + 1]
            p_start, p_end = part_ranges[cell_id], part_ranges[cell_id + 1]
            matrix[m_start:m_end, p_start:p_end] = 1
            ideal_assignments[m_start:m_end] = cell_id

        noise_mask = np.random.random((machines, parts)) < self.noise
        matrix = np.logical_xor(matrix, noise_mask).astype(np.int8)
        for machine_id in range(machines):
            if not matrix[machine_id].any():
                matrix[machine_id, random.randrange(parts)] = 1
        for part_id in range(parts):
            if not matrix[:, part_id].any():
                matrix[random.randrange(machines), part_id] = 1

        row_permutation = np.random.permutation(machines)
        col_permutation = np.random.permutation(parts)
        return matrix[row_permutation][:, col_permutation], ideal_assignments[row_permutation]


def generate_zodiac_dataset(n_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Generate supervised construction features and ideal cell labels."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    features: list[list[list[float]]] = []
    labels: list[int] = []
    for sample_id in range(n_samples):
        n_cells = 2 if sample_id % 2 == 0 else 3
        matrix, ideal = ZodiacGenerator(n_cells=n_cells, noise_level=0.15).generate()
        machines, _ = matrix.shape
        minimum_capacity = int(np.ceil(machines / n_cells) + 2)
        capacity = max(minimum_capacity, int(np.bincount(ideal, minlength=n_cells).max()))
        instance = MCDP_Instance(matrix, n_cells, capacity)
        state = MCDP_State(instance)
        for machine_id, target_cell in enumerate(ideal):
            candidate_features = [get_features(state, machine_id, cell_id) for cell_id in range(n_cells)]
            while len(candidate_features) < 3:
                candidate_features.append([0.0, 0.0, 0.0, 0.0, 1.0])
            features.append(candidate_features)
            labels.append(int(target_cell))
            state.assignments[machine_id] = int(target_cell)
            state.cell_loads[target_cell] += 1
            state.unassigned_machines.remove(machine_id)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def generate_random_instance(
    num_machines: int,
    num_parts: int,
    density: float = 0.3,
    max_cells: int = 3,
    max_machines_per_cell: int | None = None,
) -> MCDP_Instance:
    """Generate one random binary MCDP instance."""
    if num_machines < 1 or num_parts < 1 or not 0 <= density <= 1:
        raise ValueError("dimensions must be positive and density must be in [0, 1]")
    matrix = np.random.choice([0, 1], size=(num_machines, num_parts), p=[1 - density, density]).astype(np.int8)
    for machine_id in range(num_machines):
        if not matrix[machine_id].any():
            matrix[machine_id, random.randrange(num_parts)] = 1
    for part_id in range(num_parts):
        if not matrix[:, part_id].any():
            matrix[random.randrange(num_machines), part_id] = 1
    if max_machines_per_cell is None:
        max_machines_per_cell = int(np.ceil(num_machines / max_cells) + 2)
    return MCDP_Instance(matrix, max_cells, max_machines_per_cell)


def load_instances_from_file(filepath: str | Path) -> list[np.ndarray]:
    """Load comma- or whitespace-separated matrices from a text file.

    Headers such as ``matriz 1`` delimit matrices.  Every matrix is validated
    as rectangular and binary; malformed input raises an explanatory error.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"MCDP instance file not found: {path}")

    matrices: list[np.ndarray] = []
    current: list[list[int]] = []

    def flush(line_number: int) -> None:
        nonlocal current
        if not current:
            return
        widths = {len(row) for row in current}
        if len(widths) != 1:
            raise ValueError(f"non-rectangular matrix before line {line_number}")
        matrix = np.asarray(current, dtype=int)
        if not np.all(np.isin(matrix, (0, 1))):
            raise ValueError(f"matrix before line {line_number} is not binary")
        matrices.append(matrix)
        current = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith("matriz") or lowered.startswith("matrix"):
                flush(line_number)
                continue
            tokens = line.replace(",", " ").split()
            try:
                row = [int(token) for token in tokens]
            except ValueError as exc:
                raise ValueError(f"invalid matrix row at line {line_number}") from exc
            if not row:
                continue
            current.append(row)
    flush(line_number if 'line_number' in locals() else 0)
    if not matrices:
        raise ValueError(f"no matrices found in {path}")
    return matrices


def load_mcdp_instances(filepath: str | Path, max_cells: int, max_machines_per_cell: int) -> list[MCDP_Instance]:
    """Load matrices and wrap them as validated :class:`MCDP_Instance` objects."""
    return [MCDP_Instance(matrix, max_cells, max_machines_per_cell) for matrix in load_instances_from_file(filepath)]
