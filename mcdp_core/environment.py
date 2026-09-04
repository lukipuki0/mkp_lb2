"""Core model for the Machine Cell Design Problem (MCDP).

The model follows the formulation used in ``MCDP-DL-DRL``.  A solution assigns
each machine to one manufacturing cell.  The part-to-cell assignment is
derived from the majority of machines that process each part, and the
objective counts inter-cell exceptional elements.

This module is independent of PyTorch and of any metaheuristic so it can be
used later by standalone solvers and by the DTW/DDTW orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import random

import numpy as np


@dataclass(frozen=True)
class MCDPEvaluation:
    """Complete evaluation of one machine-to-cell assignment."""

    cost: float
    feasible: bool
    assignments: tuple[int, ...]
    part_assignments: tuple[int, ...]
    exceptional_elements: int
    reason: str = ""


class MCDP_Instance:
    """One binary machine-part matrix and its cell-capacity constraints.

    ``incidence_matrix[i, j] = 1`` means that machine ``i`` processes part
    ``j``.  Cells may remain empty.  An instance is allowed to represent an
    infeasible capacity configuration; evaluations then report infeasibility
    instead of failing during construction.
    """

    def __init__(self, incidence_matrix: Sequence[Sequence[int]], max_cells: int, max_machines_per_cell: int) -> None:
        matrix = np.asarray(incidence_matrix, dtype=int)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("incidence_matrix must be a non-empty 2-D matrix")
        if not np.all(np.isin(matrix, (0, 1))):
            raise ValueError("incidence_matrix must contain only 0 and 1")
        if int(max_cells) != max_cells or int(max_cells) < 1:
            raise ValueError("max_cells must be a positive integer")
        if int(max_machines_per_cell) != max_machines_per_cell or int(max_machines_per_cell) < 1:
            raise ValueError("max_machines_per_cell must be a positive integer")

        self.incidence_matrix = matrix.copy()
        self.max_cells = int(max_cells)
        self.max_machines_per_cell = int(max_machines_per_cell)
        self.num_machines, self.num_parts = self.incidence_matrix.shape
        self.n_machines = self.num_machines
        self.n_parts = self.num_parts

    def validate_assignment(self, assignments: Sequence[int], allow_partial: bool = False) -> tuple[bool, str]:
        """Validate length, cell labels and capacity constraints."""
        try:
            values = np.asarray(assignments, dtype=int)
        except (TypeError, ValueError):
            return False, "assignments must contain integers"
        if values.ndim != 1 or values.size != self.num_machines:
            return False, f"expected {self.num_machines} machine assignments"
        if allow_partial:
            if np.any(values < -1):
                return False, "partial assignments may use only -1 for unassigned machines"
        elif np.any(values < 0):
            return False, "all machines must be assigned"
        if np.any(values >= self.max_cells):
            return False, "assignment refers to a cell outside the available range"
        assigned = values[values >= 0]
        loads = np.bincount(assigned, minlength=self.max_cells)
        if np.any(loads > self.max_machines_per_cell):
            return False, "at least one cell exceeds its machine capacity"
        if not allow_partial and assigned.size != self.num_machines:
            return False, "all machines must be assigned"
        return True, ""

    def derive_part_assignments(self, assignments: Sequence[int]) -> np.ndarray:
        """Assign each part to the cell containing most of its machines.

        Ties are resolved by the lowest cell index, matching ``numpy.argmax``
        and the source implementation.
        """
        values = np.asarray(assignments, dtype=int)
        valid, reason = self.validate_assignment(values, allow_partial=False)
        if not valid:
            raise ValueError(reason)
        part_cells = np.zeros(self.num_parts, dtype=int)
        for part_id in range(self.num_parts):
            machines = np.flatnonzero(self.incidence_matrix[:, part_id] == 1)
            if machines.size == 0:
                part_cells[part_id] = 0
                continue
            votes = np.bincount(values[machines], minlength=self.max_cells)
            part_cells[part_id] = int(np.argmax(votes))
        return part_cells

    def evaluate(self, assignments: Sequence[int]) -> MCDPEvaluation:
        """Evaluate the source objective.

        For every part ``j``, if ``n_j`` machines process it and ``v_jk`` of
        them are assigned to cell ``k``, the objective is

        ``Z = sum_j (n_j - max_k v_jk)``.

        Therefore, ``Z`` counts processing ones assigned outside the majority
        cell of each part.  Zeros inside a machine-cell block are not penalized
        by the source model.
        """
        values = np.asarray(assignments, dtype=int)
        valid, reason = self.validate_assignment(values, allow_partial=False)
        if not valid:
            return MCDPEvaluation(
                cost=float("inf"),
                feasible=False,
                assignments=tuple(values.tolist()) if values.ndim == 1 else tuple(),
                part_assignments=tuple(),
                exceptional_elements=0,
                reason=reason,
            )

        part_cells = self.derive_part_assignments(values)
        cost = 0
        for part_id in range(self.num_parts):
            machines = np.flatnonzero(self.incidence_matrix[:, part_id] == 1)
            if machines.size == 0:
                continue
            votes = np.bincount(values[machines], minlength=self.max_cells)
            cost += int(machines.size - np.max(votes))

        return MCDPEvaluation(
            cost=float(cost),
            feasible=True,
            assignments=tuple(int(x) for x in values),
            part_assignments=tuple(int(x) for x in part_cells),
            exceptional_elements=int(cost),
            reason="",
        )

    def objective(self, assignments: Sequence[int]) -> float:
        """Return the minimization objective, or ``inf`` if invalid."""
        return self.evaluate(assignments).cost

    def is_feasible(self, assignments: Sequence[int]) -> bool:
        """Return whether a complete assignment satisfies all constraints."""
        return self.evaluate(assignments).feasible

    def __repr__(self) -> str:
        return f"MCDP_Instance(M={self.num_machines}, P={self.num_parts}, max_cells={self.max_cells}, capacity={self.max_machines_per_cell})"


class MCDP_State:
    """Mutable state used by constructive and local-search procedures."""

    def __init__(self, inst_info: MCDP_Instance, assignments: Sequence[int] | None = None) -> None:
        self.inst_info = inst_info
        raw = [-1] * inst_info.num_machines if assignments is None else list(assignments)
        valid, reason = inst_info.validate_assignment(raw, allow_partial=True)
        if not valid:
            raise ValueError(reason)
        self.assignments = [int(x) for x in raw]
        self.unassigned_machines = {i for i, cell in enumerate(self.assignments) if cell == -1}
        self.cell_loads = [0] * inst_info.max_cells
        for cell in self.assignments:
            if cell >= 0:
                self.cell_loads[cell] += 1
        self.is_complete = not self.unassigned_machines
        self.part_assignments = [-1] * inst_info.num_parts
        self.feasible = False
        self.cost = float("inf")
        self.reason = "incomplete assignment" if not self.is_complete else ""
        if self.is_complete:
            self.update_cost()

    def update_cost(self) -> float:
        """Recompute derived part assignments, feasibility and cost."""
        if not self.is_complete:
            self.feasible = False
            self.cost = float("inf")
            self.reason = "incomplete assignment"
            return self.cost
        evaluation = self.inst_info.evaluate(self.assignments)
        self.feasible = evaluation.feasible
        self.cost = evaluation.cost
        self.reason = evaluation.reason
        self.part_assignments = list(evaluation.part_assignments)
        return self.cost

    def evaluation(self) -> MCDPEvaluation:
        """Return the current evaluation, including incomplete-state status."""
        if not self.is_complete:
            return MCDPEvaluation(float("inf"), False, tuple(self.assignments), tuple(), 0, "incomplete assignment")
        return self.inst_info.evaluate(self.assignments)


class MCDP_Environment:
    """Generate valid constructive and local-search transitions."""

    @staticmethod
    def gen_actions(state: MCDP_State, type_action: str, max_3swaps: int = 30) -> list[tuple]:
        actions: list[tuple] = []
        inst = state.inst_info
        if type_action == "constructive":
            if state.is_complete:
                return actions
            next_machine = min(state.unassigned_machines)
            for cell_id in range(inst.max_cells):
                if state.cell_loads[cell_id] < inst.max_machines_per_cell:
                    actions.append(("assign", next_machine, cell_id))
            return actions
        if type_action != "local_search" or not state.is_complete:
            return actions

        for machine_id in range(inst.num_machines):
            current_cell = state.assignments[machine_id]
            for target_cell in range(inst.max_cells):
                if target_cell != current_cell and state.cell_loads[target_cell] < inst.max_machines_per_cell:
                    actions.append(("move", machine_id, target_cell))
        for machine_a in range(inst.num_machines):
            for machine_b in range(machine_a + 1, inst.num_machines):
                if state.assignments[machine_a] != state.assignments[machine_b]:
                    actions.append(("swap", machine_a, machine_b))
        if inst.num_machines >= 3:
            for _ in range(max(0, int(max_3swaps))):
                machine_a, machine_b, machine_c = random.sample(range(inst.num_machines), 3)
                cells = (state.assignments[machine_a], state.assignments[machine_b], state.assignments[machine_c])
                if not (cells[0] == cells[1] == cells[2]):
                    actions.append(("3swap", int(machine_a), int(machine_b), int(machine_c)))
        return actions

    @staticmethod
    def state_transition(state: MCDP_State, action: tuple) -> MCDP_State:
        """Apply one valid action and return a fresh state."""
        if not action:
            raise ValueError("action cannot be empty")
        assignments = list(state.assignments)
        action_type = action[0]
        if action_type == "assign":
            _, machine_id, cell_id = action
            if state.is_complete or machine_id not in state.unassigned_machines:
                raise ValueError("assign action requires an unassigned machine")
            if not 0 <= cell_id < state.inst_info.max_cells or state.cell_loads[cell_id] >= state.inst_info.max_machines_per_cell:
                raise ValueError("invalid or full target cell")
            assignments[machine_id] = cell_id
        elif action_type == "move":
            _, machine_id, cell_id = action
            if not state.is_complete or not 0 <= machine_id < state.inst_info.num_machines:
                raise ValueError("move action requires a complete state")
            if not 0 <= cell_id < state.inst_info.max_cells or assignments[machine_id] == cell_id:
                raise ValueError("invalid move target")
            if state.cell_loads[cell_id] >= state.inst_info.max_machines_per_cell:
                raise ValueError("target cell is full")
            assignments[machine_id] = cell_id
        elif action_type == "swap":
            _, machine_a, machine_b = action
            if not state.is_complete or machine_a == machine_b or any(not 0 <= m < state.inst_info.num_machines for m in (machine_a, machine_b)):
                raise ValueError("invalid swap")
            if assignments[machine_a] == assignments[machine_b]:
                raise ValueError("swap machines must belong to different cells")
            assignments[machine_a], assignments[machine_b] = assignments[machine_b], assignments[machine_a]
        elif action_type == "3swap":
            _, machine_a, machine_b, machine_c = action
            machines = (machine_a, machine_b, machine_c)
            if not state.is_complete or len(set(machines)) != 3 or any(not 0 <= m < state.inst_info.num_machines for m in machines):
                raise ValueError("invalid 3swap")
            cell_a, cell_b, cell_c = (assignments[m] for m in machines)
            assignments[machine_a], assignments[machine_b], assignments[machine_c] = cell_b, cell_c, cell_a
        else:
            raise ValueError(f"unknown action type: {action_type}")
        return MCDP_State(state.inst_info, assignments)


class MCDP_RLEnvironment:
    """Small Gym-like construction environment retained for future DRL use."""

    def __init__(self, instance: MCDP_Instance) -> None:
        self.inst = instance
        self.state: MCDP_State | None = None
        self.current_m = 0

    def reset(self) -> np.ndarray:
        self.state = MCDP_State(self.inst)
        self.current_m = 0
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        from .data import get_features
        if self.state is None:
            raise RuntimeError("reset() must be called before get_obs()")
        if self.current_m >= self.inst.num_machines:
            return np.zeros((self.inst.max_cells, 5), dtype=np.float32)
        sequence = [get_features(self.state, self.current_m, cell_id) for cell_id in range(self.inst.max_cells)]
        return np.asarray(sequence, dtype=np.float32)

    def get_valid_mask(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("reset() must be called before get_valid_mask()")
        return np.asarray([int(load < self.inst.max_machines_per_cell) for load in self.state.cell_loads], dtype=np.int64)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        from .data import get_features
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        if self.current_m >= self.inst.num_machines:
            raise RuntimeError("episode has already finished")
        cell_id = int(action)
        if not 0 <= cell_id < self.inst.max_cells:
            raise ValueError("action is outside the available cell range")
        if self.state.cell_loads[cell_id] >= self.inst.max_machines_per_cell:
            return self.get_obs(), -2.0, False

        features = get_features(self.state, self.current_m, cell_id)
        reward = 0.6 * features[2] + 0.4 * features[3] + (0.1 if features[4] == 1.0 else 0.0)
        self.state.assignments[self.current_m] = cell_id
        self.state.cell_loads[cell_id] += 1
        self.state.unassigned_machines.remove(self.current_m)
        self.current_m += 1
        done = self.current_m >= self.inst.num_machines
        if done:
            self.state.is_complete = True
            self.state.update_cost()
            reward += 100.0 / (self.state.cost + 1.0)
        return self.get_obs(), float(reward), done
