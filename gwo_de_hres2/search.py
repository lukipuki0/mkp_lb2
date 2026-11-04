"""
Shared-population GWO-DE on a bounded continuous minimization problem.

The population lives in the normalized box z in [-1, 1]^d; to_physical maps
it linearly to [lb_vector, ub_vector] of the function (HRES2Function decodes
that vector to capacities itself). Working in [-1, 1] lets the GWO and DE
operators of gwo_de_dtw run unchanged with v_max = 1.

Competitive mode only: each generation every individual gets one GWO and one
DE/rand/1/bin candidate, the one with lower objective wins, and it replaces
the individual when its objective is <= the current one. Leaders are the
three lowest objectives, copied at the start of the generation.

Cost: N evaluations at initialization, 2N per generation.

Spec: odd/tasks/gwo-de-hres2.md (S4).
"""

from typing import Dict

import numpy as np

from gwo_de_dtw.de import de_candidate
from gwo_de_dtw.gwo import gwo_candidate
from gwo_de_dtw.hybrid import EXPLOIT, Profile

V_MAX = 1.0


class GWODEContinuous:
    def __init__(self, func, rng: np.random.Generator, pop_size: int):
        if pop_size < 4:
            raise ValueError("pop_size must be >= 4 (DE/rand/1 needs three other individuals)")
        self.func = func
        self.rng = rng
        self.pop_size = pop_size
        self.lb = np.asarray(func.lb_vector, dtype=float)
        self.ub = np.asarray(func.ub_vector, dtype=float)
        self.profile: Profile = EXPLOIT

        self.evaluations = 0
        self.z = None
        self.obj = None
        self.best_z = None
        self.best_objective = np.inf
        self.last_step_stats: Dict[str, int] = {}

    def set_profile(self, profile: Profile) -> None:
        self.profile = profile

    def to_physical(self, z: np.ndarray) -> np.ndarray:
        return self.lb + (np.asarray(z) + 1.0) * 0.5 * (self.ub - self.lb)

    def _evaluate(self, z: np.ndarray) -> float:
        self.evaluations += 1
        return float(self.func._evaluate(self.to_physical(z)))

    def initialize(self) -> None:
        self.z = self.rng.uniform(-V_MAX, V_MAX, (self.pop_size, self.lb.shape[0]))
        self.obj = np.array([self._evaluate(row) for row in self.z])
        self._update_best()

    def step(self) -> float:
        """Run one generation (2N evaluations); returns the best-so-far objective."""
        if self.z is None:
            raise RuntimeError("initialize() must be called before step()")
        leaders = self.z[np.argsort(self.obj)[:3]].copy()
        gwo_wins = de_wins = accepted = 0
        for i in range(self.pop_size):
            u_gwo = gwo_candidate(self.z[i], leaders, self.profile.a, self.rng, V_MAX)
            u_de = de_candidate(i, self.z, self.profile.F, self.profile.CR, self.rng, V_MAX)
            f_gwo = self._evaluate(u_gwo)
            f_de = self._evaluate(u_de)
            if f_gwo <= f_de:
                u, f = u_gwo, f_gwo
                gwo_wins += 1
            else:
                u, f = u_de, f_de
                de_wins += 1
            if f <= self.obj[i]:
                self.z[i] = u
                self.obj[i] = f
                accepted += 1
        self.last_step_stats = {"gwo_wins": gwo_wins, "de_wins": de_wins, "accepted": accepted}
        self._update_best()
        return self.best_objective

    def _update_best(self) -> None:
        idx = int(np.argmin(self.obj))
        if self.obj[idx] < self.best_objective:
            self.best_objective = float(self.obj[idx])
            self.best_z = self.z[idx].copy()

    def best_solution(self) -> np.ndarray:
        return self.to_physical(self.best_z)
