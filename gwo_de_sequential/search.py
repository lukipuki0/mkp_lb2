"""Shared population: full greedy GWO sweep, then full greedy in-place DE sweep."""

import numpy as np

from . import config
from .binarize import lb2_binarize, sigmoid_binarize
from .operators import de_candidate, gwo_candidate
from .problem import reparar
from .profiles import Profile

BINARIZATIONS = ("lb2", "sigmoid")


class SequentialSearch:
    """One shared population. There is no competitive GWO-versus-DE branch."""

    def __init__(self, inst, rng, pop_size=config.POP_SIZE, v_max=None,
                 G1=config.G1, G2=config.G2, binarization=config.BINARIZATION):
        if pop_size < 4:
            raise ValueError("pop_size must be >= 4 (DE/rand/1 needs three other individuals)")
        if binarization not in BINARIZATIONS:
            raise ValueError(f"binarization must be one of {BINARIZATIONS}, got {binarization!r}")
        if v_max is None:
            v_max = config.V_MAX if binarization == "lb2" else config.SIGMOID_V_MAX
        if binarization == "lb2" and G2 >= v_max:
            raise ValueError("G2 must be below v_max (otherwise the LB2 slope changes sign)")
        self.inst = inst
        self.rng = rng
        self.pop_size = pop_size
        self.binarization = binarization
        self.v_max = v_max
        self.G1 = G1
        self.G2 = G2
        self.profile = Profile("exploit", config.EXPLOIT_A, config.EXPLOIT_F,
                               config.EXPLOIT_CR, config.EXPLOIT_G3)
        self.evaluations = 0
        self.x_real = None
        self.x_bin = None
        self.fit = None
        self.best_solution = None
        self.best_fitness = -np.inf
        self.last_step_stats = {}

    def set_profile(self, profile: Profile) -> None:
        self.profile = profile

    def _evaluate(self, real, current):
        if self.binarization == "sigmoid":
            bits, fitness = sigmoid_binarize(real, self.inst, self.rng)
            self.evaluations += 1
            return bits, float(fitness)
        bits, fitness = lb2_binarize(
            real, current, self.inst, self.rng, self.G1, self.G2, self.profile.G3, self.v_max)
        self.evaluations += 2
        return bits, float(fitness)

    def initialize(self) -> None:
        width = self.inst["n"]
        self.x_real = self.rng.uniform(-self.v_max, self.v_max, (self.pop_size, width))
        self.x_bin = np.zeros((self.pop_size, width), dtype=int)
        self.fit = np.empty(self.pop_size)
        for row in range(self.pop_size):
            if self.binarization == "sigmoid":
                self.x_bin[row], self.fit[row] = self._evaluate(self.x_real[row], self.x_bin[row])
                continue
            self.x_bin[row], self.fit[row] = reparar((self.x_real[row] > 0).astype(int), self.inst)
            self.evaluations += 1
        self._update_best()

    def step(self) -> float:
        """GWO sweep on frozen leaders, then DE sweep over the population that results."""
        if self.x_real is None:
            raise RuntimeError("initialize() must be called before step()")
        order = np.argsort(self.fit)[::-1]
        leaders = self.x_real[order[:3]].copy()
        gwo_accepted = 0
        for row in range(self.pop_size):
            candidate = gwo_candidate(self.x_real[row], leaders.copy(), self.profile.a,
                                      self.rng, self.v_max)
            bits, fitness = self._evaluate(candidate, self.x_bin[row])
            if fitness >= self.fit[row]:
                self.x_real[row] = candidate
                self.x_bin[row] = bits
                self.fit[row] = fitness
                gwo_accepted += 1
        self._update_best()

        de_accepted = 0
        for row in range(self.pop_size):
            candidate = de_candidate(row, self.x_real, self.profile.F, self.profile.CR,
                                     self.rng, self.v_max)
            bits, fitness = self._evaluate(candidate, self.x_bin[row])
            if fitness >= self.fit[row]:
                self.x_real[row] = candidate
                self.x_bin[row] = bits
                self.fit[row] = fitness
                de_accepted += 1
        self._update_best()
        self.last_step_stats = {
            "gwo_accepted": gwo_accepted,
            "de_accepted": de_accepted,
            "accepted_total": gwo_accepted + de_accepted,
        }
        return float(self.best_fitness)

    def _update_best(self) -> None:
        winner = int(np.argmax(self.fit))
        if self.fit[winner] > self.best_fitness:
            self.best_fitness = float(self.fit[winner])
            self.best_solution = self.x_bin[winner].copy()
