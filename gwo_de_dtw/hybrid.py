"""
GWO-DE hybrid over a single shared population for the binary MKP.

Every individual keeps a real vector (where GWO and DE do arithmetic), a
repaired binary solution and its fitness. Each generation, every individual
gets one GWO candidate and one DE/rand/1/bin candidate; the better of the two
replaces it when it is at least as good (real and binary written together).
The operators live in gwo.py and de.py; this module only combines them.

Binarization (see binarization.py) is selectable:
  - "lb2" (default): the real vector gives flip probabilities over the
    individual's current bits; each candidate costs two repaired evaluations
    (L1 and L2). Continuous space [-100, 100]. G1 and G2 are fixed; G3 is
    the active profile's base flip probability.
  - "sigmoid": bits are sampled with P(1) = sigmoid(v) and repaired; each
    candidate costs one evaluation. Continuous space [-6, 6]. Sigmoid does
    not use G1, G2 or G3.

Evaluations are counted in repaired candidates: N for initialization and 4N
per generation with lb2 (2 operators x 2 LB2 candidates), 2N with sigmoid.
They are only reported; the run stops after a number of generations (see
controller.run_hybrid).

Defaults come from config.py.

Spec: odd/tasks/gwo-de-dtw-hybrid.md (S1-S5, S10, S12, S14).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from . import config
from .binarization import (
    LB2_EVALUATIONS, SIGMOID_EVALUATIONS, lb2_binarize, sigmoid_binarize,
)
from .de import de_candidate
from .gwo import gwo_candidate
from .problem import reparar


@dataclass(frozen=True)
class Profile:
    """Joint GWO/DE parameter profile.

    G3 is the LB2 base flip probability for this profile. Sigmoid ignores it.
    """

    name: str
    a: float
    F: float
    CR: float
    G3: float


EXPLOIT = Profile("exploit", a=config.EXPLOIT_A, F=config.EXPLOIT_F,
                  CR=config.EXPLOIT_CR, G3=config.EXPLOIT_G3)
EXPLORE = Profile("explore", a=config.EXPLORE_A, F=config.EXPLORE_F,
                  CR=config.EXPLORE_CR, G3=config.EXPLORE_G3)

BINARIZATIONS = ("lb2", "sigmoid")


class GWODEHybrid:
    """Shared-population GWO-DE with LB2 or sigmoid binarization and greedy selection."""

    def __init__(
        self,
        inst: dict,
        rng: np.random.Generator,
        pop_size: int = config.POP_SIZE,
        v_max: Optional[float] = None,
        G1: float = config.G1,
        G2: float = config.G2,
        binarization: str = config.BINARIZATION,
    ):
        if pop_size < 4:
            raise ValueError("pop_size must be >= 4 (DE/rand/1 needs three other individuals)")
        if binarization not in BINARIZATIONS:
            raise ValueError(f"binarization must be one of {BINARIZATIONS}, got {binarization!r}")
        if v_max is None:
            # lb2 works in [-V_MAX, V_MAX], sigmoid in [-SIGMOID_V_MAX, SIGMOID_V_MAX].
            v_max = config.V_MAX if binarization == "lb2" else config.SIGMOID_V_MAX
        if binarization == "lb2" and G2 >= v_max:
            raise ValueError("G2 must be below v_max (otherwise the LB2 slope changes sign)")
        self.inst = inst
        self.rng = rng
        self.pop_size = pop_size
        self.binarization = binarization
        self.v_max = v_max
        self.G1, self.G2 = G1, G2
        self.profile = EXPLOIT

        self.evaluations = 0
        self.x_real = None
        self.x_bin = None
        self.fit = None
        self.best_solution = None
        self.best_fitness = -np.inf
        self.last_step_stats: Dict[str, int] = {}

    # ------------------------------------------------------------------ setup
    def set_profile(self, profile: Profile) -> None:
        self.profile = profile

    def _evaluate(self, x_real: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, float]:
        """Binarize one candidate; counts its repaired evaluations."""
        if self.binarization == "sigmoid":
            bits, fitness = sigmoid_binarize(x_real, self.inst, self.rng)
            self.evaluations += SIGMOID_EVALUATIONS
            return bits, fitness
        bits, fitness = lb2_binarize(x_real, current, self.inst, self.rng,
                                     self.G1, self.G2, self.profile.G3, self.v_max)
        self.evaluations += LB2_EVALUATIONS
        return bits, fitness

    def initialize(self) -> None:
        n = self.inst["n"]
        self.x_real = self.rng.uniform(-self.v_max, self.v_max, (self.pop_size, n))
        self.x_bin = np.zeros((self.pop_size, n), dtype=int)
        self.fit = np.empty(self.pop_size)
        for i in range(self.pop_size):
            if self.binarization == "sigmoid":
                # Initial bits: sampled from sigmoid(x_real), then repaired (one evaluation).
                self.x_bin[i], self.fit[i] = self._evaluate(self.x_real[i], self.x_bin[i])
                continue
            # Initial bits: positive coordinates switched on (as in hybrid_mkp), then repaired.
            self.x_bin[i], self.fit[i] = reparar((self.x_real[i] > 0).astype(int), self.inst)
            self.evaluations += 1
        self._update_best()

    # ------------------------------------------------------------ generation
    def step(self) -> float:
        """Run one generation (4N evaluations with lb2, 2N with sigmoid).

        Returns the best-so-far fitness.
        """
        if self.x_real is None:
            raise RuntimeError("initialize() must be called before step()")

        order = np.argsort(self.fit)[::-1]
        leaders = self.x_real[order[:3]].copy()  # frozen for this generation (S3)
        gwo_wins = de_wins = accepted = 0

        for i in range(self.pop_size):
            u_gwo = gwo_candidate(self.x_real[i], leaders, self.profile.a, self.rng, self.v_max)
            u_de = de_candidate(i, self.x_real, self.profile.F, self.profile.CR,
                                self.rng, self.v_max)
            b_gwo, f_gwo = self._evaluate(u_gwo, self.x_bin[i])
            b_de, f_de = self._evaluate(u_de, self.x_bin[i])

            if f_gwo >= f_de:
                u, b, f = u_gwo, b_gwo, f_gwo
                gwo_wins += 1
            else:
                u, b, f = u_de, b_de, f_de
                de_wins += 1

            if f >= self.fit[i]:
                self.x_real[i] = u
                self.x_bin[i] = b
                self.fit[i] = f
                accepted += 1

        self.last_step_stats = {"gwo_wins": gwo_wins, "de_wins": de_wins, "accepted": accepted}
        self._update_best()
        return self.best_fitness

    def _update_best(self) -> None:
        idx = int(np.argmax(self.fit))
        if self.fit[idx] > self.best_fitness:
            self.best_fitness = float(self.fit[idx])
            self.best_solution = self.x_bin[idx].copy()

    def get_best(self) -> Tuple[np.ndarray, float]:
        return self.best_solution.copy(), self.best_fitness
