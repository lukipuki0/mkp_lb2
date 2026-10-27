"""
Binary PSO estándar (Kennedy & Eberhart 1997) con sigmoid.
Implementa BaseMH para integrarse con el runner + DTW.
"""

from typing import Tuple

import numpy as np

from ..base import BaseMH
from ..problem import reparar


class BinaryPSO(BaseMH):
    """
    PSO Binario con 2 modos de parámetros:
        EXPLOIT (Clerc & Kennedy 2002): w=0.729, c1=1.49445, c2=1.49445
        EXPLORE (forzar exploración):   w=0.9,   c1=2.5,     c2=0.5
    """

    # --- Parámetros por modo ---
    W_EXPLOIT, C1_EXPLOIT, C2_EXPLOIT = 0.729, 1.49445, 1.49445
    W_EXPLORE, C1_EXPLORE, C2_EXPLORE = 0.9, 2.5, 0.5

    def __init__(self, inst: dict, rng: np.random.Generator, **kwargs):
        super().__init__(inst, rng, **kwargs)
        self.num_particulas = kwargs.get("num_particulas", 20)
        self.v_max = 6.0

        # Params activos (inician en exploit)
        self.w = self.W_EXPLOIT
        self.c1 = self.C1_EXPLOIT
        self.c2 = self.C2_EXPLOIT

        # Estado (se inicializa en initialize())
        self.poblacion = None
        self.velocidades = None
        self.pbest = None
        self.pbest_fitness = None
        self.gbest = None
        self.gbest_fitness = -np.inf

    def initialize(self) -> None:
        n = self.inst["n"]
        self.poblacion = np.zeros((self.num_particulas, n), dtype=int)
        self.velocidades = self.rng.uniform(
            -self.v_max, self.v_max, (self.num_particulas, n)
        )
        self.pbest = np.zeros((self.num_particulas, n), dtype=int)
        self.pbest_fitness = np.full(self.num_particulas, -np.inf)

        for i in range(self.num_particulas):
            sol = self.rng.integers(0, 2, n)
            sol, fit = reparar(sol, self.inst)
            self.poblacion[i] = sol
            if fit > self.pbest_fitness[i]:
                self.pbest_fitness[i] = fit
                self.pbest[i] = sol.copy()

        idx = np.argmax(self.pbest_fitness)
        self.gbest = self.pbest[idx].copy()
        self.gbest_fitness = self.pbest_fitness[idx]

    def step(self) -> float:
        n = self.inst["n"]

        for i in range(self.num_particulas):
            r1 = self.rng.random(n)
            r2 = self.rng.random(n)

            self.velocidades[i] = (
                self.w * self.velocidades[i]
                + self.c1 * r1 * (self.pbest[i] - self.poblacion[i])
                + self.c2 * r2 * (self.gbest - self.poblacion[i])
            )
            self.velocidades[i] = np.clip(self.velocidades[i], -self.v_max, self.v_max)

            prob = 1.0 / (1.0 + np.exp(-self.velocidades[i]))
            self.poblacion[i] = (self.rng.random(n) < prob).astype(int)

            self.poblacion[i], fit = reparar(self.poblacion[i], self.inst)

            if fit > self.pbest_fitness[i]:
                self.pbest_fitness[i] = fit
                self.pbest[i] = self.poblacion[i].copy()

        idx = np.argmax(self.pbest_fitness)
        if self.pbest_fitness[idx] > self.gbest_fitness:
            self.gbest_fitness = self.pbest_fitness[idx]
            self.gbest = self.pbest[idx].copy()

        return self.gbest_fitness

    def adapt(self, fire: bool) -> None:
        if fire and self.mode != "explore":
            self.mode = "explore"
            self.w = self.W_EXPLORE
            self.c1 = self.C1_EXPLORE
            self.c2 = self.C2_EXPLORE
        elif not fire and self.mode == "explore":
            self.mode = "exploit"
            self.w = self.W_EXPLOIT
            self.c1 = self.C1_EXPLOIT
            self.c2 = self.C2_EXPLOIT

    def get_best(self) -> Tuple[np.ndarray, float]:
        return self.gbest.copy(), self.gbest_fitness
