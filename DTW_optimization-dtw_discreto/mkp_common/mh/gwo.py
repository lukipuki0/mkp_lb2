"""
Binary Grey Wolf Optimizer (BGWO) para MKP.
Implementa BaseMH para integrarse con el runner + DTW.

Basado en Mirjalili et al. (2014), binarizado con sigmoid.
Jerarquía: Alpha (mejor), Beta (2do), Delta (3ro), Omega (resto).
"""

from typing import Tuple

import numpy as np

from ..base import BaseMH
from ..problem import reparar


class BinaryGWO(BaseMH):
    """
    GWO Binario con 2 modos de parámetros:
        EXPLOIT: a=0.5 → lobos convergen hacia alpha/beta/delta
        EXPLORE: a=2.0 → lobos exploran ampliamente

    En GWO estándar, 'a' decae linealmente de 2 a 0. Acá lo fijamos
    por modo y dejamos que el DTW controle cuándo explorar/explotar.
    """

    # --- Parámetro 'a' por modo ---
    A_EXPLOIT = 0.5
    A_EXPLORE = 2.0

    def __init__(self, inst: dict, rng: np.random.Generator, **kwargs):
        super().__init__(inst, rng, **kwargs)
        self.num_particulas = kwargs.get("num_particulas", 20)

        # Param activo
        self.a = self.A_EXPLOIT

        # Estado
        self.poblacion = None
        self.fitness_pop = None
        self.alpha = None
        self.beta = None
        self.delta = None
        self.gbest = None
        self.gbest_fitness = -np.inf

    def initialize(self) -> None:
        n = self.inst["n"]
        self.poblacion = []
        self.fitness_pop = []

        for _ in range(self.num_particulas):
            sol = self.rng.integers(0, 2, n)
            sol, fit = reparar(sol, self.inst)
            self.poblacion.append(sol)
            self.fitness_pop.append(fit)

        self.fitness_pop = np.array(self.fitness_pop, dtype=float)
        self._update_leaders()

    def _update_leaders(self) -> None:
        """Actualiza alpha, beta, delta (top 3) y gbest (mejor histórico)."""
        sorted_idx = np.argsort(self.fitness_pop)[::-1]  # mejor a peor

        self.alpha = self.poblacion[sorted_idx[0]].copy()
        self.beta = self.poblacion[sorted_idx[1]].copy()
        self.delta = self.poblacion[sorted_idx[2]].copy()

        # Mejor histórico (no solo el actual)
        if self.fitness_pop[sorted_idx[0]] > self.gbest_fitness:
            self.gbest_fitness = float(self.fitness_pop[sorted_idx[0]])
            self.gbest = self.alpha.copy()

    def step(self) -> float:
        n = self.inst["n"]
        a = self.a

        for i in range(self.num_particulas):
            x = self.poblacion[i].astype(float)

            # Vectores hacia Alpha
            r1, r2 = self.rng.random(n), self.rng.random(n)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * self.alpha - x)
            X1 = self.alpha - A1 * D_alpha

            # Vectores hacia Beta
            r1, r2 = self.rng.random(n), self.rng.random(n)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * self.beta - x)
            X2 = self.beta - A2 * D_beta

            # Vectores hacia Delta
            r1, r2 = self.rng.random(n), self.rng.random(n)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * self.delta - x)
            X3 = self.delta - A3 * D_delta

            # Posición promedio (continua)
            X_new = (X1 + X2 + X3) / 3.0

            # Binarización por sigmoid
            prob = 1.0 / (1.0 + np.exp(-np.clip(X_new, -10, 10)))
            new_sol = (self.rng.random(n) < prob).astype(int)

            # Reparar y evaluar
            new_sol, fit = reparar(new_sol, self.inst)
            self.poblacion[i] = new_sol
            self.fitness_pop[i] = fit

        self._update_leaders()
        return self.gbest_fitness

    def adapt(self, fire: bool) -> None:
        if fire and self.mode != "explore":
            self.mode = "explore"
            self.a = self.A_EXPLORE
        elif not fire and self.mode == "explore":
            self.mode = "exploit"
            self.a = self.A_EXPLOIT

    def get_best(self) -> Tuple[np.ndarray, float]:
        return self.gbest.copy(), self.gbest_fitness
