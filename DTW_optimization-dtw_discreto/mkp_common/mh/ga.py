"""
Algoritmo Genético binario para MKP.
Implementa BaseMH para integrarse con el runner + DTW.

Operadores: torneo, crossover uniforme, mutación bit-flip.
"""

from typing import Tuple

import numpy as np

from ..base import BaseMH
from ..problem import reparar


class GeneticAlgorithm(BaseMH):
    """
    GA Binario con 2 modos de parámetros:
        EXPLOIT: cx_rate=0.9,  mut_rate=0.01 → refinar soluciones
        EXPLORE: cx_rate=0.6,  mut_rate=0.15 → diversificar población
    """

    # --- Parámetros por modo ---
    CX_EXPLOIT, MUT_EXPLOIT = 0.9, 0.01
    CX_EXPLORE, MUT_EXPLORE = 0.6, 0.15

    def __init__(self, inst: dict, rng: np.random.Generator, **kwargs):
        super().__init__(inst, rng, **kwargs)
        self.num_particulas = kwargs.get("num_particulas", 20)  # tamaño población
        self.elitism = 2
        self.tournament_size = 3

        # Params activos (inician en exploit)
        self.crossover_rate = self.CX_EXPLOIT
        self.mutation_rate = self.MUT_EXPLOIT

        # Estado
        self.poblacion = None
        self.fitness_pop = None
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

        self.fitness_pop = np.array(self.fitness_pop)
        idx = np.argmax(self.fitness_pop)
        self.gbest = self.poblacion[idx].copy()
        self.gbest_fitness = self.fitness_pop[idx]

    # --- Operadores genéticos ---

    def _tournament(self) -> np.ndarray:
        """Selección por torneo."""
        indices = self.rng.choice(self.num_particulas, self.tournament_size, replace=False)
        best = indices[np.argmax(self.fitness_pop[indices])]
        return self.poblacion[best].copy()

    def _uniform_crossover(self, p1: np.ndarray, p2: np.ndarray):
        """Crossover uniforme: cada bit se hereda de un padre al azar."""
        mask = self.rng.random(len(p1)) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    def _mutate(self, sol: np.ndarray) -> np.ndarray:
        """Mutación bit-flip: cada bit se invierte con prob mutation_rate."""
        mask = self.rng.random(len(sol)) < self.mutation_rate
        sol = sol.copy()
        sol[mask] = 1 - sol[mask]
        return sol

    def step(self) -> float:
        # Elitismo: mantener los mejores
        elite_idx = np.argsort(self.fitness_pop)[-self.elitism:]
        nueva_pob = [self.poblacion[i].copy() for i in elite_idx]
        nuevo_fit = [float(self.fitness_pop[i]) for i in elite_idx]

        # Reproducción hasta llenar la población
        while len(nueva_pob) < self.num_particulas:
            p1 = self._tournament()
            p2 = self._tournament()

            if self.rng.random() < self.crossover_rate:
                c1, c2 = self._uniform_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = self._mutate(c1)
            c2 = self._mutate(c2)

            c1, f1 = reparar(c1, self.inst)
            c2, f2 = reparar(c2, self.inst)

            nueva_pob.append(c1)
            nuevo_fit.append(f1)
            if len(nueva_pob) < self.num_particulas:
                nueva_pob.append(c2)
                nuevo_fit.append(f2)

        self.poblacion = nueva_pob[:self.num_particulas]
        self.fitness_pop = np.array(nuevo_fit[:self.num_particulas])

        # Actualizar mejor global
        idx = np.argmax(self.fitness_pop)
        if self.fitness_pop[idx] > self.gbest_fitness:
            self.gbest_fitness = float(self.fitness_pop[idx])
            self.gbest = self.poblacion[idx].copy()

        return self.gbest_fitness

    def adapt(self, fire: bool) -> None:
        if fire and self.mode != "explore":
            self.mode = "explore"
            self.crossover_rate = self.CX_EXPLORE
            self.mutation_rate = self.MUT_EXPLORE
        elif not fire and self.mode == "explore":
            self.mode = "exploit"
            self.crossover_rate = self.CX_EXPLOIT
            self.mutation_rate = self.MUT_EXPLOIT

    def get_best(self) -> Tuple[np.ndarray, float]:
        return self.gbest.copy(), self.gbest_fitness