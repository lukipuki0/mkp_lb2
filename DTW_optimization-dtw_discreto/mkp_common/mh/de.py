"""
Binary Differential Evolution (DE/rand/1/bin) para MKP.
Implementa BaseMH para integrarse con el runner + DTW.

Basado en Storn & Price (1997). Binarizado con sigmoid transfer function.
Operadores: mutación diferencial, crossover binomial, selección greedy 1-a-1.
"""

from typing import Tuple

import numpy as np

from ..base import BaseMH
from ..problem import reparar


class BinaryDE(BaseMH):
    """
    DE/rand/1/bin binarizado con 2 modos de parámetros:
        EXPLOIT: F=0.5, CR=0.9 → perturbación suave, mucho crossover (refinar)
        EXPLORE: F=0.9, CR=0.3 → perturbación agresiva, poco crossover (diversificar)

    Cada individuo mantiene representación dual:
        - continua (R^n): donde opera la aritmética diferencial
        - binaria ({0,1}^n): donde se evalúa el fitness MKP
    """

    # --- Parámetros por modo ---
    F_EXPLOIT, CR_EXPLOIT = 0.5, 0.9
    F_EXPLORE, CR_EXPLORE = 0.9, 0.3

    def __init__(self, inst: dict, rng: np.random.Generator, **kwargs):
        super().__init__(inst, rng, **kwargs)
        self.num_particulas = kwargs.get("num_particulas", 20)
        self.v_max = 6.0  # clamp para sigmoid

        # Params activos (inician en exploit)
        self.F = self.F_EXPLOIT
        self.CR = self.CR_EXPLOIT

        # Estado dual: continuo + binario
        self.poblacion_cont = None   # R^n — aritmética DE
        self.poblacion_bin = None    # {0,1}^n — fitness MKP
        self.fitness_pop = None
        self.gbest = None
        self.gbest_fitness = -np.inf

    def initialize(self) -> None:
        n = self.inst["n"]
        NP = self.num_particulas

        # Inicializar continuo en [-v_max, v_max]
        self.poblacion_cont = self.rng.uniform(
            -self.v_max, self.v_max, (NP, n)
        )
        self.poblacion_bin = np.zeros((NP, n), dtype=int)
        self.fitness_pop = np.full(NP, -np.inf)

        # Binarizar, reparar y evaluar cada individuo
        for i in range(NP):
            prob = 1.0 / (1.0 + np.exp(-self.poblacion_cont[i]))
            sol = (self.rng.random(n) < prob).astype(int)
            sol, fit = reparar(sol, self.inst)
            self.poblacion_bin[i] = sol
            self.fitness_pop[i] = fit

        idx = np.argmax(self.fitness_pop)
        self.gbest = self.poblacion_bin[idx].copy()
        self.gbest_fitness = float(self.fitness_pop[idx])

    def step(self) -> float:
        n = self.inst["n"]
        NP = self.num_particulas

        for i in range(NP):
            # --- MUTACIÓN DE/rand/1 ---
            # Elegir r1, r2, r3 distintos y ≠ i
            candidates = [j for j in range(NP) if j != i]
            r1, r2, r3 = self.rng.choice(candidates, 3, replace=False)

            v = (
                self.poblacion_cont[r1]
                + self.F * (self.poblacion_cont[r2] - self.poblacion_cont[r3])
            )
            v = np.clip(v, -self.v_max, self.v_max)

            # --- CROSSOVER binomial ---
            j_rand = self.rng.integers(0, n)
            mask = self.rng.random(n) < self.CR
            mask[j_rand] = True  # al menos 1 dimensión del mutante

            u = np.where(mask, v, self.poblacion_cont[i])

            # --- BINARIZACIÓN ---
            prob = 1.0 / (1.0 + np.exp(-u))
            trial_bin = (self.rng.random(n) < prob).astype(int)
            trial_bin, trial_fit = reparar(trial_bin, self.inst)

            # --- SELECCIÓN greedy 1-a-1 ---
            if trial_fit >= self.fitness_pop[i]:
                self.poblacion_cont[i] = u
                self.poblacion_bin[i] = trial_bin
                self.fitness_pop[i] = trial_fit

        # Actualizar mejor global
        idx = np.argmax(self.fitness_pop)
        if self.fitness_pop[idx] > self.gbest_fitness:
            self.gbest_fitness = float(self.fitness_pop[idx])
            self.gbest = self.poblacion_bin[idx].copy()

        return self.gbest_fitness

    def adapt(self, fire: bool) -> None:
        if fire and self.mode != "explore":
            self.mode = "explore"
            self.F = self.F_EXPLORE
            self.CR = self.CR_EXPLORE
        elif not fire and self.mode == "explore":
            self.mode = "exploit"
            self.F = self.F_EXPLOIT
            self.CR = self.CR_EXPLOIT

    def get_best(self) -> Tuple[np.ndarray, float]:
        return self.gbest.copy(), self.gbest_fitness
