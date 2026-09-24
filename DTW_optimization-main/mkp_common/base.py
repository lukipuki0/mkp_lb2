"""
Clase base abstracta para metaheurísticas poblacionales.

Para agregar una nueva MH:
1. Crear archivo (ej: ga.py)
2. Heredar de BaseMH
3. Implementar initialize(), step(), adapt(), get_best()
4. Importar en __init__.py
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseMH(ABC):
    """
    Interfaz que toda metaheurística debe implementar.

    El runner orquesta así:
        mh.initialize()
        for each iteration:
            fitness = mh.step()
            mh.adapt(fire)
    """

    def __init__(self, inst: dict, rng: np.random.Generator, **kwargs):
        self.inst = inst
        self.rng = rng
        self.mode = "exploit"

    @abstractmethod
    def initialize(self) -> None:
        """Crear población inicial, evaluar y definir mejor global."""
        ...

    @abstractmethod
    def step(self) -> float:
        """Ejecutar UNA iteración. Retorna gbest_fitness actual."""
        ...

    @abstractmethod
    def adapt(self, fire: bool) -> None:
        """
        Adaptar parámetros según señal del DTW.
        fire=True  → estancamiento detectado, forzar exploración.
        fire=False → progreso recuperado, volver a explotación.
        """
        ...

    def adapt_continuous(self, intensity: float) -> None:
        """
        Adaptar parámetros de forma continua.
        intensity=0.0 → parámetros de explotación.
        intensity=1.0 → parámetros de exploración.
        0 < intensity < 1 → interpolación lineal.

        Por defecto no hace nada; cada MH debe sobrescribirlo.
        """
        pass

    @abstractmethod
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Retorna (mejor_solucion, mejor_fitness)."""
        ...
