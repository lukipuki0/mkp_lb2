"""
problem.py
──────────
Define el problema MKP (Multidimensional Knapsack Problem) como un objeto
autocontenido.  Encapsula los datos de la instancia y calcula las densidades
de los ítems —necesarias para la función de reparación.

Uso típico:
    from mkp_core.problem import MKPInstance
    inst = MKPInstance.from_dict(data)   # data viene de data_loader
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MKPInstance:
    """Encapsula una instancia MKP y precalcula las densidades.

    Atributos públicos
    ------------------
    n : int
        Número de ítems.
    m : int
        Número de restricciones (dimensiones de la mochila).
    valor_optimo : float
        Mejor valor conocido (para calcular el gap al final).
    p : NDArray[float]  shape (n,)
        Vector de ganancias.
    r : NDArray[float]  shape (m, n)
        Matriz de pesos: r[i, j] = peso del ítem j en la restricción i.
    b : NDArray[float]  shape (m,)
        Vector de capacidades de cada restricción.
    density : NDArray[float]  shape (n,)
        Densidad mínima de cada ítem (ganancia / peso) usada en reparación.
    indices_ascendentes : NDArray[int]  shape (n,)
        Ítems ordenados de menor a mayor densidad.
    """

    def __init__(
        self,
        n: int,
        m: int,
        valor_optimo: float,
        p: NDArray,
        r: NDArray,
        b: NDArray,
    ) -> None:
        self.n = n
        self.m = m
        self.valor_optimo = valor_optimo
        self.p = p
        self.r = r
        self.b = b

        self.density, self.indices_ascendentes = self._calcular_densidades()

    # ── Construcción ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "MKPInstance":
        """Construye una instancia a partir del dict devuelto por data_loader."""
        return cls(
            n=data["n"],
            m=data["m"],
            valor_optimo=data["valor_optimo"],
            p=data["p"],
            r=data["r"],
            b=data["b"],
        )

    # ── Cálculo de densidades ─────────────────────────────────────────────────

    def _calcular_densidades(self) -> tuple[NDArray, NDArray]:
        """Calcula la densidad mínima de cada ítem y su orden ascendente.

        Densidad de ítem j en restricción i: p[j] / r[i, j]
        Densidad global del ítem j          : min_i(p[j] / r[i, j])

        Los infinitos (división por 0) se sustituyen por 10× el máximo finito.
        """
        # density_per_bag[i, j] = p[j] / r[i, j]
        density_per_bag = np.array([self.p / self.r[i] for i in range(self.m)])

        finite_max = np.max(density_per_bag[np.isfinite(density_per_bag)])
        density_per_bag[np.isinf(density_per_bag)] = finite_max * 10

        density = np.min(density_per_bag, axis=0)
        indices_ascendentes = np.argsort(density)

        return density, indices_ascendentes

    # ── Evaluación ────────────────────────────────────────────────────────────

    def es_factible(self, solucion: list[int]) -> bool:
        """Devuelve True si *solucion* respeta todas las restricciones."""
        return bool(np.all(np.dot(solucion, self.r.T) <= self.b))

    def evaluar(self, solucion: list[int]) -> float:
        """Devuelve el valor total de *solucion* (ganancia acumulada)."""
        return float(np.sum(np.array(solucion) * self.p))

    def gap(self, valor: float) -> float:
        """Gap relativo respecto al óptimo conocido (en %)."""
        return (self.valor_optimo - valor) / self.valor_optimo * 100

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MKPInstance(n={self.n}, m={self.m}, "
            f"valor_optimo={self.valor_optimo})"
        )
