"""
ga_mkp/operators.py
--------------------
Operadores geneticos para el GA binario aplicado al MKP.

Cruce:
  - crossover_uniform  : por cada gen, elecc aleatoria entre padre A y B
  - crossover_1point   : un punto de corte, intercambio de segmentos
  - crossover_2point   : dos puntos de corte, intercambio del segmento medio

Mutacion:
  - mutate_bitflip     : invierte cada bit con probabilidad p
  - mutate_swap        : intercambia la posicion de dos bits aleatorios

Seleccion:
  - torneo             : k individuos al azar, el de mayor fitness gana

Todas las funciones retornan soluciones que luego pasan por reparar_solucion()
para garantizar factibilidad.
"""

from __future__ import annotations

import random
from typing import List


# ── Seleccion ─────────────────────────────────────────────────────────────────

def torneo(
    poblacion: list[list[int]],
    fitnesses: list[float],
    k: int = 3,
) -> list[int]:
    """Seleccion por torneo: elige k individuos al azar y devuelve el mejor."""
    competidores = random.sample(range(len(poblacion)), min(k, len(poblacion)))
    ganador      = max(competidores, key=lambda i: fitnesses[i])
    return poblacion[ganador].copy()


# ── Cruce ─────────────────────────────────────────────────────────────────────

def crossover_uniform(
    padre_a: list[int],
    padre_b: list[int],
) -> tuple[list[int], list[int]]:
    """Cruce uniforme: cada gen se hereda de A o B con prob 0.5 independientemente."""
    n    = len(padre_a)
    h1   = []
    h2   = []
    for i in range(n):
        if random.random() < 0.5:
            h1.append(padre_a[i]); h2.append(padre_b[i])
        else:
            h1.append(padre_b[i]); h2.append(padre_a[i])
    return h1, h2


def crossover_1point(
    padre_a: list[int],
    padre_b: list[int],
) -> tuple[list[int], list[int]]:
    """Cruce de 1 punto: un punto de corte aleatorio, intercambio de segmentos."""
    n     = len(padre_a)
    punto = random.randint(1, n - 1)
    h1    = padre_a[:punto] + padre_b[punto:]
    h2    = padre_b[:punto] + padre_a[punto:]
    return h1, h2


def crossover_2point(
    padre_a: list[int],
    padre_b: list[int],
) -> tuple[list[int], list[int]]:
    """Cruce de 2 puntos: dos puntos de corte, el segmento central se intercambia."""
    n       = len(padre_a)
    p1, p2  = sorted(random.sample(range(1, n), 2))
    h1      = padre_a[:p1] + padre_b[p1:p2] + padre_a[p2:]
    h2      = padre_b[:p1] + padre_a[p1:p2] + padre_b[p2:]
    return h1, h2


# ── Mutacion ─────────────────────────────────────────────────────────────────

def mutate_bitflip(sol: list[int], rate: float) -> list[int]:
    """Mutacion bit-flip: cada bit se invierte con probabilidad `rate`."""
    return [1 - b if random.random() < rate else b for b in sol]


def mutate_swap(sol: list[int], rate: float) -> list[int]:
    """Mutacion swap: intercambia la posicion de dos genes seleccionados por `rate`.
    
    El parametro `rate` controla la probabilidad de que ocurra el swap.
    Puede llamarse varias veces si rate es muy alto (un swap por call).
    """
    if random.random() < rate:
        n = len(sol)
        i, j = random.sample(range(n), 2)
        sol = sol.copy()
        sol[i], sol[j] = sol[j], sol[i]
    return sol


# ── Registro de operadores ────────────────────────────────────────────────────

CROSSOVER_OPS = {
    "uniform": crossover_uniform,
    "1point" : crossover_1point,
    "2point" : crossover_2point,
}

MUTATION_OPS = {
    "bitflip": mutate_bitflip,
    "swap"   : mutate_swap,
}


def get_crossover(nombre: str):
    if nombre not in CROSSOVER_OPS:
        raise KeyError(f"Cruce '{nombre}' no encontrado. Disponibles: {list(CROSSOVER_OPS)}")
    return CROSSOVER_OPS[nombre]


def get_mutation(nombre: str):
    if nombre not in MUTATION_OPS:
        raise KeyError(f"Mutacion '{nombre}' no encontrado. Disponibles: {list(MUTATION_OPS)}")
    return MUTATION_OPS[nombre]
