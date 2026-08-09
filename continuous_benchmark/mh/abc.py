"""
continuous_benchmark/mh/abc.py
------------------------------
Artificial Bee Colony (ABC) Algorithm para optimización continua (minimización).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ABCParams:
    pop_size   : int   = 30
    iterations : int   = 1000
    limit      : int | None = None # Si es None, se asigna pop_size * dim / 2


@dataclass
class ABCResult:
    mejor_valor    : float
    mejor_solucion : list[float]
    historial      : list[float] = field(default_factory=list)


def fitness_transform(f_val: float) -> float:
    """Convierte valor objetivo de minimización a fitness positivo para selección por ruleta."""
    if f_val >= 0:
        return 1.0 / (1.0 + f_val)
    else:
        return 1.0 + abs(f_val)


def ejecutar_abc(func, params: ABCParams, seed: int | None = None) -> ABCResult:
    """
    Ejecuta el Algoritmo de Colonia de Abejas Artificiales (ABC).
    """
    rng = np.random.default_rng(seed)
    
    n_dim = func.n_dim
    lb, ub = func.lb, func.ub
    pop_size = params.pop_size
    n_foods = pop_size // 2
    limit = params.limit if params.limit is not None else int(n_foods * n_dim)
    
    # Inicialización de fuentes de alimento
    foods = rng.uniform(lb, ub, size=(n_foods, n_dim))
    obj_vals = np.array([func.func(f) for f in foods])
    trials = np.zeros(n_foods, dtype=int)
    
    gbest_idx = np.argmin(obj_vals)
    gbest_val = float(obj_vals[gbest_idx])
    gbest_pos = foods[gbest_idx].copy()
    
    historial = []
    
    for it in range(params.iterations):
        # 1. Fase Abejas Empleadas
        for i in range(n_foods):
            # Elegir k != i
            k = i
            while k == i:
                k = rng.integers(0, n_foods)
            
            # Elegir dimensión j
            j = rng.integers(0, n_dim)
            phi = rng.uniform(-1.0, 1.0)
            
            v = foods[i].copy()
            v[j] = v[j] + phi * (v[j] - foods[k][j])
            v[j] = np.clip(v[j], lb, ub)
            
            f_v = func.func(v)
            
            if f_v < obj_vals[i]:
                foods[i] = v
                obj_vals[i] = f_v
                trials[i] = 0
            else:
                trials[i] += 1
                
        # 2. Fase Abejas Observadoras
        fits = np.array([fitness_transform(val) for val in obj_vals])
        prob = fits / np.sum(fits)
        prob = prob / np.sum(prob) # Asegurar suma estricta a 1.0 para rng.choice
        
        for _ in range(n_foods):
            # Selección por ruleta
            i = rng.choice(n_foods, p=prob)
            
            k = i
            while k == i:
                k = rng.integers(0, n_foods)
                
            j = rng.integers(0, n_dim)
            phi = rng.uniform(-1.0, 1.0)
            
            v = foods[i].copy()
            v[j] = v[j] + phi * (v[j] - foods[k][j])
            v[j] = np.clip(v[j], lb, ub)
            
            f_v = func.func(v)
            
            if f_v < obj_vals[i]:
                foods[i] = v
                obj_vals[i] = f_v
                trials[i] = 0
            else:
                trials[i] += 1
                
        # Actualizar mejor global
        curr_best_idx = np.argmin(obj_vals)
        if obj_vals[curr_best_idx] < gbest_val:
            gbest_val = float(obj_vals[curr_best_idx])
            gbest_pos = foods[curr_best_idx].copy()
            
        # 3. Fase Abeja Exploradora (Scout)
        max_trial_idx = np.argmax(trials)
        if trials[max_trial_idx] > limit:
            foods[max_trial_idx] = rng.uniform(lb, ub, size=n_dim)
            obj_vals[max_trial_idx] = func.func(foods[max_trial_idx])
            trials[max_trial_idx] = 0
            
            if obj_vals[max_trial_idx] < gbest_val:
                gbest_val = float(obj_vals[max_trial_idx])
                gbest_pos = foods[max_trial_idx].copy()
                
        historial.append(gbest_val)
        
    return ABCResult(
        mejor_valor = gbest_val,
        mejor_solucion = gbest_pos.tolist(),
        historial = historial,
    )
