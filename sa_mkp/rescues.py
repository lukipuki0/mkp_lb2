"""
rescues.py
──────────
Estrategias de rescate avanzadas para cuando el SA detecta estancamiento.
Contiene las lógicas correspondientes a las versiones V5-V8 del benchmark.
"""

from __future__ import annotations

import random
import numpy as np
from scipy.optimize import linprog

from mkp_core.problem import MKPInstance
from mkp_core.repair import reparar_solucion


def heuristic_rebuild(inst: MKPInstance) -> tuple[list[int], float]:
    """V5: Intensificación Heurística.
    Construye una solución desde cero guiada puramente por la densidad.
    En lugar de ser determinista, añade un poco de ruido para no generar
    siempre la misma solución.
    """
    sol = [0] * inst.n
    # Añadir ruido a las densidades para que no sea siempre igual
    ruido = np.random.uniform(0.9, 1.1, size=inst.n)
    densidades_ruido = inst.density * ruido
    
    indices_desc = np.argsort(densidades_ruido)[::-1]
    
    for idx in indices_desc:
        sol[int(idx)] = 1
        if not inst.es_factible(sol):
            sol[int(idx)] = 0
            
    return sol, inst.evaluar(sol)


def lp_guided_rebuild(sol_actual: list[int], inst: MKPInstance) -> tuple[list[int], float]:
    """V6: Guiada por Relajación Lineal.
    Resuelve LP, obtiene precios sombra, y activa ítems con alto potencial.
    """
    res = linprog(
        -inst.p, 
        A_ub=inst.r, 
        b_ub=inst.b, 
        bounds=(0, 1), 
        method='highs'
    )
    
    duales = res.ineqlin.marginals if res.success and hasattr(res.ineqlin, 'marginals') else np.ones(inst.m)
    
    # Evaluar la "utilidad" de cada ítem basada en LP
    utilidad = []
    for j in range(inst.n):
        costo_dual = sum(inst.r[i][j] * abs(duales[i]) for i in range(inst.m))
        if costo_dual == 0:
            utilidad.append(inst.p[j])
        else:
            utilidad.append(inst.p[j] / costo_dual)
            
    # Flip de bits guiado por utilidad (con probabilidad proporcional)
    utilidad_np = np.array(utilidad)
    prob_activar = utilidad_np / (np.max(utilidad_np) + 1e-9)
    
    nueva_sol = sol_actual.copy()
    for j in range(inst.n):
        if random.random() < prob_activar[j]:
            nueva_sol[j] = 1
        elif random.random() > prob_activar[j]:
            nueva_sol[j] = 0
            
    return reparar_solucion(nueva_sol, inst)


def tabu_lp_rebuild(
    sol_actual: list[int], 
    inst: MKPInstance, 
    tabu_list: list[int]
) -> tuple[list[int], float]:
    """V7: Memoria Tabú + LP.
    Igual que V6 pero prohíbe tocar los bits que están en la lista tabú.
    """
    res = linprog(-inst.p, A_ub=inst.r, b_ub=inst.b, bounds=(0, 1), method='highs')
    duales = res.ineqlin.marginals if res.success and hasattr(res.ineqlin, 'marginals') else np.ones(inst.m)
    
    utilidad = []
    for j in range(inst.n):
        costo_dual = sum(inst.r[i][j] * abs(duales[i]) for i in range(inst.m))
        utilidad.append(inst.p[j] / costo_dual if costo_dual != 0 else inst.p[j])
            
    utilidad_np = np.array(utilidad)
    prob_activar = utilidad_np / (np.max(utilidad_np) + 1e-9)
    
    nueva_sol = sol_actual.copy()
    for j in range(inst.n):
        if j in tabu_list:
            continue  # Tabú: no tocar
        
        if random.random() < prob_activar[j]:
            nueva_sol[j] = 1
        elif random.random() > prob_activar[j]:
            nueva_sol[j] = 0
            
    return reparar_solucion(nueva_sol, inst)


def ruin_and_recreate(sol_actual: list[int], inst: MKPInstance) -> tuple[list[int], float]:
    """V8: Ruin & Recreate Adaptativo.
    Destruye aleatoriamente una gran parte de la solución (50%) 
    y la reconstruye usando eficiencia pura.
    """
    nueva_sol = sol_actual.copy()
    
    # RUIN: poner el 50% de los bits activos en 0
    bits_activos = [i for i, v in enumerate(nueva_sol) if v == 1]
    a_destruir = random.sample(bits_activos, k=len(bits_activos) // 2)
    for idx in a_destruir:
        nueva_sol[idx] = 0
        
    # RECREATE: intentar llenar la mochila usando orden descendente de densidad
    # (como la fase de inserción de reparar_solucion, pero partiendo del ruin)
    for idx in inst.indices_ascendentes[::-1]:
        if nueva_sol[int(idx)] == 0:
            candidato = nueva_sol.copy()
            candidato[int(idx)] = 1
            if inst.es_factible(candidato):
                nueva_sol = candidato
                
    return nueva_sol, inst.evaluar(nueva_sol)
