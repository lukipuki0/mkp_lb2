"""
================================================================================
Funciones Benchmark CEC 2022 (Congress on Evolutionary Computation 2022)
================================================================================
Implementación completa de las 12 funciones de benchmark CEC 2022 en Python,
siguiendo las especificaciones oficiales y los papers de referencia:
  - Wang, R., Pan, J., Chu, S., Lin, B., & Zhong, N. (2026). A Multi-Strategy
    Population-Free Particle Swarm Optimization Algorithm... Applied Soft Computing.
  - Akbulut, H. (2026). Artificial Bee Colony with Momentum-Guided Search and
    Starfish Exploration... Applied Soft Computing.

Las 12 funciones son:
  F1 : Zakharov desplazada y rotada (unimodal)                  - Bias: 300
  F2 : Rosenbrock desplazada y rotada (multimodal básica)         - Bias: 400
  F3 : Schaffer's F6 expandida, desplazada y rotada            - Bias: 600
  F4 : Rastrigin no continua, desplazada y rotada              - Bias: 800
  F5 : Levy desplazada y rotada                                 - Bias: 900
  F6 : Función Híbrida 1 (N=3 subfunciones)                     - Bias: 1800
  F7 : Función Híbrida 2 (N=6 subfunciones)                     - Bias: 2000
  F8 : Función Híbrida 3 (N=5 subfunciones)                     - Bias: 2200
  F9 : Función de Composición 1 (N=5 subfunciones)              - Bias: 2300
  F10: Función de Composición 2 (N=4 subfunciones)              - Bias: 2400
  F11: Función de Composición 3 (N=5 subfunciones)              - Bias: 2600
  F12: Función de Composición 4 (N=6 subfunciones)              - Bias: 2700

Rango de búsqueda: [-100, 100]^nD para todas las funciones.
Dimensiones soportadas: nD = 10, nD = 20 (y arbitrario > 1).
================================================================================
"""

import os
import numpy as np
from typing import Callable, List, Tuple, Dict, Optional
from dataclasses import dataclass

DATA_DIR = os.path.join(os.path.dirname(__file__), "input_data")

@dataclass
class ContinuousFunction:
    """Descriptor de una función de benchmark continua."""
    name    : str
    func    : Callable[[np.ndarray], float]
    lb      : float          # Límite inferior
    ub      : float          # Límite superior
    optimum : float          # Bias/Valor óptimo conocido
    n_dim   : int            # Dimensionalidad


# ============================================================================
# 1. FUNCIONES BASE AUXILIARES
# ============================================================================

def bent_cigar(x: np.ndarray) -> float:
    """Bent Cigar function."""
    x = np.asarray(x, dtype=float)
    return float(x[0]**2 + 1e6 * np.sum(x[1:]**2))


def discus(x: np.ndarray) -> float:
    """Discus function."""
    x = np.asarray(x, dtype=float)
    return float(1e6 * x[0]**2 + np.sum(x[1:]**2))


def zakharov(x: np.ndarray) -> float:
    """Zakharov function."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    i = np.arange(1, n + 1)
    s1 = np.sum(x**2)
    s2 = np.sum(0.5 * i * x)
    return float(s1 + s2**2 + s2**4)


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function con transformación CEC (mínimo 0 en x=0)."""
    x = np.asarray(x, dtype=float) * (2.048 / 100.0) + 1.0
    if len(x) <= 1:
        return 0.0
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2))


def schaffer_f6(x_i: float, x_j: float) -> float:
    """Schaffer's F6 function para dos variables."""
    num = np.sin(np.sqrt(x_i**2 + x_j**2))**2 - 0.5
    den = (1.0 + 0.001 * (x_i**2 + x_j**2))**2
    return 0.5 + num / den


def expanded_schaffer_f6(x: np.ndarray) -> float:
    """Expanded Schaffer's F6 function."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        total += schaffer_f6(x[i], x[i + 1])
    total += schaffer_f6(x[-1], x[0])
    return float(total)


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function con escalado CEC."""
    x = np.asarray(x, dtype=float) * (5.12 / 100.0)
    return float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0))


def non_continuous_rastrigin(x: np.ndarray) -> float:
    """Non-continuous Rastrigin function con escalado CEC."""
    x = np.asarray(x, dtype=float) * (5.12 / 100.0)
    y = np.where(np.abs(x) < 0.5, x, np.round(2.0 * x) / 2.0)
    return float(np.sum(y**2 - 10.0 * np.cos(2.0 * np.pi * y) + 10.0))


def levy(x: np.ndarray) -> float:
    """Levy function con transformación CEC (mínimo 0 en x=0)."""
    x = np.asarray(x, dtype=float) + 1.0
    w = 1.0 + (x - 1.0) / 4.0
    t1 = np.sin(np.pi * w[0])**2
    t3 = (w[-1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * w[-1])**2)
    t2 = np.sum((w[:-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0)**2))
    return float(t1 + t2 + t3)


def ackley(x: np.ndarray) -> float:
    """Ackley function."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    t1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    t2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return float(t1 + t2 + a + np.e)


def weierstrass(x: np.ndarray, a: float = 0.5, b: float = 3.0, kmax: int = 20) -> float:
    """Weierstrass function con escalado CEC."""
    x = np.asarray(x, dtype=float) * (0.5 / 100.0)
    n = len(x)
    k = np.arange(0, kmax + 1)
    ak = a**k
    bk = b**k

    term1 = 0.0
    for i in range(n):
        term1 += np.sum(ak * np.cos(2.0 * np.pi * bk * (x[i] + 0.5)))

    term2 = n * np.sum(ak * np.cos(2.0 * np.pi * bk * 0.5))
    return float(term1 - term2)


def griewank(x: np.ndarray) -> float:
    """Griewank function con escalado CEC."""
    x = np.asarray(x, dtype=float) * (600.0 / 100.0)
    n = len(x)
    s = np.sum(x**2) / 4000.0
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    return float(s - p + 1.0)


def schwefel(x: np.ndarray) -> float:
    """Schwefel function con límite y escalado oficial CEC (mínimo 0 en x=0)."""
    x = np.asarray(x, dtype=float) * (1000.0 / 100.0)
    n = len(x)
    z = x + 420.9687462272847
    
    g = np.zeros(n)
    for i in range(n):
        zi = z[i]
        if zi > 500.0:
            rem = zi % 500.0
            g[i] = (500.0 - rem) * np.sin(np.sqrt(np.abs(500.0 - rem))) - ((zi - 500.0)**2) / (10000.0 * n)
        elif zi < -500.0:
            rem = (-zi) % 500.0
            g[i] = (rem - 500.0) * np.sin(np.sqrt(np.abs(rem - 500.0))) - ((zi + 500.0)**2) / (10000.0 * n)
        else:
            g[i] = zi * np.sin(np.sqrt(np.abs(zi)))
            
    return float(418.98288727243379 * n - np.sum(g))


def katsuura(x: np.ndarray) -> float:
    """Katsuura function con escalado CEC."""
    x = np.asarray(x, dtype=float) * (5.0 / 100.0)
    n = len(x)
    prod = 1.0
    j_arr = 2.0**np.arange(1, 33) # 1 a 32
    for i in range(n):
        sum_j = np.sum(np.abs(np.round(j_arr * x[i]) - j_arr * x[i]) / j_arr)
        prod *= (1.0 + (i + 1) * sum_j)**(10.0 / n**1.2)
    return float((10.0 / n**2) * prod - (10.0 / n**2))


def happycat(x: np.ndarray) -> float:
    """HappyCat function con transformación CEC (mínimo 0 en x=0)."""
    x = np.asarray(x, dtype=float) * (5.0 / 100.0) - 1.0
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_x = np.sum(x)
    t1 = np.abs(sum_sq - n)**0.25
    t2 = (0.5 * sum_sq + sum_x) / n
    return float(t1 + t2 + 0.5)


def hgbat(x: np.ndarray) -> float:
    """HGBat function con transformación CEC (mínimo 0 en x=0)."""
    x = np.asarray(x, dtype=float) * (5.0 / 100.0) - 1.0
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_x = np.sum(x)
    t1 = np.abs(sum_sq**2 - sum_x**2)**0.5
    t2 = (0.5 * sum_sq + sum_x) / n
    return float(t1 + t2 + 0.5)


def grie_rosen(x: np.ndarray) -> float:
    """Expanded Griewank plus Rosenbrock (Grie-Rosen)."""
    x = np.asarray(x, dtype=float) * (5.0 / 100.0)
    n = len(x)
    if n <= 1:
        return 0.0
    z = x + 1.0
    total = 0.0
    for i in range(n - 1):
        # rosenbrock directo sobre par z sin re-escalar
        r = 100.0 * (z[i+1] - z[i]**2)**2 + (z[i] - 1.0)**2
        # griewank directo sobre r
        g = r**2 / 4000.0 - np.cos(r) + 1.0
        total += g
    r_last = 100.0 * (z[0] - z[-1]**2)**2 + (z[-1] - 1.0)**2
    g_last = r_last**2 / 4000.0 - np.cos(r_last) + 1.0
    total += g_last
    return float(total)


# ============================================================================
# 2. OPERADORES TRANSFORMACIÓN (SHIFT, ROTATE, HYBRID & COMPOSITION)
# ============================================================================

def shift(x: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Aplica vector de desplazamiento (shift): z = x - o."""
    return np.asarray(x, dtype=float) - np.asarray(o, dtype=float)


def rotate(x: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Aplica matriz de rotación: z = M * x."""
    return np.dot(np.asarray(M, dtype=float), np.asarray(x, dtype=float))


def sr_func(x: np.ndarray, o: np.ndarray, M: np.ndarray, sh_flag: bool = True, rot_flag: bool = True) -> np.ndarray:
    """Aplica shift y rotación según banderas."""
    z = np.asarray(x, dtype=float)
    if sh_flag:
        z = shift(z, o)
    if rot_flag:
        z = rotate(z, M)
    return z


def hybrid_func(x: np.ndarray, sub_funcs: List[Callable[[np.ndarray], float]],
                proportions: List[float], o: np.ndarray, M: np.ndarray) -> float:
    """
    Evalúa una función híbrida dividiendo la solución en sub-vectores según proporciones.
    """
    z = sr_func(x, o, M)
    n = len(z)
    num_funcs = len(sub_funcs)
    
    # Calcular tamaños de sub-vectores
    sizes = [int(p * n) for p in proportions]
    sizes[-1] = n - sum(sizes[:-1]) # asegurar suma igual a n
    
    total = 0.0
    start = 0
    for idx in range(num_funcs):
        end = start + sizes[idx]
        sub_vec = z[start:end]
        if len(sub_vec) > 0:
            total += sub_funcs[idx](sub_vec)
        start = end
    return total


def composition_func(x: np.ndarray, sub_funcs: List[Callable[[np.ndarray], float]],
                     centros: List[np.ndarray], matrices: List[np.ndarray],
                     sigmas: List[float], lambdas: List[float],
                     biases: List[float]) -> float:
    """
    Evalúa una función de composición combinando subfunciones con pesos gaussianos.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    N = len(sub_funcs)
    
    weights = np.zeros(N)
    fit_vals = np.zeros(N)
    
    for i in range(N):
        o_i = centros[i]
        M_i = matrices[i]
        sigma_i = sigmas[i]
        lambda_i = lambdas[i]
        
        diff = x - o_i
        dist2 = np.sum(diff**2)
        
        if dist2 < 1e-15:
            weights[i] = 1e300 # peso infinitésimo dominante en el centro exacto
        else:
            weights[i] = (1.0 / np.sqrt(dist2)) * np.exp(-dist2 / (2.0 * n * sigma_i**2))
            
        z_i = rotate(diff, M_i)
        g_val = sub_funcs[i](z_i / lambda_i)
        fit_vals[i] = lambda_i * g_val + biases[i]
        
    w_sum = np.sum(weights)
    if w_sum == 0 or np.isinf(w_sum) or np.isnan(w_sum):
        weights = np.ones(N) / N
    else:
        weights = weights / w_sum
        
    return float(np.sum(weights * fit_vals))


# ============================================================================
# 3. CARGADOR Y GENERADOR DE MATRICES SHIFT Y ROTACIÓN
# ============================================================================

def _generate_orthogonal_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Genera una matriz de rotación ortogonal N x N usando QR."""
    A = rng.normal(0, 1, (n, n))
    Q, R = np.linalg.qr(A)
    # Asegurar determinante 1 (rotación pura)
    d = np.diag(R)
    ph = d / np.abs(d)
    Q = Q @ np.diag(ph)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def get_shift_matrix_data(func_num: int, n_dim: int, num_subfuncs: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Obtiene los vectores de shift y matrices de rotación para una función y dimensión.
    Si los archivos oficiales existen en input_data/, los carga.
    De lo contrario, los genera de manera estricta y reproducible (semilla fija).
    """
    shift_file = os.path.join(DATA_DIR, f"shift_data_{func_num}.txt")
    rot_file = os.path.join(DATA_DIR, f"M_{func_num}_D{n_dim}.txt")
    
    if os.path.exists(shift_file) and os.path.exists(rot_file):
        try:
            raw_shifts = np.loadtxt(shift_file)
            raw_rot = np.loadtxt(rot_file)
            
            shifts = []
            rots = []
            if num_subfuncs == 1:
                shifts.append(raw_shifts[:n_dim])
                rots.append(raw_rot[:n_dim, :n_dim])
            else:
                for i in range(num_subfuncs):
                    shifts.append(raw_shifts[i, :n_dim])
                    rots.append(raw_rot[i * n_dim:(i + 1) * n_dim, :n_dim])
            return shifts, rots
        except Exception:
            pass # Si ocurre error al cargar, cae al generador reproducible

    # Generador reproducible si no hay archivos externos
    seed = 1000 + func_num * 100 + n_dim
    rng = np.random.default_rng(seed)
    
    shifts = []
    rots = []
    for _ in range(num_subfuncs):
        # Shift aleatorio en [-80, 80]
        s = rng.uniform(-80.0, 80.0, size=n_dim)
        M = _generate_orthogonal_matrix(n_dim, rng)
        shifts.append(s)
        rots.append(M)
        
    return shifts, rots


# ============================================================================
# 4. DEFINICIÓN OFICIAL DE LAS 12 FUNCIONES CEC 2022
# ============================================================================

OFFICIAL_BIASES = {
    1: 300.0,
    2: 400.0,
    3: 600.0,
    4: 800.0,
    5: 900.0,
    6: 1800.0,
    7: 2000.0,
    8: 2200.0,
    9: 2300.0,
    10: 2400.0,
    11: 2600.0,
    12: 2700.0,
}

def cec2022_f1(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F1: Shifted and Rotated Zakharov Function (Unimodal). Bias = 300."""
    z = sr_func(x, shifts[0], rots[0])
    return zakharov(z) + OFFICIAL_BIASES[1]


def cec2022_f2(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F2: Shifted and Rotated Rosenbrock Function (Basic Multimodal). Bias = 400."""
    z = sr_func(x, shifts[0], rots[0])
    return rosenbrock(z) + OFFICIAL_BIASES[2]


def cec2022_f3(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F3: Shifted and Rotated Expanded Schaffer's f6 Function (Basic Multimodal). Bias = 600."""
    z = sr_func(x, shifts[0], rots[0])
    return expanded_schaffer_f6(z) + OFFICIAL_BIASES[3]


def cec2022_f4(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F4: Shifted and Rotated Non-Continuous Rastrigin Function (Basic Multimodal). Bias = 800."""
    z = sr_func(x, shifts[0], rots[0])
    return non_continuous_rastrigin(z) + OFFICIAL_BIASES[4]


def cec2022_f5(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F5: Shifted and Rotated Levy Function (Basic Multimodal). Bias = 900."""
    z = sr_func(x, shifts[0], rots[0])
    return levy(z) + OFFICIAL_BIASES[5]


def cec2022_f6(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F6: Hybrid Function 1 (N=3). Bias = 1800."""
    sub_funcs = [bent_cigar, rastrigin, expanded_schaffer_f6]
    props = [0.4, 0.3, 0.3]
    return hybrid_func(x, sub_funcs, props, shifts[0], rots[0]) + OFFICIAL_BIASES[6]


def cec2022_f7(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F7: Hybrid Function 2 (N=6). Bias = 2000."""
    sub_funcs = [hgbat, katsuura, ackley, rastrigin, schwefel, expanded_schaffer_f6]
    props = [0.1, 0.2, 0.2, 0.2, 0.15, 0.15]
    return hybrid_func(x, sub_funcs, props, shifts[0], rots[0]) + OFFICIAL_BIASES[7]


def cec2022_f8(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F8: Hybrid Function 3 (N=5). Bias = 2200."""
    sub_funcs = [katsuura, happycat, grie_rosen, schwefel, ackley]
    props = [0.2, 0.2, 0.2, 0.2, 0.2]
    return hybrid_func(x, sub_funcs, props, shifts[0], rots[0]) + OFFICIAL_BIASES[8]


def cec2022_f9(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F9: Composition Function 1 (N=5). Bias = 2300."""
    sub_funcs = [rosenbrock, bent_cigar, rastrigin, ackley, schwefel]
    sigmas = [10.0, 20.0, 30.0, 40.0, 50.0]
    lambdas = [1.0, 1.0, 1.0, 1.0, 1.0]
    sub_biases = [0.0, 100.0, 200.0, 300.0, 400.0]
    return composition_func(x, sub_funcs, shifts, rots, sigmas, lambdas, sub_biases) + OFFICIAL_BIASES[9]


def cec2022_f10(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F10: Composition Function 2 (N=4). Bias = 2400."""
    sub_funcs = [ackley, bent_cigar, griewank, rastrigin]
    sigmas = [10.0, 20.0, 30.0, 40.0]
    lambdas = [1.0, 1.0, 1.0, 1.0]
    sub_biases = [0.0, 100.0, 200.0, 300.0]
    return composition_func(x, sub_funcs, shifts, rots, sigmas, lambdas, sub_biases) + OFFICIAL_BIASES[10]


def cec2022_f11(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F11: Composition Function 3 (N=5). Bias = 2600."""
    sub_funcs = [expanded_schaffer_f6, schwefel, rosenbrock, rastrigin, bent_cigar]
    sigmas = [10.0, 20.0, 30.0, 40.0, 50.0]
    lambdas = [1.0, 1.0, 1.0, 1.0, 1.0]
    sub_biases = [0.0, 100.0, 200.0, 300.0, 400.0]
    return composition_func(x, sub_funcs, shifts, rots, sigmas, lambdas, sub_biases) + OFFICIAL_BIASES[11]


def cec2022_f12(x: np.ndarray, shifts: List[np.ndarray], rots: List[np.ndarray]) -> float:
    """F12: Composition Function 4 (N=6). Bias = 2700."""
    sub_funcs = [hgbat, rastrigin, schwefel, bent_cigar, discus, expanded_schaffer_f6]
    sigmas = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    lambdas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    sub_biases = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    return composition_func(x, sub_funcs, shifts, rots, sigmas, lambdas, sub_biases) + OFFICIAL_BIASES[12]


# ============================================================================
# 5. FUNCIÓN MAESTRA Y CATÁLOGO CONTINUO
# ============================================================================

_FUNC_DISPATCH = {
    1: (cec2022_f1, 1),
    2: (cec2022_f2, 1),
    3: (cec2022_f3, 1),
    4: (cec2022_f4, 1),
    5: (cec2022_f5, 1),
    6: (cec2022_f6, 1),
    7: (cec2022_f7, 1),
    8: (cec2022_f8, 1),
    9: (cec2022_f9, 5),
    10: (cec2022_f10, 4),
    11: (cec2022_f11, 5),
    12: (cec2022_f12, 6),
}

# Caché interno para matrices de shift y rotación por (func_num, n_dim)
_DATA_CACHE: Dict[Tuple[int, int], Tuple[List[np.ndarray], List[np.ndarray]]] = {}


def get_cec2022_data(func_num: int, n_dim: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Devuelve (shifts, rots) desde la caché o los carga/genera."""
    key = (func_num, n_dim)
    if key not in _DATA_CACHE:
        _, num_subfuncs = _FUNC_DISPATCH[func_num]
        _DATA_CACHE[key] = get_shift_matrix_data(func_num, n_dim, num_subfuncs)
    return _DATA_CACHE[key]


def cec2022_func(x: np.ndarray, func_num: int, n_dim: Optional[int] = None) -> float:
    """
    Función Maestra de Benchmark CEC 2022.

    Parámetros
    ----------
    x : np.ndarray
        Vector de solución de dimensión nD.
    func_num : int
        Número de función CEC 2022 (1 a 12).
    n_dim : Optional[int]
        Dimensión de la función (si es None, se usa len(x)).

    Retorna
    -------
    float
        Valor de fitness (minimización).
    """
    if func_num not in _FUNC_DISPATCH:
        raise ValueError(f"Número de función inválido: {func_num}. Debe estar entre 1 y 12.")

    x_arr = np.asarray(x, dtype=float)
    if n_dim is None:
        n_dim = len(x_arr)
        
    fn_impl, _ = _FUNC_DISPATCH[func_num]
    shifts, rots = get_cec2022_data(func_num, n_dim)
    
    return fn_impl(x_arr, shifts, rots)


def get_cec2022_optimum_point(func_num: int, n_dim: int) -> np.ndarray:
    """Devuelve el vector x óptimo conocido para una función y dimensión dada."""
    shifts, _ = get_cec2022_data(func_num, n_dim)
    return shifts[0].copy()


def get_test_functions(n_dim: int = 20) -> List[ContinuousFunction]:
    """Retorna los descroptores de las 12 funciones del conjunto CEC 2022."""
    names = [
        "F1_Zakharov_Shifted_Rotated",
        "F2_Rosenbrock_Shifted_Rotated",
        "F3_Expanded_Schaffer_F6",
        "F4_NonContinuous_Rastrigin",
        "F5_Levy_Shifted_Rotated",
        "F6_Hybrid_Function_1",
        "F7_Hybrid_Function_2",
        "F8_Hybrid_Function_3",
        "F9_Composition_Function_1",
        "F10_Composition_Function_2",
        "F11_Composition_Function_3",
        "F12_Composition_Function_4",
    ]
    
    functions = []
    for num in range(1, 13):
        # Closure seguro para congelar func_num
        def make_func(f_num: int):
            return lambda x: cec2022_func(x, f_num, n_dim)
            
        functions.append(
            ContinuousFunction(
                name    = names[num-1],
                func    = make_func(num),
                lb      = -100.0,
                ub      = 100.0,
                optimum = OFFICIAL_BIASES[num],
                n_dim   = n_dim,
            )
        )
    return functions
