"""
================================================================================
Funciones objetivo continuas tipo CEC 2022 - Implementación didáctica
================================================================================
Estas son las funciones BASE matemáticas en las que se inspiran las 12
funciones del benchmark CEC 2022 mencionado en el paper MSE-ABC.

NOTA IMPORTANTE:
El benchmark CEC 2022 "oficial" aplica además una ROTACIÓN (matriz M) y un
DESPLAZAMIENTO (vector o) a cada función, usando matrices/vectores fijos
publicados por los organizadores de la competencia, para que el óptimo no
esté trivialmente en el origen. Aquí implementamos las funciones base SIN
esa rotación/desplazamiento (forma "shifted=False, rotated=False"), que es
la forma estándar en que se enseñan y se usan para experimentación rápida.
Si necesitas exactamente los resultados numéricos del paper (Tabla 2), hay
que usar los archivos de rotación/desplazamiento oficiales de la competencia
CEC2022 (disponibles en el repositorio oficial del IEEE CEC).

Todas las funciones reciben x como un array de numpy de tamaño n.
Todas son funciones de MINIMIZACIÓN (igual que en el paper: f(·) a minimizar).
================================================================================
"""

import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class ContinuousFunction:
    """Descriptor de una funcion de benchmark continua."""
    name    : str
    func    : Callable[[np.ndarray], float]
    lb      : float          # limite inferior de cada dimension
    ub      : float          # limite superior de cada dimension
    optimum : float          # valor optimo conocido (minimo global)
    n_dim   : int            # dimensionalidad


# ============================================================================
# Funciones clasicas
# ============================================================================

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def griewank(x: np.ndarray) -> float:
    sum_term  = np.sum(x ** 2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(sum_term - prod_term + 1.0)


def ackley(x: np.ndarray) -> float:
    n = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    t1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / n))
    t2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return float(t1 + t2 + a + np.exp(1))


# ============================================================================
# CEC1 - Zakharov (unimodal)
# ============================================================================
def zakharov(x: np.ndarray) -> float:
    """
    f(x) = sum(x_i^2) + [sum(0.5*i*x_i)]^2 + [sum(0.5*i*x_i)]^4

    Unimodal: un solo mínimo global en x = [0,...,0], f(0) = 0.
    Prueba pura capacidad de explotación/convergencia.
    """
    n = len(x)
    i = np.arange(1, n + 1)
    s1 = np.sum(x ** 2)
    s2 = np.sum(0.5 * i * x)
    return s1 + s2 ** 2 + s2 ** 4


# ============================================================================
# CEC2 - Rosenbrock (unimodal, valle curvo estrecho)
# ============================================================================
def rosenbrock(x: np.ndarray) -> float:
    """
    f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]

    Mínimo global en x = [1,1,...,1], f = 0.
    El "valle del plátano": muy fácil entrar al valle, muy difícil
    seguir su curvatura hasta el mínimo exacto.
    """
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


# ============================================================================
# CEC3 - Schaffer's F6 expandida (multimodal, ondas concéntricas)
# ============================================================================
def schaffer_f6(x_i: float, x_j: float) -> float:
    """F6 de Schaffer para un par de variables."""
    num = np.sin(np.sqrt(x_i ** 2 + x_j ** 2)) ** 2 - 0.5
    den = (1 + 0.001 * (x_i ** 2 + x_j ** 2)) ** 2
    return 0.5 + num / den


def expanded_schaffer_f6(x: np.ndarray) -> float:
    """
    f(x) = sum_{i=1}^{n-1} F6(x_i, x_{i+1}) + F6(x_n, x_1)

    Mínimo global en x = [0,...,0], f = 0.
    Genera muchísimos óptimos locales en anillos concéntricos.
    """
    n = len(x)
    total = 0.0
    for i in range(n - 1):
        total += schaffer_f6(x[i], x[i + 1])
    total += schaffer_f6(x[-1], x[0])  # cierra el ciclo
    return total


# ============================================================================
# CEC4 - Rastrigin (altamente multimodal, "papel de huevos")
# ============================================================================
def rastrigin(x: np.ndarray) -> float:
    """
    f(x) = sum [x_i^2 - 10*cos(2*pi*x_i) + 10]

    Mínimo global en x = [0,...,0], f = 0.
    El coseno crea infinitos mínimos locales regularmente espaciados.
    """
    n = len(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


# ============================================================================
# CEC5 - Levy (superficie irregular con transiciones abruptas)
# ============================================================================
def levy(x: np.ndarray) -> float:
    """
    w_i = 1 + (x_i - 1) / 4

    f(x) = sin^2(pi*w_1)
         + sum_{i=1}^{n-1} (w_i - 1)^2 * [1 + 10*sin^2(pi*w_i + 1)]
         + (w_n - 1)^2 * [1 + sin^2(2*pi*w_n)]

    Mínimo global en x = [1,...,1], f = 0.
    """
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    return term1 + term2 + term3


# ============================================================================
# CEC6-8 - Funciones HÍBRIDAS (combinación ponderada por subconjuntos)
# ============================================================================
def hybrid_function(x: np.ndarray, funciones_base, pesos_proporcion) -> float:
    """
    Estructura general de una función híbrida CEC:
        f(x) = sum_k w_k * g_k(x_{S_k})

    donde cada g_k es una función base aplicada SOLO a un subconjunto S_k
    de las dimensiones (particionadas), y w_k son pesos de ponderación.

    Parámetros
    ----------
    funciones_base : lista de funciones, ej. [rastrigin, rosenbrock, levy]
    pesos_proporcion : lista de fracciones que suman 1, ej. [0.4, 0.3, 0.3]
                       define qué fracción de las dimensiones va a cada función
    """
    n = len(x)
    k = len(funciones_base)
    assert abs(sum(pesos_proporcion) - 1.0) < 1e-6, "los pesos deben sumar 1"

    # particionar las dimensiones según las proporciones
    cortes = np.cumsum([int(p * n) for p in pesos_proporcion])
    cortes[-1] = n  # asegurar que cubra todas las dimensiones
    inicio = 0
    total = 0.0
    for idx in range(k):
        fin = cortes[idx]
        subvector = x[inicio:fin] if fin > inicio else x[inicio:inicio + 1]
        total += funciones_base[idx](subvector)
        inicio = fin
    return total


def cec_hibrida_ejemplo(x: np.ndarray) -> float:
    """
    Ejemplo concreto de función híbrida (estilo CEC6-8):
    40% de las dimensiones -> Rastrigin
    30% de las dimensiones -> Rosenbrock
    30% de las dimensiones -> Levy
    """
    return hybrid_function(x, [rastrigin, rosenbrock, levy], [0.4, 0.3, 0.3])


# ============================================================================
# CEC9-12 - Funciones de COMPOSICIÓN (mezcla gaussiana de varias funciones)
# ============================================================================
def composition_function(x: np.ndarray, funciones_base, centros, sigmas,
                          lambdas, biases) -> float:
    """
    Estructura general de una función de composición CEC:

        f(x) = sum_i [ w_i * (lambda_i * g_i((x - o_i)/lambda_i) + bias_i) ]

        w_i = exp(-sum((x-o_i)^2) / (2*sigma_i^2)) / sum_k exp(-sum((x-o_k)^2)/(2*sigma_k^2))

    Cada función base g_i tiene su propio "centro" o_i (dónde está su óptimo
    local), su propia escala sigma_i (qué tan amplia es su zona de influencia)
    y su propio bias_i (un offset vertical para diferenciarlas).

    Parámetros
    ----------
    funciones_base : lista de funciones base, ej. [rastrigin, levy, zakharov]
    centros        : lista de vectores o_i (mismo tamaño que x) por función
    sigmas         : lista de anchos de influencia por función
    lambdas        : lista de factores de escala por función
    biases         : lista de offsets verticales por función
    """
    N = len(funciones_base)
    pesos = np.zeros(N)
    valores = np.zeros(N)

    for i in range(N):
        diff = x - centros[i]
        dist2 = np.sum(diff ** 2)
        pesos[i] = np.exp(-dist2 / (2 * sigmas[i] ** 2 + 1e-12))
        g_val = funciones_base[i](diff / lambdas[i])
        valores[i] = lambdas[i] * g_val + biases[i]

    pesos_sum = np.sum(pesos) + 1e-12
    pesos_norm = pesos / pesos_sum

    return float(np.sum(pesos_norm * valores))


def cec_composicion_ejemplo(x: np.ndarray) -> float:
    """
    Ejemplo concreto de función de composición (estilo CEC9-12) combinando
    Rastrigin, Levy y Zakharov, cada una con su propio centro desplazado.
    """
    n = len(x)
    rng = np.random.default_rng(0)  # fijo para reproducibilidad del ejemplo

    funciones_base = [rastrigin, levy, zakharov]
    centros = [rng.uniform(-2, 2, n) for _ in funciones_base]
    sigmas = [10, 20, 30]
    lambdas = [1.0, 1.0, 1.0]
    biases = [0, 100, 200]

    return composition_function(x, funciones_base, centros, sigmas, lambdas, biases)


# ============================================================================
# DICCIONARIO DE TODAS LAS FUNCIONES (para uso fácil en un algoritmo de optimización)
# ============================================================================
FUNCIONES_CEC2022_DIDACTICAS = {
    "CEC1_zakharov": zakharov,
    "CEC2_rosenbrock": rosenbrock,
    "CEC3_schaffer_f6": expanded_schaffer_f6,
    "CEC4_rastrigin": rastrigin,
    "CEC5_levy": levy,
    "CEC6_8_hibrida_ejemplo": cec_hibrida_ejemplo,
    "CEC9_12_composicion_ejemplo": cec_composicion_ejemplo,
}


# ============================================================================
# CATÁLOGO DE FUNCIONES DE BENCHMARK
# ============================================================================

def get_test_functions(n_dim: int = 30) -> list[ContinuousFunction]:
    """Retorna la lista de todas las funciones continuas a ejecutar en la corrida."""
    return [
        ContinuousFunction("Sphere",     sphere,     -100.0,   100.0,   0.0, n_dim),
        ContinuousFunction("Rastrigin",  rastrigin,    -5.12,    5.12,  0.0, n_dim),
        ContinuousFunction("Rosenbrock", rosenbrock,  -30.0,    30.0,   0.0, n_dim),
        ContinuousFunction("Griewank",   griewank,   -600.0,   600.0,   0.0, n_dim),
        ContinuousFunction("Ackley",     ackley,      -32.768,  32.768, 0.0, n_dim),
        # Funciones CEC2022
        ContinuousFunction("CEC1_Zakharov", zakharov, -100.0, 100.0, 0.0, n_dim),
        ContinuousFunction("CEC3_Schaffer", expanded_schaffer_f6, -100.0, 100.0, 0.0, n_dim),
        ContinuousFunction("CEC5_Levy", levy, -10.0, 10.0, 0.0, n_dim),
        ContinuousFunction("CEC6-8_Hybrid", cec_hibrida_ejemplo, -100.0, 100.0, 0.0, n_dim),
        ContinuousFunction("CEC9-12_Comp", cec_composicion_ejemplo, -100.0, 100.0, 300.0, n_dim),
    ]
