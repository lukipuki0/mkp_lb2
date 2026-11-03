"""
Carga de instancias MKP (OR-Library) y reparación greedy.
"""

from pathlib import Path
from typing import Tuple

import numpy as np


def cargar_instancia(ruta: str, idx: int = 0) -> dict:
    """
    Carga una instancia MKP desde un archivo OR-Library.

    Args:
        ruta: Path al .txt (ej: "mh_mkp/instances/mknapcb1.txt")
        idx:  Indice de la instancia (0-based)

    Returns:
        dict con: n, m, p (profits), r (weights m*n), b (capacities),
                  optimo, densidad
    """
    texto = Path(ruta).read_text()
    tokens = list(map(float, texto.split()))
    it = iter(tokens)

    num_instancias = int(next(it))
    if not (0 <= idx < num_instancias):
        raise ValueError(f"idx={idx} fuera de rango [0, {num_instancias})")

    for _ in range(idx):
        n_skip = int(next(it))
        m_skip = int(next(it))
        next(it)
        for _ in range(n_skip):
            next(it)
        for _ in range(m_skip * n_skip):
            next(it)
        for _ in range(m_skip):
            next(it)

    n = int(next(it))
    m = int(next(it))
    optimo = float(next(it))

    p = np.array([next(it) for _ in range(n)])
    r = np.array([next(it) for _ in range(m * n)]).reshape(m, n)
    b = np.array([next(it) for _ in range(m)])

    pesos_norm = r / b[:, np.newaxis]
    peso_prom = np.mean(pesos_norm, axis=0)
    densidad = p / (peso_prom + 1e-12)

    # Si el archivo no tiene óptimo, buscamos en la tabla de Chu & Beasley
    if optimo == 0:
        optimo = _buscar_optimo_conocido(ruta, idx)

    return {
        "n": n, "m": m, "p": p, "r": r, "b": b,
        "optimo": optimo, "densidad": densidad,
    }


# Óptimos conocidos — Chu & Beasley (1998), OR-Library mkcbres
# Formato: (m, n) → {idx: valor}  donde m=constraints, n=items
# Archivos: mknapcb1=5.100, mknapcb2=5.250, mknapcb3=5.500,
#           mknapcb4=10.100, mknapcb5=10.250, mknapcb6=10.500,
#           mknapcb7=30.100, mknapcb8=30.250, mknapcb9=30.500
_OPTIMOS_CB = {
    "mknapcb1": [
        24381, 24274, 23551, 23534, 23991, 24613, 25591, 23410, 24216, 24411,
        42757, 42545, 41968, 45090, 42218, 42927, 42009, 45020, 43441, 44554,
        59822, 62081, 59802, 60479, 61091, 58959, 61538, 61520, 59453, 59965,
    ],
    "mknapcb2": [
        59312, 61472, 62130, 59446, 58951, 60056, 60414, 61472, 61885, 58959,
        109109, 109841, 108489, 109383, 110720, 110256, 109016, 109037, 109957, 107038,
        149659, 155940, 149316, 152130, 150353, 150045, 148607, 149772, 155075, 154662,
    ],
    "mknapcb3": [
        120130, 117837, 121109, 120798, 122319, 122007, 119113, 120568, 121575, 120699,
        218422, 221191, 217534, 223558, 218962, 220514, 219987, 218194, 216976, 219693,
        295828, 308077, 299796, 306476, 300342, 302560, 301322, 306430, 302814, 299904,
    ],
    "mknapcb4": [
        23064, 22801, 22131, 22772, 22751, 22777, 21875, 22635, 22511, 22702,
        41395, 42344, 42401, 45624, 41884, 42995, 43559, 42970, 42212, 41207,
        57375, 58978, 58391, 61966, 60803, 61437, 56377, 59391, 60205, 60633,
    ],
    "mknapcb5": [
        59187, 58662, 58094, 61000, 58092, 58803, 58607, 58917, 59384, 59193,
        110863, 108659, 108932, 110037, 108423, 110841, 106075, 106686, 109825, 106723,
        151790, 148772, 151900, 151275, 151948, 152109, 153131, 153520, 149155, 149704,
    ],
    "mknapcb6": [
        117726, 119139, 119159, 118802, 116434, 119454, 119749, 118288, 117779, 119125,
        217318, 219022, 217772, 216802, 213809, 215013, 217896, 219949, 214332, 220833,
        304344, 302332, 302354, 300743, 304344, 301730, 304949, 296437, 301313, 307014,
    ],
    "mknapcb7": [
        21946, 21716, 20754, 21464, 21814, 22176, 21799, 21397, 22493, 20983,
        40767, 41304, 41560, 41041, 40872, 41058, 41062, 42719, 42230, 41700,
        57494, 60027, 58025, 60776, 58884, 60011, 58132, 59064, 58975, 60603,
    ],
    "mknapcb8": [
        56693, 58318, 56553, 56863, 56629, 57119, 56292, 56403, 57442, 56447,
        107689, 108338, 106385, 106796, 107396, 107246, 106308, 103993, 106835, 105751,
        150083, 149907, 152993, 153169, 150287, 148544, 147471, 152841, 149568, 149572,
    ],
    "mknapcb9": [
        115868, 114667, 116661, 115237, 116353, 115604, 113952, 114199, 115247, 116947,
        217995, 214534, 215854, 217836, 215566, 215762, 215772, 216336, 217290, 214624,
        301627, 299985, 304995, 301935, 304404, 296894, 303233, 306944, 303057, 300460,
    ],
}


def _buscar_optimo_conocido(ruta: str, idx: int) -> float:
    """Busca el óptimo conocido en la tabla de Chu & Beasley."""
    nombre = Path(ruta).stem.lower()  # ej: "mknapcb1"
    if nombre in _OPTIMOS_CB:
        tabla = _OPTIMOS_CB[nombre]
        if 0 <= idx < len(tabla):
            return float(tabla[idx])
    return 0.0


def reparar(solucion: np.ndarray, inst: dict) -> Tuple[np.ndarray, float]:
    """
    Repara una solucion binaria infactible:
    1. Remueve items de menor densidad hasta factibilidad
    2. Agrega items de mayor densidad que quepan (greedy add)
    """
    x = solucion.copy()
    p, r, b = inst["p"], inst["r"], inst["b"]
    orden = np.argsort(inst["densidad"])

    for idx in orden:
        if np.all(r @ x <= b):
            break
        if x[idx] == 1:
            x[idx] = 0

    for idx in reversed(orden):
        if x[idx] == 1:
            continue
        x[idx] = 1
        if not np.all(r @ x <= b):
            x[idx] = 0

    fitness = float(np.dot(p, x))
    return x, fitness
