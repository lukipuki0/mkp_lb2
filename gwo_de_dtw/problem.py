"""
MKP instance loading and greedy repair, ported from hybrid_mkp so this package
does not depend on it:

- parsing:  hybrid_mkp/mkp_core/data_loader.py::parsear_instancias (local file only)
- optimum:  the third number of the instance header; when it is 0 (as in
            instancias/mknapcb*.txt), the best known value of that instance from
            mkcbres.txt in the same folder as the instance file (OR-Library,
            Chu & Beasley 1997, "Best Feasible Solution Value" table); if that
            file or entry is missing, 0 (no gap)
- density:  hybrid_mkp/mkp_core/problem.py::MKPInstance._calcular_densidades
- repair:   hybrid_mkp/mkp_core/repair.py::reparar_solucion

The instance is a dict: n, m, p (profits), r (weights m x n), b (capacities),
optimo, densidad.
"""

from pathlib import Path
from typing import Tuple

import numpy as np

# OR-Library name prefix (m.n) of each mknapcb file, as used in mkcbres.txt.
FAMILY_PREFIX = {
    "mknapcb1": "5.100", "mknapcb2": "5.250", "mknapcb3": "5.500",
    "mknapcb4": "10.100", "mknapcb5": "10.250", "mknapcb6": "10.500",
    "mknapcb7": "30.100", "mknapcb8": "30.250", "mknapcb9": "30.500",
}
BEST_KNOWN_FILE = "mkcbres.txt"


def leer_mejores_conocidos(ruta: Path) -> dict:
    """Read the first table of mkcbres.txt: {'5.100-00': 24381.0, ...}."""
    mejores = {}
    for linea in ruta.read_text().splitlines():
        if "Problem Name" in linea and mejores:
            break                      # second table (LP optimal) starts
        partes = linea.split()
        if len(partes) == 2 and "-" in partes[0] and partes[0][0].isdigit():
            mejores[partes[0]] = float(partes[1])
    return mejores


def mejor_conocido(ruta_instancia: str, idx: int) -> float:
    """Best known value of instance `idx` from mkcbres.txt next to the file, or 0."""
    archivo = Path(ruta_instancia).parent / BEST_KNOWN_FILE
    prefijo = FAMILY_PREFIX.get(Path(ruta_instancia).stem.lower())
    if prefijo is None or not archivo.is_file():
        return 0.0
    return leer_mejores_conocidos(archivo).get(f"{prefijo}-{idx:02d}", 0.0)


def parsear_instancias(cadena: str) -> list:
    """Parse OR-Library text into dicts with n, m, valor_optimo, p, r, b."""
    lineas = cadena.strip().split("\n")
    numeros = [list(map(float, linea.split())) for linea in lineas if linea.strip()]

    ptr = 0
    n_instancias = int(numeros[ptr][0])
    ptr += 1

    instancias = []
    for _ in range(n_instancias):
        n, m, valor_optimo = numeros[ptr]
        n, m = int(n), int(m)
        ptr += 1

        p = []
        while len(p) < n:
            p.extend(numeros[ptr])
            ptr += 1

        r_filas = []
        for _ in range(m):
            fila = []
            while len(fila) < n:
                fila.extend(numeros[ptr])
                ptr += 1
            r_filas.append(fila)

        b = []
        while len(b) < m:
            b.extend(numeros[ptr])
            ptr += 1

        instancias.append({
            "n": n, "m": m, "valor_optimo": valor_optimo,
            "p": np.array(p), "r": np.array(r_filas), "b": np.array(b),
        })
    return instancias


def densidad_minima(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """density_j = min_i(p_j / r_ij); infinite ratios become 10 x the maximum finite one."""
    with np.errstate(divide="ignore"):
        density_per_bag = np.array([p / r[i] for i in range(r.shape[0])])
    finite_max = np.max(density_per_bag[np.isfinite(density_per_bag)])
    density_per_bag[np.isinf(density_per_bag)] = finite_max * 10
    return np.min(density_per_bag, axis=0)


def cargar_instancia(ruta: str, idx: int = 0) -> dict:
    """Load instance `idx` (0-based) of an OR-Library mknapcb file."""
    instancias = parsear_instancias(Path(ruta).read_text())
    if not (0 <= idx < len(instancias)):
        raise ValueError(f"idx={idx} fuera de rango [0, {len(instancias)})")
    data = instancias[idx]

    optimo = float(data["valor_optimo"])
    if optimo <= 0:
        optimo = mejor_conocido(ruta, idx)

    return {
        "n": data["n"], "m": data["m"], "p": data["p"], "r": data["r"], "b": data["b"],
        "optimo": optimo, "densidad": densidad_minima(data["p"], data["r"]),
    }


def reparar(solucion: np.ndarray, inst: dict) -> Tuple[np.ndarray, float]:
    """
    Repair a binary solution (the input is not modified):
    1. Expulsion: from lowest to highest density, drop items until feasible.
    2. Insertion: from highest to lowest density, add every item that fits.
    """
    x = np.array(solucion, dtype=int)
    p, r, b = inst["p"], inst["r"], inst["b"]
    indices_asc = np.argsort(inst["densidad"])
    uso = r @ x

    for idx in indices_asc:
        if np.all(uso <= b):
            break
        if x[idx] == 1:
            x[idx] = 0
            uso = uso - r[:, idx]

    for idx in indices_asc[::-1]:
        if x[idx] == 0:
            w = r[:, idx]
            if np.all(uso + w <= b):
                x[idx] = 1
                uso = uso + w

    fitness = float(x @ p)
    return x, fitness
