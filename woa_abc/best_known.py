"""
Best known values of the OR-Library mknapcb instances (spec S9).

Copied from gwo_de_dtw/problem.py on feat/gwo-de-dtw-hybrid (cross-branch
import is impossible): the instance headers of instancias/mknapcb*.txt store
0, so the reference is the "Best Feasible Solution Value" of that instance in
mkcbres.txt in the same folder as the instance file (OR-Library, Chu &
Beasley 1997). Only the first table is read; a missing file or entry is 0
(unknown, no gap).
"""

from pathlib import Path
from typing import Tuple

# OR-Library name prefix (m.n) of each mknapcb file, as used in mkcbres.txt.
FAMILY_PREFIX = {
    "mknapcb1": "5.100", "mknapcb2": "5.250", "mknapcb3": "5.500",
    "mknapcb4": "10.100", "mknapcb5": "10.250", "mknapcb6": "10.500",
    "mknapcb7": "30.100", "mknapcb8": "30.250", "mknapcb9": "30.500",
}
BEST_KNOWN_FILE = "mkcbres.txt"

# Where a resolved reference came from.
SOURCE_HEADER = "header"
SOURCE_BEST_KNOWN = BEST_KNOWN_FILE
SOURCE_UNKNOWN = "unknown"


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


def resolve_reference(ruta_instancia, idx: int, stored: float) -> Tuple[float, str]:
    """(reference, source): the header value when positive, else mkcbres.txt, else unknown.

    A non-positive header without a best-known entry is returned unchanged, so
    the solver still rejects a negative one.
    """
    if stored > 0:
        return float(stored), SOURCE_HEADER
    best = mejor_conocido(str(ruta_instancia), idx)
    if best > 0:
        return float(best), SOURCE_BEST_KNOWN
    return float(stored), SOURCE_UNKNOWN
