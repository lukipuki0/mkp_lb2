"""
Vanilla strategy config — no DTW adaptation.

All shared values come from mkp_common.config.
Change things there to affect ALL strategies at once.
"""

from mkp_common import BinaryDE, BinaryGWO, GeneticAlgorithm, BinaryPSO
from mkp_common.config import (
    RUTA_INSTANCIA,
    INDICE_INSTANCIA,
    NUM_PARTICULAS,
    NUM_ITERACIONES,
    EPOCHS,
    SEMILLA,
    VERBOSE,
)

# --- Default MH for single-run scripts ---
MH_CLASS = BinaryPSO
