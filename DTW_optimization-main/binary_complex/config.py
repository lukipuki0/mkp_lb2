"""
Fire Binario (A4) config — 3-condition DTW fire.

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
    DTW_FIRE_BINARIO,
)

# --- Default MH for single-run scripts ---
MH_CLASS = BinaryPSO

# --- DTW config for this strategy ---
DTW_CFG = DTW_FIRE_BINARIO
