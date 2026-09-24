"""
B3 D2-Direct config — continuous intensity: 1 - clip(D2 / (theta_c * scale), 0, 1).

All shared values come from mkp_common.config.
Change things there to affect ALL strategies at once.
"""

from mkp_common.config import (
    RUTA_INSTANCIA,
    INDICE_INSTANCIA,
    NUM_PARTICULAS,
    NUM_ITERACIONES,
    EPOCHS,
    SEMILLA,
    VERBOSE,
    DTW_B3_D2,
    B3_SCALE,
)

from mkp_common import BinaryPSO

# --- Default MH for single-run scripts ---
MH_CLASS = BinaryPSO

# --- DTW config for this strategy ---
DTW_CFG = DTW_B3_D2

# --- B3 hyperparameter (re-exported for local scripts) ---
SCALE = B3_SCALE
