"""
Sigmoid Delta (B1) config — continuous sigmoid adaptation on delta/theta_delta.

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
    DTW_SIGMOID_DELTA,
    B1_K,
    B1_CENTER,
)

from mkp_common import BinaryPSO

# --- Default MH for single-run scripts ---
MH_CLASS = BinaryPSO

# --- DTW config for this strategy ---
DTW_CFG = DTW_SIGMOID_DELTA

# --- B1 sigmoid hyperparameters (re-exported for local scripts) ---
K = B1_K
CENTER = B1_CENTER
