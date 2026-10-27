"""
Fire D2 (A3) config — D2-pure fire (fire when D2 <= theta_c).

All shared values come from mkp_common.config.
Change things there to affect ALL strategies at once.
"""

from mkp_common import BinaryDE, BinaryGWO, BinaryPSO, GeneticAlgorithm
from mkp_common.config import (
    RUTA_INSTANCIA,
    INDICE_INSTANCIA,
    NUM_PARTICULAS,
    NUM_ITERACIONES,
    EPOCHS,
    SEMILLA,
    VERBOSE,
    DTW_FIRE_D2,
)

# --- Default MH for single-run scripts ---
MH_CLASS = BinaryDE

# --- DTW config for this strategy ---
DTW_CFG = DTW_FIRE_D2


# --- A3 decision function ---
def fire_d2(out: dict) -> bool:
    """
    Estrategia A3: fire if D2 <= theta_c.
    "Is the fitness curve flat?"

    - D2 low → window is a plateau → stagnation → fire=True
    - theta_c auto-adapts to D2 history (percentile p_low)
    """
    return out["D2_vs_const"] <= out["theta_c"]
