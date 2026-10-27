"""
Centralized configuration for ALL DTW strategy versions.
=========================================================
Change values HERE and they affect every strategy (vanilla, fire_d2)
without touching individual config files.

Each strategy's own config.py imports from here and adds only what is
specific to that strategy (MH_CLASS, DTW_CFG, hypers, fire_fn).
"""

import os

from .monitor import StagnationConfig

# ═══════════════════════════════════════════════════════════════════════════
# INSTANCE (overridable via environment variables)
# ═══════════════════════════════════════════════════════════════════════════
RUTA_INSTANCIA = os.environ.get("MKP_INSTANCIA", "instances/mknapcb4.txt")
INDICE_INSTANCIA = int(os.environ.get("MKP_INDICE", "0"))

# ═══════════════════════════════════════════════════════════════════════════
# POPULATION & BUDGET — shared across all strategies
# ═══════════════════════════════════════════════════════════════════════════
NUM_PARTICULAS = 20
NUM_ITERACIONES = 2000 #1000
EPOCHS = 31 #31
VERBOSE = False
SEMILLA = 1   # None for real randomness across runs

# ═══════════════════════════════════════════════════════════════════════════
# DTW — base fields shared by all DTW-enabled strategies
# ═══════════════════════════════════════════════════════════════════════════
_DTW_BASE = dict(
    window=200,
    band=2,
    min_slope=2.0,
    use_ddtw=True,
    adapt_thresholds=True,
    p_low=40.0,   # Percentil para theta_c (estancamiento / meseta D2)
    p_high=60.0,  # Percentil para theta_r y theta_delta (progreso / rampa D1)
)

# --- Fire D2 (A3) — D2-pure: fire when D2 <= theta_c ---
# plateau_max / patience are required by the monitor internally
# but are NOT used in the A3 decision function.
DTW_FIRE_D2 = StagnationConfig(
    **_DTW_BASE,
    plateau_max=5,
    patience=3,
)
