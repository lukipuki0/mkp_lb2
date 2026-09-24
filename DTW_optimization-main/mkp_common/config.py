"""
Centralized configuration for ALL DTW strategy versions.
=========================================================
Change values HERE and they affect every strategy (vanilla, fire_binario,
fire_d2, sigmoid_delta, b3_d2) without touching individual config files.

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
NUM_ITERACIONES = 1000
EPOCHS = 31
VERBOSE = False
SEMILLA = 1   # None for real randomness across runs

# ═══════════════════════════════════════════════════════════════════════════
# DTW — base fields shared by all DTW-enabled strategies
# ═══════════════════════════════════════════════════════════════════════════
_DTW_BASE = dict(
    window=20,
    band=2,
    min_slope=2.0,
    use_ddtw=True,
    adapt_thresholds=True,
)

# --- Fire Binario (A4) — 3-condition fire: plateau + patience + no-improve ---
DTW_FIRE_BINARIO = StagnationConfig(
    **_DTW_BASE,
    plateau_max=4,
    patience=2,
)

# --- Fire D2 (A3) — D2-pure: fire when D2 <= theta_c ---
# plateau_max / patience are required by the monitor internally
# but are NOT used in the A3 decision function.
DTW_FIRE_D2 = StagnationConfig(
    **_DTW_BASE,
    plateau_max=4,
    patience=2,
)

# --- Sigmoid Delta (B1) — continuous sigmoid on delta / theta_delta ---
DTW_SIGMOID_DELTA = StagnationConfig(
    **_DTW_BASE,
    p_low=30.0,
    p_high=70.0,
)

# --- B3 D2-Direct — continuous intensity from D2 / (theta_c * scale) ---
DTW_B3_D2 = StagnationConfig(
    **_DTW_BASE,
    p_low=30.0,
    p_high=70.0,
)

# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# --- B1 Sigmoid Delta ---
B1_K = 5.0          # steepness (higher = sharper transition)
B1_CENTER = 0.5     # inflection point on r_balance scale
# r_balance = delta / (theta_delta + eps)
# intensity = 1 / (1 + exp(-K * (r_balance - CENTER)))

# --- B3 D2-Direct ---
B3_SCALE = 2.0
# intensity = 1 - clip(D2 / (theta_c * SCALE + eps), 0, 1)
# scale > 1 → more tolerant (needs stronger stagnation to explore)
# scale < 1 → more sensitive
# scale = 1 → D2 = theta_c gives intensity = 0 (pure exploit)
