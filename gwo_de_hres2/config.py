"""
Defaults of the GWO-DE hybrid on HRES2.

Profiles (a, F, CR) come from gwo_de_dtw.config; G3 is not used here. The
monitor follows the HRES2 paper nominal configuration
(HRES2-H2/benchmark_hres2.py), not the MKP one (whose min_slope 2.0 is far
above an LCOE in CNY/kWh).

Spec: odd/tasks/gwo-de-hres2.md (S4, S5).
"""

from dtw_stagnation import StagnationConfig

# Run: stop after ITERATIONS generations.
POP_SIZE = 30
ITERATIONS = 1000
SEED = 1

# "ddtw" or "dtw"
SIGNAL = "ddtw"

# Stagnation monitor (HRES2 paper nominal)
WINDOW = 40
BAND = 4
MIN_SLOPE = 0.1
PLATEAU_MAX = 15
PATIENCE = 8
ADAPT_THRESHOLDS = True
P_LOW = 30.0
P_HIGH = 70.0
IMPROVEMENT_TOL = 0.0


def monitor_config(use_ddtw: bool) -> StagnationConfig:
    return StagnationConfig(window=WINDOW, band=BAND, min_slope=MIN_SLOPE,
                            plateau_max=PLATEAU_MAX, patience=PATIENCE, use_ddtw=use_ddtw,
                            adapt_thresholds=ADAPT_THRESHOLDS, p_low=P_LOW, p_high=P_HIGH,
                            improvement_tol=IMPROVEMENT_TOL)
