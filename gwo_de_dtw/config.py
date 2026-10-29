"""
All defaults of the GWO-DE hybrid in one place.

hybrid.py, controller.py and run.py take their defaults from here. Monitor
values follow hybrid_mkp (dtw_stagnation.StagnationConfig) except
MIN_SLOPE = 2.0 and USE_DDTW = True.

Spec: odd/tasks/gwo-de-dtw-hybrid.md (S12, S14, S15).
"""

# Instance
INSTANCE = "instancias/mknapcb1.txt"
INDEX = 0

# Run: stop after ITERATIONS generations; one run with SEED for now.
POP_SIZE = 30
ITERATIONS = 1000
SEED = 1

# Binarization: "lb2" (continuous space [-V_MAX, V_MAX]) or
# "sigmoid" (continuous space [-SIGMOID_V_MAX, SIGMOID_V_MAX])
BINARIZATION = "lb2"
SIGMOID_V_MAX = 6.0

# Continuous space [-V_MAX, V_MAX] and LB2 binarization
V_MAX = 100.0
G1 = 0.5
G2 = 0.5
# LB2 base flip probability per profile; same value in both = fixed G3
EXPLOIT_G3 = 0.1
EXPLORE_G3 = 0.1

# Joint GWO/DE profiles
EXPLOIT_A, EXPLOIT_F, EXPLOIT_CR = 0.5, 0.5, 0.9
EXPLORE_A, EXPLORE_F, EXPLORE_CR = 2.0, 0.9, 0.3

# Plot color of each profile (plots/)
PROFILE_COLORS = {"exploit": "#1f77b4", "explore": "#ff7f0e"}

# Stagnation monitor (monitor.StagnationConfig)
WINDOW = 40
BAND = 0              # 0 -> 10% of WINDOW (4)
MIN_SLOPE = 2.0
PLATEAU_MAX = 15
PATIENCE = 8
USE_DDTW = True
ADAPT_THRESHOLDS = True
P_LOW = 40.0
P_HIGH = 60.0
IMPROVEMENT_TOL = 0.0
