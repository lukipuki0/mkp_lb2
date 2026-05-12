"""
ga_mkp/config.py
----------------
Hiperparametros del Algoritmo Genetico para el MKP.
Edita aqui para cambiar el comportamiento del GA y la deteccion de estancamiento.
"""

# ── Instancia ─────────────────────────────────────────────────────────────────
INSTANCE_URL   = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb1.txt"
INSTANCE_INDEX = 9    # Que instancia del archivo usar (0-indexed)

# ── Reproducibilidad ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Parametros del GA ─────────────────────────────────────────────────────────
POP_SIZE        = 60     # Numero de individuos en la poblacion
GENERATIONS     = 300    # Generaciones por epoch
EPOCHS          = 10     # Veces que se re-arranca el GA desde cero
ELITISM         = 2      # Cuantos mejores pasan directamente a la siguiente gen
TOURNAMENT_SIZE = 3      # Competidores en el torneo de seleccion

# Tasas de operadores por defecto
CROSSOVER_RATE  = 0.85   # Probabilidad de cruzar dos padres (vs. clonar)
MUTATION_RATE   = 0.04   # Probabilidad por bit de mutar (bitflip) o de swapear

# Operadores por defecto:
#   crossover: "uniform" | "1point" | "2point"
#   mutation:  "bitflip" | "swap"
DEFAULT_CROSSOVER = "uniform"
DEFAULT_MUTATION  = "bitflip"

# ── Stagnation Monitor ────────────────────────────────────────────────────────
USE_STAGNATION = True

STAG_WINDOW      = 25    # Ventana de generaciones a observar
STAG_BAND        = 3
STAG_MIN_SLOPE   = 1.0
STAG_PLATEAU_MAX = 8
STAG_PATIENCE    = 2
STAG_USE_DDTW    = True
STAG_ADAPT       = True
STAG_P_LOW       = 30.0
STAG_P_HIGH      = 70.0

# Estrategia de rescate al detectar estancamiento:
#   "hypermutation"   (Original) -> Sube mutation_rate al 20% temporalmente
#   "v1_exploit"      (V1)       -> Cruce 1point + Hill Climbing al top 10%
#   "v2_cycle"        (V2)       -> Alterna (uniform+bitflip) <-> (2point+swap)
#   "v3_explore"      (V3)       -> Cruce 2point + swap + 50% inmigrantes
#   "v4_nonlinear"    (V4)       -> Mutacion exponencial (sube con fires)
#   "v5_heuristic"    (V5)       -> Inyecta individuos heuristicos en la peor mitad
#   "v6_lp"           (V6)       -> Mutacion guiada por precios sombra (LP)
#   "v7_tabu_lp"      (V7)       -> Igual a V6 con memoria tabu de genes
#   "v8_ruin_recreate"(V8)       -> Ruin & Recreate al top 30% de la poblacion
STAG_STRATEGY  = "hypermutation"
STAG_MAX_FIRES = 3       # Maximo de rescates por epoch (0 = ilimitado)

# ── Salida ────────────────────────────────────────────────────────────────────
GRAFICAR = True
