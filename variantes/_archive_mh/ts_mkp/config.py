"""
ts_mkp/config.py
----------------
Hiperparametros del Algoritmo Tabu Search (TS) para el MKP.
"""

# ── Instancia ─────────────────────────────────────────────────────────────────
INSTANCE_URL   = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb1.txt"
INSTANCE_INDEX = 9

# ── Reproducibilidad ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Parametros del TS ─────────────────────────────────────────────────────────
EPOCHS          = 10     # Veces que se arranca la busqueda desde cero
ITERATIONS      = 2000   # Iteraciones max por epoch
TABU_TENURE     = 10     # Tamaño base de la lista tabu
NEIGHBORHOOD_SZ = 30     # Cuantos vecinos de 1-bit evaluar por iter (n=100 es caro evaluar todos)

# ── Stagnation Monitor ────────────────────────────────────────────────────────
USE_STAGNATION = True

STAG_WINDOW      = 25
STAG_BAND        = 3
STAG_MIN_SLOPE   = 1.0
STAG_PLATEAU_MAX = 8
STAG_PATIENCE    = 2
STAG_USE_DDTW    = True
STAG_ADAPT       = True
STAG_P_LOW       = 30.0
STAG_P_HIGH      = 70.0

# Variantes TS:
#   "random_restart"   (Original) -> Reinicio suave (limpia tabu + random walk corto)
#   "v1_exploit"       (V1)       -> Ignora lista Tabu (Tenure=0), explota agresivamente
#   "v2_cycle"         (V2)       -> Alterna entre Tenure=30 (explorar) y Tenure=2 (explotar)
#   "v3_explore"       (V3)       -> Reinicio fuerte masivo (solucion 100% random nueva)
#   "v4_nonlinear"     (V4)       -> Tabu Tenure crece exponencialmente
#   "v5_heuristic"     (V5)       -> Reinicia usando la densidad heuristica + ruido
#   "v6_lp"            (V6)       -> Resuelve LP y salta a la solucion LP
#   "v7_tabu_lp"       (V7)       -> Igual a V6 pero agrega variables forzadas a Tabu largo
#   "v8_ruin_recreate" (V8)       -> Destruye 50% de la solucion actual y reconstruye
STAG_STRATEGY  = "random_restart"
STAG_MAX_FIRES = 4

# ── Salida ────────────────────────────────────────────────────────────────────
GRAFICAR = True
