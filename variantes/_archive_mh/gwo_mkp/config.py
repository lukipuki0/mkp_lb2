"""
gwo_mkp/config.py
-----------------
Hiperparametros del Grey Wolf Optimizer (GWO) para el MKP.
Edita aqui para cambiar el comportamiento del GWO, la binarizacion LB2
y la deteccion de estancamiento.
"""

# ── Instancia ─────────────────────────────────────────────────────────────────
INSTANCE_URL   = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb1.txt"
INSTANCE_INDEX = 9    # Que instancia del archivo usar (0-indexed)

# ── Reproducibilidad ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Parametros del GWO ────────────────────────────────────────────────────────
POP_SIZE    = 30     # Numero de lobos en la manada
ITERATIONS  = 300    # Iteraciones por epoch
EPOCHS      = 10     # Veces que se re-arranca el GWO desde cero
V_MAX       = 6.0    # Velocidad maxima (clamp) para la binarizacion LB2

# ── Binarizacion LB2 ─────────────────────────────────────────────────────────
# Parametros G iniciales (exploracion) y finales (explotacion).
# La transicion puede ser lineal (ciega) o adaptativa via DTW.
G1_INITIAL = 0.5     # Pendiente inicial (baja = explorar)
G1_FINAL   = 1.0     # Pendiente final   (alta = explotar)
G2_INITIAL = 0.5     # Compresion inicial
G2_FINAL   = 7.2     # Compresion final
G3_INITIAL = 0.5     # Offset/piso inicial (alto = explorar)
G3_FINAL   = 0.0     # Offset/piso final   (bajo = explotar)

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

# Estrategia de rescate al detectar estancamiento:
#   "adapt_g"           (Original) -> Modulacion continua de G via DTW (4 estados)
#   "v1_exploit"        (V1)       -> Fuerza G a explotacion maxima + a=0.1
#   "v2_cycle"          (V2)       -> Alterna G exploratorios <-> G explotadores
#   "v3_explore"        (V3)       -> Fuerza G a exploracion maxima + 50% lobos nuevos
#   "v4_nonlinear"      (V4)       -> G3 crece exponencialmente con fires
#   "v5_heuristic"      (V5)       -> Reemplaza peores lobos con soluciones por densidad
#   "v6_lp"             (V6)       -> Muta posiciones guiadas por relajacion LP
#   "v7_tabu_lp"        (V7)       -> Igual a V6 pero protege dimensiones en lista tabu
#   "v8_ruin_recreate"  (V8)       -> Destruye 50% de bits activos y reconstruye
STAG_STRATEGY  = "adapt_g"
STAG_MAX_FIRES = 4

# ── Salida ────────────────────────────────────────────────────────────────────
GRAFICAR = True
