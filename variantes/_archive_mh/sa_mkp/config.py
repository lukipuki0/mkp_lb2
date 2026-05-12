"""
config.py
─────────
Parámetros centralizados del algoritmo Simulated Annealing para MKP.
Modifica este archivo para cambiar la instancia o los hiperparámetros.
"""

# ── Fuente de datos ──────────────────────────────────────────────────────────
INSTANCE_URL: str = (
    "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb1.txt"
)

# Índice de la instancia a resolver (0-based dentro del archivo).
# El notebook original usa la instancia 9.
INSTANCE_INDEX: int = 9

# ── Hiperparámetros SA ───────────────────────────────────────────────────────
T_INICIAL: float = 5_000.0   # Temperatura inicial
T_FINAL: float   = 1.0       # Temperatura de parada
ALPHA: float     = 0.97      # Factor de enfriamiento geométrico
ITER_POR_T: int  = 50        # Iteraciones internas por temperatura
NUM_FLIP: int    = 3         # Bits a perturbar por vecino
EPOCHS: int      = 10        # Número de reinicios independientes

# ── Reproducibilidad ─────────────────────────────────────────────────────────
RANDOM_SEED: int | None = None   # None → no fijar semilla; int → reproducible

# ── StagnationMonitor (DTW) ─────────────────────────────────────────────────
# Activa la detección de estancamiento dentro de cada epoch.
USE_STAGNATION: bool  = True    # False → desactivar el monitor completamente

STAG_WINDOW:      int   = 30    # Tamaño de ventana de análisis
STAG_BAND:        int   = 0     # Banda Sakoe-Chiba (0 → 10% de window)
STAG_MIN_SLOPE:   float = 0.0   # Pendiente mínima de la rampa (0 → auto)
STAG_PLATEAU_MAX: int   = 15    # Iteraciones sin mejora para considerar meseta
STAG_PATIENCE:    int   = 3     # Epochs consecutivos de trigger para disparar
STAG_USE_DDTW:    bool  = False # True → usar DDTW (derivada) en vez de DTW
STAG_ADAPT:       bool  = True  # Umbrales adaptativos por percentiles
STAG_P_LOW:       float = 30.0
STAG_P_HIGH:      float = 70.0

# Estrategia al detectar estancamiento:
#   "reheat"             (Original) -> sube la temperatura al % de T_inicial indicado
#   "random"                        -> reinicia con solución aleatoria completamente nueva
#   "v1_exploit"         (V1)       -> baja T y reduce vecindario (explotación)
#   "v2_cycle"           (V2)       -> alterna exploración y explotación
#   "v3_explore"         (V3)       -> sube T y amplía vecindario
#   "v4_nonlinear"       (V4)       -> T decae exponencialmente según los rescates
#   "v5_heuristic"       (V5)       -> reconstruye guiado por densidad (ganancia/peso)
#   "v6_lp"              (V6)       -> reconstruye guiado por precios sombra de relajación LP
#   "v7_tabu_lp"         (V7)       -> igual a V6 pero con memoria tabú temporal
#   "v8_ruin_recreate"   (V8)       -> destruye el 50% y reconstruye vorazmente
STAG_STRATEGY:        str   = "reheat"
STAG_REHEAT_FRACTION: float = 0.4  # % de T_inicial al recalentar
STAG_MAX_FIRES:       int   = 3    # Maximo de recalentamientos por epoch (0 = ilimitado)

# ── Salida ───────────────────────────────────────────────────────────────────
CONVERGENCE_PLOT_PATH: str = "convergencia_SA_MKP.png"
PLOT_DPI: int = 150
