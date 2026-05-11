"""
lb2 – Módulo de binarización LB2 reutilizable.

Exporta las funciones de transferencia L1/L2, la rutina de
binarización `binarizar_posicion`, y las utilidades para la
transición de los parámetros G (lineal o adaptada por DTW).

Referencia: kirito/LB2_Binarization_Explicacion.md
            kirito/LB2_MKP.ipynb
"""

from lb2.binarization import (        # noqa: F401
    binarizar_posicion,
    interpolar_G,
    adaptar_G_por_dtw,
)
