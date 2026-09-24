"""Compatibilidad para el WOA--ABC continuo usado por CEC2022.

La implementación vive en ``woa_abc`` para
mantener en un solo sitio la lógica cooperativa. Este módulo ofrece la
convención de importación del resto de las MH continuas.
"""

from woa_abc.cooperativo_cec_dtw import (
    CooperativeCECParams,
    CooperativeCECEpochResult,
    CooperativeCECResult,
    cooperative_cec_epoch,
    ejecutar_cec_cooperativo,
    ejecutar_epoch,
)

__all__ = [
    "CooperativeCECParams",
    "CooperativeCECEpochResult",
    "CooperativeCECResult",
    "cooperative_cec_epoch",
    "ejecutar_cec_cooperativo",
    "ejecutar_epoch",
]
