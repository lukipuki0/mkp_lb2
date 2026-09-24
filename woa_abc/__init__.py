"""Motores WOA--ABC adaptativos para CEC2022 y HRES2.

El solver MCDP se archivó en ``planes/futuro_mcdp`` y ya no forma parte de la
API activa de esta carpeta.
"""

from woa_abc.adaptive import VARIANT_NAMES

from woa_abc.cooperativo_cec_dtw import (
    CooperativeCECParams,
    CooperativeCECEpochResult,
    CooperativeCECResult,
    cooperative_cec_epoch,
    ejecutar_cec_cooperativo,
)
from woa_abc.cooperativo_hres2_dtw import (
    CooperativeHRES2EpochResult,
    CooperativeHRES2Params,
    CooperativeHRES2Result,
    ejecutar_hres2_cooperativo,
    ejecutar_hres2_epoch,
)

__all__ = [
    "CooperativeCECParams",
    "CooperativeCECEpochResult",
    "CooperativeCECResult",
    "CooperativeHRES2EpochResult",
    "CooperativeHRES2Params",
    "CooperativeHRES2Result",
    "VARIANT_NAMES",
    "cooperative_cec_epoch",
    "ejecutar_cec_cooperativo",
    "ejecutar_hres2_cooperativo",
    "ejecutar_hres2_epoch",
]
