"""
mezclas_mh/woa_abc/algoritmos
-----------------------------
Módulo que contiene las implementaciones de los algoritmos de hibridación WOA-ABC.
"""

from mezclas_mh.woa_abc.algoritmos.variante_a import (  # noqa: F401
    VariantAParams,
    VariantAEpochResult,
    ejecutar_epoch as variant_a_epoch,
    ejecutar_epoch_continuo as variant_a_epoch_continuo,
)
from mezclas_mh.woa_abc.algoritmos.variante_b import (  # noqa: F401
    VariantBParams,
    VariantBEpochResult,
    ejecutar_epoch as variant_b_epoch,
    ejecutar_epoch_continuo as variant_b_epoch_continuo,
)
from mezclas_mh.woa_abc.algoritmos.variante_c import (  # noqa: F401
    VariantCParams,
    VariantCEpochResult,
    ejecutar_epoch as variant_c_epoch,
    ejecutar_epoch_continuo as variant_c_epoch_continuo,
)
from mezclas_mh.woa_abc.algoritmos.mdg_wabc import (  # noqa: F401
    MDGWABCParams,
    MDGWABCEpochResult,
    ejecutar_epoch as mdg_wabc_epoch,
    ejecutar_epoch_continuo as mdg_wabc_epoch_continuo,
)
from mezclas_mh.woa_abc.algoritmos.variante_d_dtw import (  # noqa: F401
    DTWWOAABCParams,
    DTWWOAABCEpochResult,
    ejecutar_epoch as variant_d_epoch,
    ejecutar_epoch_continuo as variant_d_epoch_continuo,
)
