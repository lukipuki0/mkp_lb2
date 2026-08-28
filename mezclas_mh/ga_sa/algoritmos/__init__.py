"""
mezclas_mh/ga_sa/algoritmos
---------------------------
Subpaquete con las implementaciones de las 5 variantes híbridas GA-SA.
"""

from mezclas_mh.ga_sa.algoritmos.variante_a import (  # noqa: F401
    VariantAParams,
    VariantAEpochResult,
    ejecutar_epoch as variant_a_epoch,
    ejecutar_epoch_continuo as variant_a_epoch_continuo,
)
from mezclas_mh.ga_sa.algoritmos.variante_b import (  # noqa: F401
    VariantBParams,
    VariantBEpochResult,
    ejecutar_epoch as variant_b_epoch,
    ejecutar_epoch_continuo as variant_b_epoch_continuo,
)
from mezclas_mh.ga_sa.algoritmos.variante_c import (  # noqa: F401
    VariantCParams,
    VariantCEpochResult,
    ejecutar_epoch as variant_c_epoch,
    ejecutar_epoch_continuo as variant_c_epoch_continuo,
)
from mezclas_mh.ga_sa.algoritmos.mdg_gasa import (  # noqa: F401
    MDGGASAParams,
    MDGGASAEpochResult,
    ejecutar_epoch as mdg_gasa_epoch,
    ejecutar_epoch_continuo as mdg_gasa_epoch_continuo,
)
from mezclas_mh.ga_sa.algoritmos.variante_e_dtw import (  # noqa: F401
    DTWGASAParams,
    DTWGASAEpochResult,
    ejecutar_epoch as variant_e_epoch,
    ejecutar_epoch_continuo as variant_e_epoch_continuo,
)
