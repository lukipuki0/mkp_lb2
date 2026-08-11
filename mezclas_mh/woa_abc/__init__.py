"""
mezclas_mh/woa_abc
------------------
Módulo de hibridaciones de WOA y ABC (Variantes A, B, C y MDG-WABC).

Exporta los hiperparámetros y funciones de ejecución tanto para MKP como para el benchmark continuo.
"""

from mezclas_mh.woa_abc.variante_a import VariantAParams, VariantAEpochResult, ejecutar_epoch as variant_a_epoch, ejecutar_epoch_continuo as variant_a_epoch_continuo  # noqa: F401
from mezclas_mh.woa_abc.variante_b import VariantBParams, VariantBEpochResult, ejecutar_epoch as variant_b_epoch, ejecutar_epoch_continuo as variant_b_epoch_continuo  # noqa: F401
from mezclas_mh.woa_abc.variante_c import VariantCParams, VariantCEpochResult, ejecutar_epoch as variant_c_epoch, ejecutar_epoch_continuo as variant_c_epoch_continuo  # noqa: F401
from mezclas_mh.woa_abc.mdg_wabc import MDGWABCParams, MDGWABCEpochResult, ejecutar_epoch as mdg_wabc_epoch, ejecutar_epoch_continuo as mdg_wabc_epoch_continuo  # noqa: F401
