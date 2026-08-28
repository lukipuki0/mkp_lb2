"""
mezclas_mh/ga_sa
----------------
Módulo principal de hibridaciones GA-SA (Algoritmo Genético + Simulated Annealing).

Exporta las 5 variantes y sus hiperparámetros para problemas discretos (MKP)
y problemas continuos / reales (CEC2022, HRES, Hypertuning).
"""

from mezclas_mh.ga_sa.algoritmos import (  # noqa: F401
    VariantAParams,
    VariantAEpochResult,
    variant_a_epoch,
    variant_a_epoch_continuo,
    VariantBParams,
    VariantBEpochResult,
    variant_b_epoch,
    variant_b_epoch_continuo,
    VariantCParams,
    VariantCEpochResult,
    variant_c_epoch,
    variant_c_epoch_continuo,
    MDGGASAParams,
    MDGGASAEpochResult,
    mdg_gasa_epoch,
    mdg_gasa_epoch_continuo,
    DTWGASAParams,
    DTWGASAEpochResult,
    variant_e_epoch,
    variant_e_epoch_continuo,
)
