# Análisis Estadístico Inferencial — CEC2022 — F4_Shifted_Rotated_NonContinuous_Rastrigin

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DDTW`
- **Métrica:** Fitness — F4_Shifted_Rotated_NonContinuous_Rastrigin (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `69.0529`  |  p-value = `1.613123e-13`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `GWO` | 1.82 | `813.723871` | `5.046631` | `812.083901` | [811.9473, 815.5004] | `9.9708e-04` | `7.8647e-02` | `1.5729e-01` | **Similar (=) ns** |
| 2 | **`Hybrid DDTW`** | 2.68 | `815.741935` | `4.946162` | `816.000000` | [814.0008, 817.4831] | `8.1125e-01` | `1.0000e+00` | `1.0000e+00` | **=** |
| 3 | `PSO` | 3.35 | `818.903226` | `7.725951` | `817.000000` | [816.1835, 821.6230] | `1.7176e-02` | `1.5577e-01` | `1.5729e-01` | **Similar (=) ns** |
| 4 | `ACO` | 3.55 | `818.259348` | `3.403486` | `819.000000` | [817.0612, 819.4575] | `5.2909e-01` | `1.4852e-02` | `4.4556e-02` | **Mejor (+) *** |
| 5 | `EHO` | 4.15 | `824.725240` | `10.360581` | `821.000000` | [821.0780, 828.3724] | `2.8798e-02` | `3.6276e-05` | `1.4510e-04` | **Mejor (+) ***** |
| 6 | `WOA` | 5.45 | `845.414185` | `17.890064` | `845.149598` | [839.1164, 851.7120] | `7.0693e-01` | `2.1112e-06` | `1.0556e-05` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
