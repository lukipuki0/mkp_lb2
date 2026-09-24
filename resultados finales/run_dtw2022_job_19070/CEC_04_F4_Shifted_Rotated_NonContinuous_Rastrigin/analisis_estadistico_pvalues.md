# Análisis Estadístico Inferencial — CEC2022 — F4_Shifted_Rotated_NonContinuous_Rastrigin

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F4_Shifted_Rotated_NonContinuous_Rastrigin (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `71.7841`  |  p-value = `4.356832e-14`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `GWO` | 1.81 | `813.723871` | `5.046631` | `812.083901` | [811.9473, 815.5004] | `9.9708e-04` | `3.5582e-02` | `1.0675e-01` | **Similar (=) ns** |
| 2 | **`Hybrid DTW`** | 2.68 | `816.257942` | `4.923583` | `815.000000` | [814.5247, 817.9912] | `1.8111e-01` | `1.0000e+00` | `1.0000e+00` | **=** |
| 3 | `PSO` | 3.42 | `818.903226` | `7.725951` | `817.000000` | [816.1835, 821.6230] | `1.7176e-02` | `1.0212e-01` | `1.3896e-01` | **Similar (=) ns** |
| 4 | `ACO` | 3.42 | `818.259348` | `3.403486` | `819.000000` | [817.0612, 819.4575] | `5.2909e-01` | `6.9481e-02` | `1.3896e-01` | **Similar (=) ns** |
| 5 | `EHO` | 4.16 | `824.725240` | `10.360581` | `821.000000` | [821.0780, 828.3724] | `2.8798e-02` | `1.5354e-04` | `6.1415e-04` | **Mejor (+) ***** |
| 6 | `WOA` | 5.52 | `845.414185` | `17.890064` | `845.149598` | [839.1164, 851.7120] | `7.0693e-01` | `1.5766e-06` | `7.8832e-06` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
