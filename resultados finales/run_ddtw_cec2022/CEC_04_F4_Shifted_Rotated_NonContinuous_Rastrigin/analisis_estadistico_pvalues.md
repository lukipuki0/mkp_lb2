# Análisis Estadístico Inferencial — CEC2022 — F4_Shifted_Rotated_NonContinuous_Rastrigin

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F4_Shifted_Rotated_NonContinuous_Rastrigin (Minimización)
- **Friedman χ²:** `67.5628`  |  p-value = `3.291881e-13`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------|---------------|
| 1 | `GWO` | 1.81 | `813.723871` | `5.046631` | `812.083901` | [811.9473, 815.5004] | `9.9708e-04` | `2.6009e-02` | **Mejor (+) *** |
| 2 | **`Hybrid DTW`** | 2.81 | `816.967742` | `6.057964` | `818.000000` | [814.8352, 819.1003] | `2.6467e-01` | `1.0000e+00` | **=** |
| 3 | `PSO` | 3.44 | `818.903226` | `7.725951` | `817.000000` | [816.1835, 821.6230] | `1.7176e-02` | `1.9489e-01` | **Similar (=) ns** |
| 4 | `ACO` | 3.45 | `818.259348` | `3.403486` | `819.000000` | [817.0612, 819.4575] | `5.2909e-01` | `4.6776e-01` | **Similar (=) ns** |
| 5 | `EHO` | 4.02 | `824.725240` | `10.360581` | `821.000000` | [821.0780, 828.3724] | `2.8798e-02` | `2.2971e-03` | **Peor (-) **** |
| 6 | `WOA` | 5.48 | `845.414185` | `17.890064` | `845.149598` | [839.1164, 851.7120] | `7.0693e-01` | `3.0734e-08` | **Peor (-) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
