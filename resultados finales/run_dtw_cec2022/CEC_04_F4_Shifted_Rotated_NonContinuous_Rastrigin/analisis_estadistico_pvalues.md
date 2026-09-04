# Análisis Estadístico Inferencial — CEC2022 — F4_Shifted_Rotated_NonContinuous_Rastrigin

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F4_Shifted_Rotated_NonContinuous_Rastrigin (Minimización)
- **Friedman χ²:** `64.6654`  |  p-value = `1.314909e-12`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------|---------------|
| 1 | `GWO` | 1.87 | `813.723871` | `5.046631` | `812.083901` | [811.9473, 815.5004] | `9.9708e-04` | `1.6942e-01` | **Similar (=) ns** |
| 2 | **`Hybrid DTW`** | 2.73 | `816.302366` | `7.931841` | `815.000000` | [813.5101, 819.0946] | `1.3306e-01` | `1.0000e+00` | **=** |
| 3 | `PSO` | 3.45 | `818.903226` | `7.725951` | `817.000000` | [816.1835, 821.6230] | `1.7176e-02` | `5.4232e-02` | **Similar (=) ns** |
| 4 | `ACO` | 3.48 | `818.259348` | `3.403486` | `819.000000` | [817.0612, 819.4575] | `5.2909e-01` | `2.4744e-01` | **Similar (=) ns** |
| 5 | `EHO` | 4.05 | `824.725240` | `10.360581` | `821.000000` | [821.0780, 828.3724] | `2.8798e-02` | `2.6401e-03` | **Peor (-) **** |
| 6 | `WOA` | 5.42 | `845.414185` | `17.890064` | `845.149598` | [839.1164, 851.7120] | `7.0693e-01` | `4.5246e-06` | **Peor (-) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
