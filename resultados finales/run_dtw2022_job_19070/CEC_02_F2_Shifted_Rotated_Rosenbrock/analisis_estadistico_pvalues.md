# Análisis Estadístico Inferencial — CEC2022 — F2_Shifted_Rotated_Rosenbrock

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F2_Shifted_Rotated_Rosenbrock (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `32.2326`  |  p-value = `5.343123e-06`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `EHO` | 2.31 | `405.244922` | `3.132731` | `404.333182` | [404.1421, 406.3477] | `8.7962e-05` | `2.0122e-01` | `4.0244e-01` | **Similar (=) ns** |
| 2 | **`Hybrid DTW`** | 2.85 | `405.995119` | `3.766298` | `408.916102` | [404.6693, 407.3210] | `1.6623e-06` | `1.0000e+00` | `1.0000e+00` | **=** |
| 3 | `ACO` | 3.35 | `409.368794` | `12.820682` | `406.787122` | [404.8556, 413.8820] | `4.6757e-11` | `2.4736e-01` | `4.0244e-01` | **Similar (=) ns** |
| 4 | `WOA` | 3.87 | `419.622425` | `25.311566` | `408.917590` | [410.7121, 428.5328] | `7.6970e-06` | `9.7369e-03` | `2.9211e-02` | **Mejor (+) *** |
| 5 | `PSO` | 3.94 | `416.970040` | `24.108638` | `408.916102` | [408.4832, 425.4569] | `2.2714e-08` | `4.8096e-03` | `1.9238e-02` | **Mejor (+) *** |
| 6 | `GWO` | 4.68 | `416.113663` | `18.593271` | `409.021850` | [409.5683, 422.6590] | `1.7531e-08` | `7.6402e-04` | `3.8201e-03` | **Mejor (+) **** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
