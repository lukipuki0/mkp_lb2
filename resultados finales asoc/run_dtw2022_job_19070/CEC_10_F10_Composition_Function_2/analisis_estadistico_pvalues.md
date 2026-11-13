# Análisis Estadístico Inferencial — CEC2022 — F10_Composition_Function_2

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F10_Composition_Function_2 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `77.5069`  |  p-value = `2.786866e-15`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | **`Hybrid DTW`** | 2.00 | `2522.945165` | `46.728406` | `2500.513584` | [2506.4955, 2539.3948] | `4.3813e-09` | `1.0000e+00` | `1.0000e+00` | **=** |
| 2 | `ACO` | 2.48 | `2538.038196` | `59.831137` | `2500.480575` | [2516.9761, 2559.1003] | `4.0769e-08` | `2.6389e-01` | `2.6389e-01` | **Similar (=) ns** |
| 3 | `GWO` | 3.23 | `2589.451105` | `54.945240` | `2611.458107` | [2570.1089, 2608.7933] | `4.8020e-06` | `4.6194e-06` | `1.8477e-05` | **Mejor (+) ***** |
| 4 | `PSO` | 3.55 | `2603.304604` | `126.900065` | `2613.624374` | [2558.6324, 2647.9768] | `8.7161e-09` | `4.0052e-05` | `1.0710e-04` | **Mejor (+) ***** |
| 5 | `EHO` | 3.97 | `2610.018757` | `153.684613` | `2615.306918` | [2555.9177, 2664.1198] | `7.1338e-07` | `3.5700e-05` | `1.0710e-04` | **Mejor (+) ***** |
| 6 | `WOA` | 5.77 | `2893.668025` | `475.538929` | `2658.908756` | [2726.2658, 3061.0703] | `5.1488e-06` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
