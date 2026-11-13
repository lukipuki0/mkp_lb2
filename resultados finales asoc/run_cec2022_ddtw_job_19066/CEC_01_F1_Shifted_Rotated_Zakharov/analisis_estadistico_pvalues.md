# Análisis Estadístico Inferencial — CEC2022 — F1_Shifted_Rotated_Zakharov

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DDTW`
- **Métrica:** Fitness — F1_Shifted_Rotated_Zakharov (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `146.5888`  |  p-value = `7.103576e-30`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `PSO` | 1.37 | `300.000000` | `0.000000` | `300.000000` | [300.0000, 300.0000] | `9.3126e-07` | `7.0013e-06` | `1.3812e-05` | **Peor (-) ***** |
| 2 | `EHO` | 1.84 | `300.000000` | `0.000000` | `300.000000` | [300.0000, 300.0000] | `9.7235e-06` | `6.9061e-06` | `1.3812e-05` | **Peor (-) ***** |
| 3 | **`Hybrid DDTW`** | 2.79 | `300.468821` | `2.609459` | `300.000000` | [299.5502, 301.3874] | `4.6423e-12` | `1.0000e+00` | `1.0000e+00` | **=** |
| 4 | `GWO` | 4.13 | `362.369193` | `47.613877` | `349.391110` | [345.6079, 379.1305] | `5.4872e-03` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |
| 5 | `ACO` | 4.87 | `624.791562` | `287.425476` | `535.641085` | [523.6102, 725.9729] | `1.2041e-03` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |
| 6 | `WOA` | 6.00 | `9391.783797` | `6952.541853` | `6800.717847` | [6944.3056, 11839.2620] | `5.2274e-04` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
