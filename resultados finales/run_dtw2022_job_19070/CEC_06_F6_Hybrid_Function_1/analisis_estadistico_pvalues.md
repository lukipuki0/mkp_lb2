# Análisis Estadístico Inferencial — CEC2022 — F6_Hybrid_Function_1

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F6_Hybrid_Function_1 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `34.1336`  |  p-value = `2.239498e-06`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `WOA` | 2.68 | `3535.329296` | `1840.390742` | `2933.601869` | [2887.4632, 4183.1954] | `4.4217e-05` | `6.2194e-01` | `1.0000e+00` | **Similar (=) ns** |
| 2 | **`Hybrid DTW`** | 2.94 | `3488.596966` | `1821.985314` | `2718.090658` | [2847.2100, 4129.9839] | `1.6304e-04` | `1.0000e+00` | `1.0000e+00` | **=** |
| 3 | `EHO` | 3.13 | `3829.465392` | `1767.135167` | `3830.716717` | [3207.3872, 4451.5436] | `6.3831e-03` | `5.5514e-01` | `1.0000e+00` | **Similar (=) ns** |
| 4 | `PSO` | 3.16 | `3947.251652` | `2174.028503` | `2792.084620` | [3181.9362, 4712.5671] | `1.1877e-04` | `1.8208e-01` | `5.4623e-01` | **Similar (=) ns** |
| 5 | `GWO` | 4.10 | `5093.885833` | `2635.228745` | `4801.590188` | [4166.2158, 6021.5559] | `7.7851e-05` | `1.3199e-02` | `5.2794e-02` | **Similar (=) ns** |
| 6 | `ACO` | 5.00 | `15608.376822` | `33620.129927` | `6944.202899` | [3773.2038, 27443.5499] | `1.6780e-10` | `4.6194e-06` | `2.3097e-05` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
