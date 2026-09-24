# Análisis Estadístico Inferencial — CEC2022 — F9_Composition_Function_1

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F9_Composition_Function_1 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `118.4173`  |  p-value = `6.790451e-24`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `EHO` | 1.82 | `2529.284383` | `0.000000` | `2529.284383` | [2529.2844, 2529.2844] | `1.0000e+00` | `3.1731e-01` | `3.1731e-01` | **Similar (=) ns** |
| 2 | **`Hybrid DTW`** | 1.89 | `2529.284383` | `0.000000` | `2529.284383` | [2529.2844, 2529.2844] | `4.6359e-12` | `1.0000e+00` | `1.0000e+00` | **=** |
| 3 | `PSO` | 3.03 | `2532.767251` | `17.518701` | `2529.284383` | [2526.6002, 2538.9343] | `6.7894e-12` | `3.6038e-03` | `7.2076e-03` | **Mejor (+) **** |
| 4 | `ACO` | 3.65 | `2529.284383` | `0.000000` | `2529.284383` | [2529.2844, 2529.2844] | `1.3760e-06` | `2.1106e-05` | `6.3318e-05` | **Peor (-) ***** |
| 5 | `GWO` | 5.16 | `2540.916918` | `36.965163` | `2529.435554` | [2527.9042, 2553.9296] | `1.1965e-10` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |
| 6 | `WOA` | 5.45 | `2547.356146` | `34.502985` | `2529.845919` | [2535.2102, 2559.5021] | `4.0727e-08` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
