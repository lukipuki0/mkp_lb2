# Análisis Estadístico Inferencial — CEC2022 — F12_Composition_Function_4

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DDTW`
- **Métrica:** Fitness — F12_Composition_Function_4 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `106.8727`  |  p-value = `1.875605e-21`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | **`Hybrid DDTW`** | 2.02 | `2863.757163` | `1.357503` | `2863.714850` | [2863.2793, 2864.2350] | `2.3808e-01` | `1.0000e+00` | `1.0000e+00` | **=** |
| 2 | `GWO` | 2.32 | `2864.652968` | `4.031071` | `2863.874578` | [2863.2339, 2866.0720] | `4.3397e-10` | `2.0231e-01` | `2.0231e-01` | **Similar (=) ns** |
| 3 | `ACO` | 2.92 | `2864.769839` | `2.306446` | `2863.924512` | [2863.9579, 2865.5818] | `4.3205e-09` | `2.5633e-02` | `7.6900e-02` | **Similar (=) ns** |
| 4 | `EHO` | 2.94 | `2865.020990` | `2.694695` | `2864.921942` | [2864.0724, 2865.9696] | `1.8651e-04` | `5.0248e-02` | `1.0050e-01` | **Similar (=) ns** |
| 5 | `PSO` | 4.87 | `2879.459334` | `25.550102` | `2871.544850` | [2870.4650, 2888.4536] | `1.2414e-08` | `1.4294e-06` | `5.7175e-06` | **Mejor (+) ***** |
| 6 | `WOA` | 5.94 | `2958.748087` | `57.304662` | `2950.915977` | [2938.5753, 2978.9208] | `1.7969e-01` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
