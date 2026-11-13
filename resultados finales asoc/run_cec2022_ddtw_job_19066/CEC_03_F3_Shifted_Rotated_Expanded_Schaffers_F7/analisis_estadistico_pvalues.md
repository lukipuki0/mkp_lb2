# Análisis Estadístico Inferencial — CEC2022 — F3_Shifted_Rotated_Expanded_Schaffers_F7

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DDTW`
- **Métrica:** Fitness — F3_Shifted_Rotated_Expanded_Schaffers_F7 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `118.4133`  |  p-value = `6.803888e-24`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `ACO` | 1.16 | `600.238870` | `0.920769` | `600.009712` | [599.9147, 600.5630] | `2.1549e-11` | `7.0967e-07` | `2.8387e-06` | **Peor (-) ***** |
| 2 | **`Hybrid DDTW`** | 2.85 | `603.687758` | `6.050480` | `601.080270` | [601.5578, 605.8177] | `1.0736e-07` | `1.0000e+00` | `1.0000e+00` | **=** |
| 3 | `GWO` | 3.03 | `603.519268` | `3.656066` | `602.153962` | [602.2322, 604.8063] | `1.4905e-04` | `4.2138e-01` | `4.2138e-01` | **Similar (=) ns** |
| 4 | `PSO` | 3.45 | `606.579164` | `8.166518` | `603.910822` | [603.7043, 609.4540] | `3.0289e-06` | `8.2860e-02` | `1.6572e-01` | **Similar (=) ns** |
| 5 | `EHO` | 4.50 | `610.754948` | `7.226694` | `609.802494` | [608.2110, 613.2989] | `6.5421e-02` | `3.7243e-05` | `1.1173e-04` | **Mejor (+) ***** |
| 6 | `WOA` | 6.00 | `642.164910` | `14.104227` | `640.945365` | [637.1998, 647.1300] | `5.0555e-01` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
