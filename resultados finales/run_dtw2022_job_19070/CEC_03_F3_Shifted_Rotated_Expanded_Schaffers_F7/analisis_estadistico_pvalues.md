# Análisis Estadístico Inferencial — CEC2022 — F3_Shifted_Rotated_Expanded_Schaffers_F7

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F3_Shifted_Rotated_Expanded_Schaffers_F7 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `119.2396`  |  p-value = `4.547501e-24`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `ACO` | 1.19 | `600.238870` | `0.920769` | `600.009712` | [599.9147, 600.5630] | `2.1549e-11` | `8.0196e-06` | `2.4059e-05` | **Peor (-) ***** |
| 2 | **`Hybrid DTW`** | 2.74 | `603.534403` | `5.945872` | `601.416843` | [601.4413, 605.6275] | `6.2707e-08` | `1.0000e+00` | `1.0000e+00` | **=** |
| 3 | `GWO` | 3.03 | `603.519268` | `3.656066` | `602.153962` | [602.2322, 604.8063] | `1.4905e-04` | `3.8838e-01` | `3.8838e-01` | **Similar (=) ns** |
| 4 | `PSO` | 3.48 | `606.579164` | `8.166518` | `603.910822` | [603.7043, 609.4540] | `3.0289e-06` | `3.2111e-02` | `6.4223e-02` | **Similar (=) ns** |
| 5 | `EHO` | 4.55 | `610.754948` | `7.226694` | `609.802494` | [608.2110, 613.2989] | `6.5421e-02` | `5.3179e-06` | `2.1271e-05` | **Mejor (+) ***** |
| 6 | `WOA` | 6.00 | `642.164910` | `14.104227` | `640.945365` | [637.1998, 647.1300] | `5.0555e-01` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
