# Análisis Estadístico Inferencial — CEC2022 — F3_Shifted_Rotated_Expanded_Schaffers_F7

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F3_Shifted_Rotated_Expanded_Schaffers_F7 (Minimización)
- **Friedman χ²:** `116.5037`  |  p-value = `1.725869e-23`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------|---------------|
| 1 | `ACO` | 1.19 | `600.238870` | `0.920769` | `600.009712` | [599.9147, 600.5630] | `2.1549e-11` | `7.0967e-07` | **Mejor (+) ***** |
| 2 | **`Hybrid DTW`** | 2.95 | `603.252574` | `3.657200` | `601.913215` | [601.9651, 604.5400] | `2.2514e-04` | `1.0000e+00` | **=** |
| 3 | `GWO` | 3.00 | `603.519268` | `3.656066` | `602.153962` | [602.2322, 604.8063] | `1.4905e-04` | `5.5514e-01` | **Similar (=) ns** |
| 4 | `PSO` | 3.35 | `606.579164` | `8.166518` | `603.910822` | [603.7043, 609.4540] | `3.0289e-06` | `1.1561e-01` | **Similar (=) ns** |
| 5 | `EHO` | 4.50 | `610.754948` | `7.226694` | `609.802494` | [608.2110, 613.2989] | `6.5421e-02` | `4.0715e-05` | **Peor (-) ***** |
| 6 | `WOA` | 6.00 | `642.164910` | `14.104227` | `640.945365` | [637.1998, 647.1300] | `5.0555e-01` | `9.3132e-10` | **Peor (-) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
