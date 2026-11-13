# Análisis Estadístico Inferencial — CEC2022 — F5_Shifted_Rotated_Levy

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F5_Shifted_Rotated_Levy (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `107.6177`  |  p-value = `1.305573e-21`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `ACO` | 1.76 | `900.020432` | `0.083566` | `900.000000` | [899.9910, 900.0498] | `2.4331e-11` | `9.8402e-03` | `1.9680e-02` | **Peor (-) *** |
| 2 | **`Hybrid DTW`** | 2.23 | `901.672452` | `5.595086` | `900.000069` | [899.7028, 903.6421] | `8.0984e-11` | `1.0000e+00` | `1.0000e+00` | **=** |
| 3 | `PSO` | 2.85 | `901.039608` | `1.247313` | `900.454324` | [900.6005, 901.4787] | `7.9831e-05` | `2.7009e-02` | `2.7009e-02` | **Peor (-) *** |
| 4 | `GWO` | 3.74 | `905.148792` | `11.602828` | `900.611148` | [901.0643, 909.2333] | `2.8029e-09` | `2.1575e-03` | `6.4726e-03` | **Mejor (+) **** |
| 5 | `EHO` | 4.45 | `942.258965` | `66.361752` | `911.704274` | [918.8979, 965.6201] | `4.4957e-07` | `1.4847e-05` | `5.9390e-05` | **Mejor (+) ***** |
| 6 | `WOA` | 5.97 | `1482.675898` | `440.872772` | `1388.529592` | [1327.4770, 1637.8747] | `2.1168e-05` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
