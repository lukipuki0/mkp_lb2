# Análisis Estadístico Inferencial — CEC2022 — F5_Shifted_Rotated_Levy

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DDTW`
- **Métrica:** Fitness — F5_Shifted_Rotated_Levy (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `107.9151`  |  p-value = `1.129760e-21`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `ACO` | 1.77 | `900.020432` | `0.083566` | `900.000000` | [899.9910, 900.0498] | `2.4331e-11` | `1.6810e-02` | `3.3620e-02` | **Peor (-) *** |
| 2 | **`Hybrid DDTW`** | 2.18 | `901.972045` | `5.980934` | `900.000037` | [899.8666, 904.0775] | `2.1375e-10` | `1.0000e+00` | `1.0000e+00` | **=** |
| 3 | `PSO` | 2.85 | `901.039608` | `1.247313` | `900.454324` | [900.6005, 901.4787] | `7.9831e-05` | `5.4424e-02` | `5.4424e-02` | **Similar (=) ns** |
| 4 | `GWO` | 3.81 | `905.148792` | `11.602828` | `900.611148` | [901.0643, 909.2333] | `2.8029e-09` | `6.4402e-04` | `1.9320e-03` | **Mejor (+) **** |
| 5 | `EHO` | 4.42 | `942.258965` | `66.361752` | `911.704274` | [918.8979, 965.6201] | `4.4957e-07` | `1.9360e-05` | `7.7439e-05` | **Mejor (+) ***** |
| 6 | `WOA` | 5.97 | `1482.675898` | `440.872772` | `1388.529592` | [1327.4770, 1637.8747] | `2.1168e-05` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
