# Análisis Estadístico Inferencial — CEC2022 — F6_Hybrid_Function_1

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DDTW`
- **Métrica:** Fitness — F6_Hybrid_Function_1 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `32.4378`  |  p-value = `4.865582e-06`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | `WOA` | 2.68 | `3535.329296` | `1840.390742` | `2933.601869` | [2887.4632, 4183.1954] | `4.4217e-05` | `5.9481e-01` | `7.9841e-01` | **Similar (=) ns** |
| 2 | **`Hybrid DDTW`** | 2.97 | `3532.529734` | `1942.164049` | `2738.486063` | [2848.8367, 4216.2227] | `4.3266e-05` | `1.0000e+00` | `1.0000e+00` | **=** |
| 3 | `PSO` | 3.16 | `3947.251652` | `2174.028503` | `2792.084620` | [3181.9362, 4712.5671] | `1.1877e-04` | `1.8865e-01` | `5.6596e-01` | **Similar (=) ns** |
| 4 | `EHO` | 3.16 | `3829.465392` | `1767.135167` | `3830.716717` | [3207.3872, 4451.5436] | `6.3831e-03` | `3.9921e-01` | `7.9841e-01` | **Similar (=) ns** |
| 5 | `GWO` | 4.06 | `5093.885833` | `2635.228745` | `4801.590188` | [4166.2158, 6021.5559] | `7.7851e-05` | `1.7662e-02` | `7.0646e-02` | **Similar (=) ns** |
| 6 | `ACO` | 4.97 | `15608.376822` | `33620.129927` | `6944.202899` | [3773.2038, 27443.5499] | `1.6780e-10` | `9.1624e-06` | `4.5812e-05` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
