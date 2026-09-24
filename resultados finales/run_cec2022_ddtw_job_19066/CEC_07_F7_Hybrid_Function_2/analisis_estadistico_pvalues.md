# Análisis Estadístico Inferencial — CEC2022 — F7_Hybrid_Function_2

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DDTW`
- **Métrica:** Fitness — F7_Hybrid_Function_2 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `78.1152`  |  p-value = `2.079650e-15`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | **`Hybrid DDTW`** | 2.16 | `2022.698363` | `9.806440` | `2021.989966` | [2019.2462, 2026.1505] | `2.1950e-03` | `1.0000e+00` | `1.0000e+00` | **=** |
| 2 | `PSO` | 2.81 | `2026.262947` | `8.461329` | `2024.974868` | [2023.2843, 2029.2416] | `4.4377e-02` | `6.6421e-02` | `1.3284e-01` | **Similar (=) ns** |
| 3 | `ACO` | 2.87 | `2028.819996` | `22.083139` | `2024.535839` | [2021.0461, 2036.5938] | `3.6473e-11` | `1.1561e-01` | `1.3284e-01` | **Similar (=) ns** |
| 4 | `EHO` | 3.45 | `2033.226916` | `22.170281` | `2026.040318` | [2025.4224, 2041.0314] | `3.9823e-09` | `3.3502e-03` | `1.0051e-02` | **Mejor (+) *** |
| 5 | `GWO` | 3.74 | `2031.485879` | `8.615513` | `2031.613287` | [2028.4530, 2034.5188] | `7.9871e-01` | `2.5049e-03` | `1.0020e-02` | **Mejor (+) *** |
| 6 | `WOA` | 5.97 | `2129.259508` | `49.039741` | `2120.334590` | [2111.9962, 2146.5228] | `2.5544e-01` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
