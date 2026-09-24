# Análisis Estadístico Inferencial — CEC2022 — F7_Hybrid_Function_2

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F7_Hybrid_Function_2 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `81.3779`  |  p-value = `4.320220e-16`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | **`Hybrid DTW`** | 2.06 | `2020.432528` | `9.159555` | `2021.722811` | [2017.2081, 2023.6569] | `1.2765e-04` | `1.0000e+00` | `1.0000e+00` | **=** |
| 2 | `PSO` | 2.77 | `2026.262947` | `8.461329` | `2024.974868` | [2023.2843, 2029.2416] | `4.4377e-02` | `1.6679e-02` | `1.6679e-02` | **Mejor (+) *** |
| 3 | `ACO` | 2.87 | `2028.819996` | `22.083139` | `2024.535839` | [2021.0461, 2036.5938] | `3.6473e-11` | `3.3502e-03` | `6.7005e-03` | **Mejor (+) **** |
| 4 | `EHO` | 3.48 | `2033.226916` | `22.170281` | `2026.040318` | [2025.4224, 2041.0314] | `3.9823e-09` | `1.6037e-04` | `4.8112e-04` | **Mejor (+) ***** |
| 5 | `GWO` | 3.84 | `2031.485879` | `8.615513` | `2031.613287` | [2028.4530, 2034.5188] | `7.9871e-01` | `3.5700e-05` | `1.4280e-04` | **Mejor (+) ***** |
| 6 | `WOA` | 5.97 | `2129.259508` | `49.039741` | `2120.334590` | [2111.9962, 2146.5228] | `2.5544e-01` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
