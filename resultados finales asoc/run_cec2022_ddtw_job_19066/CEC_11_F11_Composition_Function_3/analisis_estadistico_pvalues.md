# Análisis Estadístico Inferencial — CEC2022 — F11_Composition_Function_3

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DDTW`
- **Métrica:** Fitness — F11_Composition_Function_3 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `51.0761`  |  p-value = `8.343886e-10`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | **`Hybrid DDTW`** | 1.92 | `2653.690121` | `91.135223` | `2600.000000` | [2621.6081, 2685.7721] | `1.0673e-07` | `1.0000e+00` | `1.0000e+00` | **=** |
| 2 | `EHO` | 2.66 | `2756.622449` | `152.057466` | `2750.427166` | [2703.0942, 2810.1507] | `1.6257e-04` | `2.0671e-02` | `2.0671e-02` | **Mejor (+) *** |
| 3 | `ACO` | 3.35 | `2738.939224` | `139.488657` | `2750.427164` | [2689.8355, 2788.0429] | `9.8311e-05` | `3.1183e-03` | `6.2366e-03` | **Mejor (+) **** |
| 4 | `GWO` | 3.90 | `2783.773136` | `155.129897` | `2731.540320` | [2729.1633, 2838.3830] | `2.0708e-03` | `4.9541e-04` | `1.4862e-03` | **Mejor (+) **** |
| 5 | `WOA` | 4.52 | `2884.927506` | `278.984233` | `2766.785312` | [2786.7177, 2983.1373] | `1.7767e-03` | `4.0047e-08` | `2.0023e-07` | **Mejor (+) ***** |
| 6 | `PSO` | 4.65 | `2852.792613` | `154.209406` | `2781.733277` | [2798.5068, 2907.0784] | `4.8741e-04` | `4.0047e-08` | `2.0023e-07` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
