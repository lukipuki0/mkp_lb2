# Análisis Estadístico Inferencial — CEC2022 — F11_Composition_Function_3

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F11_Composition_Function_3 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `56.2071`  |  p-value = `7.366567e-11`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | **`Hybrid DTW`** | 1.77 | `2651.786891` | `92.724365` | `2600.000000` | [2619.1455, 2684.4283] | `4.0130e-08` | `1.0000e+00` | `1.0000e+00` | **=** |
| 2 | `EHO` | 2.71 | `2756.622449` | `152.057466` | `2750.427166` | [2703.0942, 2810.1507] | `1.6257e-04` | `6.2317e-03` | `6.2317e-03` | **Mejor (+) **** |
| 3 | `ACO` | 3.35 | `2738.939224` | `139.488657` | `2750.427164` | [2689.8355, 2788.0429] | `9.8311e-05` | `2.9006e-03` | `5.8012e-03` | **Mejor (+) **** |
| 4 | `GWO` | 3.94 | `2783.773136` | `155.129897` | `2731.540320` | [2729.1633, 2838.3830] | `2.0708e-03` | `6.2664e-05` | `1.8799e-04` | **Mejor (+) ***** |
| 5 | `WOA` | 4.55 | `2884.927506` | `278.984233` | `2766.785312` | [2786.7177, 2983.1373] | `1.7767e-03` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |
| 6 | `PSO` | 4.68 | `2852.792613` | `154.209406` | `2781.733277` | [2798.5068, 2907.0784] | `4.8741e-04` | `1.3039e-08` | `5.2154e-08` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
