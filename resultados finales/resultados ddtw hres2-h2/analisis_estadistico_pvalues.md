# Reporte de Análisis Estadístico Inferencial y p-valores (HRES2-H2)

- **Número de Corridas Independientes:** 31
- **Algoritmo de Control (Referencia):** `Hybrid DTW`
- **Test de Friedman Chi2:** `96.7740` (p-value = `1.945469e-17`)

## Tabla de Pruebas de Hipótesis y p-valores (Wilcoxon Signed-Rank Test)

| Rank | Algoritmo | Mean Rank | Mean LCOE | Std | Mediana | Shapiro p-val | Wilcoxon p-val | Significancia vs Control |
|---|---|---|---|---|---|---|---|---|
| 1 | **Hybrid DTW** | 2.03 | `0.267160` | `0.000005` | `0.267159` | `4.9783e-12` | `1.0000e+00` | **=** |
| 2 | **GWO** | 3.97 | `0.268368` | `0.003191` | `0.267160` | `3.4628e-10` | `1.3039e-08` | **Peor (-) ***** |
| 3 | **WOA** | 4.29 | `0.275026` | `0.009688` | `0.267159` | `1.8974e-06` | `6.5451e-03` | **Peor (-) **** |
| 4 | **ILS** | 4.55 | `0.267185` | `0.000022` | `0.267179` | `1.1311e-03` | `9.3132e-10` | **Peor (-) ***** |
| 5 | **PSO** | 5.11 | `0.277749` | `0.009978` | `0.274579` | `4.7329e-05` | `1.3244e-04` | **Peor (-) ***** |
| 6 | **ACO** | 5.18 | `0.275005` | `0.008381` | `0.274579` | `1.1297e-05` | `4.1720e-05` | **Peor (-) ***** |
| 7 | **EHO** | 5.76 | `0.278113` | `0.008044` | `0.276524` | `6.1644e-05` | `1.6121e-05` | **Peor (-) ***** |
| 8 | **ABC** | 5.76 | `0.272004` | `0.003687` | `0.270931` | `8.4202e-04` | `9.3132e-10` | **Peor (-) ***** |
| 9 | **SA** | 8.35 | `0.293362` | `0.012709` | `0.293275` | `4.8378e-03` | `9.3132e-10` | **Peor (-) ***** |


*Leyenda de Significancia:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: No significativo (p >= 0.05)`.
