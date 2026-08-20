# Reporte de Análisis Estadístico Inferencial y p-valores (HRES2-H2)

- **Número de Corridas Independientes:** 31
- **Algoritmo de Control (Referencia):** `Hybrid DTW`
- **Test de Friedman Chi2:** `108.5480` (p-value = `7.567390e-20`)

## Tabla de Pruebas de Hipótesis y p-valores (Wilcoxon Signed-Rank Test)

| Rank | Algoritmo | Mean Rank | Mean LCOE | Std | Mediana | Shapiro p-val | Wilcoxon p-val | Significancia vs Control |
|---|---|---|---|---|---|---|---|---|
| 1 | **Hybrid DTW** | 1.82 | `0.267160` | `0.000001` | `0.267159` | `4.6380e-12` | `1.0000e+00` | **=** |
| 2 | **GWO** | 3.77 | `0.267764` | `0.002339` | `0.267160` | `2.5942e-11` | `1.6196e-06` | **Peor (-) ***** |
| 3 | **ACO** | 4.21 | `0.271395` | `0.006181` | `0.267159` | `1.3327e-06` | `4.8479e-04` | **Peor (-) ***** |
| 4 | **ILS** | 4.45 | `0.267178` | `0.000018` | `0.267171` | `1.2739e-04` | `9.3132e-10` | **Peor (-) ***** |
| 5 | **PSO** | 4.68 | `0.275970` | `0.009699` | `0.274579` | `6.7961e-06` | `8.0386e-04` | **Peor (-) ***** |
| 6 | **WOA** | 5.35 | `0.277463` | `0.009765` | `0.276524` | `7.7718e-06` | `9.5790e-05` | **Peor (-) ***** |
| 7 | **ABC** | 5.79 | `0.271782` | `0.004379` | `0.270612` | `1.6263e-04` | `1.1673e-06` | **Peor (-) ***** |
| 8 | **EHO** | 7.05 | `0.279582` | `0.007006` | `0.276524` | `7.0969e-05` | `1.8443e-06` | **Peor (-) ***** |
| 9 | **SA** | 7.87 | `0.284300` | `0.011174` | `0.280553` | `5.4110e-03` | `9.3132e-10` | **Peor (-) ***** |


*Leyenda de Significancia:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: No significativo (p >= 0.05)`.
