# Análisis Estadístico Inferencial — CEC2022 — F8_Hybrid_Function_3

- **Runs independientes:** 31
- **Referencia (control):** `Hybrid DTW`
- **Métrica:** Fitness — F8_Hybrid_Function_3 (Minimización)
- **Desviación estándar:** muestral (`ddof=1`)
- **IC 95%:** aproximación normal `media ± 1.96·SE`
- **Corrección por comparaciones múltiples:** Holm sobre 5 comparaciones pareadas contra la referencia.
- **Friedman χ²:** `78.4470`  |  p-value = `1.772690e-15`  ✅ Diferencia significativa

## Tabla de Resultados y p-valores

| Rank | Algoritmo | Mean Rank | Media | Std | Mediana | IC 95% | Shapiro p | Wilcoxon p bruto | Wilcoxon p Holm | Significancia |
|------|-----------|-----------|-------|-----|---------|--------|-----------|------------------|-----------------|---------------|
| 1 | **`Hybrid DTW`** | 1.81 | `2220.337218` | `4.644298` | `2220.940519` | [2218.7023, 2221.9721] | `1.3255e-08` | `1.0000e+00` | `1.0000e+00` | **=** |
| 2 | `EHO` | 2.55 | `2222.127509` | `1.987983` | `2221.828514` | [2221.4277, 2222.8273] | `2.7251e-02` | `2.8927e-02` | `2.8927e-02` | **Mejor (+) *** |
| 3 | `PSO` | 3.29 | `2261.909459` | `59.090445` | `2221.351960` | [2241.1081, 2282.7109] | `3.7583e-07` | `6.6399e-03` | `1.3280e-02` | **Mejor (+) *** |
| 4 | `GWO` | 3.48 | `2223.959390` | `7.253282` | `2226.409525` | [2221.4060, 2226.5127] | `4.5246e-05` | `3.3502e-03` | `1.0051e-02` | **Mejor (+) *** |
| 5 | `ACO` | 4.26 | `2227.499838` | `1.898362` | `2227.355414` | [2226.8316, 2228.1681] | `2.6306e-01` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |
| 6 | `WOA` | 5.61 | `2253.110486` | `43.433550` | `2238.699141` | [2237.8207, 2268.4002] | `1.1992e-08` | `9.3132e-10` | `4.6566e-09` | **Mejor (+) ***** |


*Leyenda:* `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p ≥ 0.05`.
