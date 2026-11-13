# Resumen global descriptivo — CEC2022

> Este archivo no ejecuta pruebas Friedman/Wilcoxon entre funciones. Las funciones CEC2022 tienen escalas y óptimos diferentes, por lo que tratarlas como algoritmos produciría una tabla estadísticamente mal interpretada.

- **Fecha de la corrida:** `20260906_185100`
- **Funciones:** `12`
- **Corridas por función:** `31`
- **Pruebas inferenciales correctas:** en cada carpeta `CEC_XX_*`, contra las MH standalone.

- **Resultado frente al óptimo:** se determina con el mejor run, no con la media de las corridas.

- **Significancia estadística:** resume los cinco contrastes Wilcoxon corregidos por Holm frente a los algoritmos base. El formato es `+favorables/=similares/-desfavorables`.

| Función | Media | Std | Mediana | Mejor | Peor | Óptimo | Gap media (%) | Gap mejor (%) | Mejor run | Resultado frente al óptimo | Significancia estadística |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `F1_Shifted_Rotated_Zakharov` | 300.468821 | 2.609459 | 300.000000 | 300.000000 | 314.529000 | 300.000000 | 0.156 | 0.000 | 8 | Óptimo alcanzado | Mixta significativa (+3/=0/-2) |
| `F2_Shifted_Rotated_Rosenbrock` | 405.935823 | 3.761034 | 408.916102 | 400.010867 | 408.916118 | 400.000000 | 1.484 | 0.003 | 4 | Peor que el óptimo | Favorable significativa (+3/=2/-0) |
| `F3_Shifted_Rotated_Expanded_Schaffers_F7` | 603.687758 | 6.050480 | 601.080270 | 600.017585 | 627.667954 | 600.000000 | 0.615 | 0.003 | 16 | Peor que el óptimo | Mixta significativa (+2/=2/-1) |
| `F4_Shifted_Rotated_NonContinuous_Rastrigin` | 815.741935 | 4.946162 | 816.000000 | 806.000000 | 827.000000 | 800.000000 | 1.968 | 0.750 | 30 | Peor que el óptimo | Favorable significativa (+3/=2/-0) |
| `F5_Shifted_Rotated_Levy` | 901.972045 | 5.980934 | 900.000037 | 900.000000 | 924.744911 | 900.000000 | 0.219 | 0.000 | 3 | Óptimo alcanzado | Mixta significativa (+3/=1/-1) |
| `F6_Hybrid_Function_1` | 3532.529734 | 1942.164049 | 2738.486063 | 1830.399635 | 8017.950244 | 1800.000000 | 96.252 | 1.689 | 2 | Peor que el óptimo | Favorable significativa (+1/=4/-0) |
| `F7_Hybrid_Function_2` | 2022.698363 | 9.806440 | 2021.989966 | 2001.495208 | 2045.873099 | 2000.000000 | 1.135 | 0.075 | 8 | Peor que el óptimo | Favorable significativa (+3/=2/-0) |
| `F8_Hybrid_Function_3` | 2221.176794 | 3.347798 | 2221.094770 | 2207.624893 | 2231.043273 | 2200.000000 | 0.963 | 0.347 | 7 | Peor que el óptimo | Favorable significativa (+4/=1/-0) |
| `F9_Composition_Function_1` | 2529.284383 | 0.000000 | 2529.284383 | 2529.284383 | 2529.284383 | 2300.000000 | 9.969 | 9.969 | 1 | Peor que el óptimo | Favorable significativa (+4/=1/-0) |
| `F10_Composition_Function_2` | 2522.829709 | 46.354958 | 2500.528814 | 2500.281271 | 2625.648477 | 2400.000000 | 5.118 | 4.178 | 29 | Peor que el óptimo | Favorable significativa (+4/=1/-0) |
| `F11_Composition_Function_3` | 2653.690121 | 91.135223 | 2600.000000 | 2600.000000 | 2900.000000 | 2600.000000 | 2.065 | 0.000 | 13 | Óptimo alcanzado | Favorable significativa (+5/=0/-0) |
| `F12_Composition_Function_4` | 2863.757163 | 1.357503 | 2863.714850 | 2860.680176 | 2866.682168 | 2700.000000 | 6.065 | 5.951 | 1 | Peor que el óptimo | Favorable significativa (+2/=3/-0) |

Para comparar DTW/DDTW contra PSO, GWO, WOA, EHO y ACO, usar los archivos `CEC_XX_*/analisis_estadistico_pvalues.md`.
