# Resumen global descriptivo — CEC2022

> Este archivo no ejecuta pruebas Friedman/Wilcoxon entre funciones. Las funciones CEC2022 tienen escalas y óptimos diferentes, por lo que tratarlas como algoritmos produciría una tabla estadísticamente mal interpretada.

- **Fecha de la corrida:** `20260906_210948`
- **Funciones:** `12`
- **Corridas por función:** `31`
- **Pruebas inferenciales correctas:** en cada carpeta `CEC_XX_*`, contra las MH standalone.

- **Resultado frente al óptimo:** se determina con el mejor run, no con la media de las corridas.

- **Significancia estadística:** resume los cinco contrastes Wilcoxon corregidos por Holm frente a los algoritmos base. El formato es `+favorables/=similares/-desfavorables`.

| Función | Media | Std | Mediana | Mejor | Peor | Óptimo | Gap media (%) | Gap mejor (%) | Mejor run | Resultado frente al óptimo | Significancia estadística |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `F1_Shifted_Rotated_Zakharov` | 300.451283 | 2.512340 | 300.000000 | 300.000000 | 313.988173 | 300.000000 | 0.150 | 0.000 | 9 | Óptimo alcanzado | Mixta significativa (+3/=0/-2) |
| `F2_Shifted_Rotated_Rosenbrock` | 405.995119 | 3.766298 | 408.916102 | 400.010867 | 408.916107 | 400.000000 | 1.499 | 0.003 | 4 | Peor que el óptimo | Favorable significativa (+3/=2/-0) |
| `F3_Shifted_Rotated_Expanded_Schaffers_F7` | 603.534403 | 5.945872 | 601.416843 | 600.010816 | 627.667954 | 600.000000 | 0.589 | 0.002 | 14 | Peor que el óptimo | Mixta significativa (+2/=2/-1) |
| `F4_Shifted_Rotated_NonContinuous_Rastrigin` | 816.257942 | 4.923583 | 815.000000 | 808.000000 | 827.000000 | 800.000000 | 2.032 | 1.000 | 8 | Peor que el óptimo | Favorable significativa (+2/=3/-0) |
| `F5_Shifted_Rotated_Levy` | 901.672452 | 5.595086 | 900.000069 | 900.000000 | 924.744911 | 900.000000 | 0.186 | 0.000 | 11 | Óptimo alcanzado | Mixta significativa (+3/=0/-2) |
| `F6_Hybrid_Function_1` | 3488.596966 | 1821.985314 | 2718.090658 | 1846.429085 | 7923.928513 | 1800.000000 | 93.811 | 2.579 | 2 | Peor que el óptimo | Favorable significativa (+1/=4/-0) |
| `F7_Hybrid_Function_2` | 2020.432528 | 9.159555 | 2021.722811 | 2000.500988 | 2043.420739 | 2000.000000 | 1.022 | 0.025 | 12 | Peor que el óptimo | Favorable significativa (+5/=0/-0) |
| `F8_Hybrid_Function_3` | 2220.337218 | 4.644298 | 2220.940519 | 2201.213746 | 2225.463756 | 2200.000000 | 0.924 | 0.055 | 26 | Peor que el óptimo | Favorable significativa (+5/=0/-0) |
| `F9_Composition_Function_1` | 2529.284383 | 0.000000 | 2529.284383 | 2529.284383 | 2529.284383 | 2300.000000 | 9.969 | 9.969 | 1 | Peor que el óptimo | Mixta significativa (+3/=1/-1) |
| `F10_Composition_Function_2` | 2522.945165 | 46.728406 | 2500.513584 | 2500.258225 | 2632.527622 | 2400.000000 | 5.123 | 4.177 | 13 | Peor que el óptimo | Favorable significativa (+4/=1/-0) |
| `F11_Composition_Function_3` | 2651.786891 | 92.724365 | 2600.000000 | 2600.000000 | 3000.000000 | 2600.000000 | 1.992 | 0.000 | 13 | Óptimo alcanzado | Favorable significativa (+5/=0/-0) |
| `F12_Composition_Function_4` | 2863.929552 | 1.322803 | 2863.877931 | 2860.680176 | 2866.682300 | 2700.000000 | 6.071 | 5.951 | 1 | Peor que el óptimo | Favorable significativa (+2/=3/-0) |

Para comparar DTW/DDTW contra PSO, GWO, WOA, EHO y ACO, usar los archivos `CEC_XX_*/analisis_estadistico_pvalues.md`.
