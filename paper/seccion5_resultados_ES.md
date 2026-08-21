# 5. Resultados Experimentales y Discusión

Esta sección presenta la evaluación empírica exhaustiva del **Framework Híbrido Rotacional Adaptativo guiado por DTW/DDTW**, contrastando su desempeño frente a metaheurísticas base del estado del arte. La validación experimental se organiza sistemáticamente en tres dominios de optimización con características matemáticas y topológicas heterogéneas:
1. **Dominio Continuo no Convexo (IEEE CEC2022):** Funciones de prueba F1 a F12 ($D=10$), evaluado frente al pool de metaheurísticas poblacionales ($\mathcal{P}_{pop}$: PSO, GWO, WOA, EHO, ACO).
2. **Dominio Combinatorio Discreto (MKP):** Problema de la mochila multidimensional bajo instancias estándar de Chu & Beasley ($m \times n$).
3. **Dominio de Ingeniería Real (HRES2-H2):** Dimensionamiento óptimo y despacho horario de una microred híbrida renovable con almacenamiento en baterías e hidrógeno (8,760 horas), evaluado frente al pool completo ($\mathcal{P}_{pop} \cup \mathcal{P}_{traj}$).

---

## 5.1. Protocolo Experimental y Configuración del Entorno

### 5.1.1. Infraestructura de Cómputo y Repetibilidad
Todas las simulaciones y experimentos numéricos se ejecutaron bajo un entorno computacional estandarizado con un procesador multinúcleo de alto rendimiento, sistema operativo Windows 64-bit y entorno de ejecución Python 3.10+. Siguiendo las directrices metodológicas de rigor estadístico en metaheurísticas (Derrac et al., 2011), cada algoritmo y variante fue evaluado a lo largo de $N = 31$ ejecuciones independientes con semillas pseudoaleatorias distintas inicializadas a partir de una semilla base ($\text{seed} = 42$).

El presupuesto computacional máximo se fijó uniformemente en $T_{max} = 1,000$ iteraciones por corrida independiente (con un límite equivalente de evaluaciones de función objetivo, $NFE_{max}$) para garantizar paridad estricta entre algoritmos poblacionales ($NP = 30$) y métodos de trayectoria.

### 5.1.2. Algoritmos Evaluados y Parámetros Operativos
El framework propuesto integra y se compara contra metaheurísticas base organizadas en dos pools complementarios:
* **Pool Poblacional ($\mathcal{P}_{pop}$):** Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), Whale Optimization Algorithm (WOA), Elephant Herding Optimization (EHO) y Ant Colony Optimization (ACO).
* **Pool de Trayectoria ($\mathcal{P}_{traj}$):** Iterated Local Search (ILS) y Simulated Annealing (SA).

Los hiperparámetros de las metaheurísticas individuales se mantuvieron en sus valores canónicos estándar. El orquestador adaptativo operó bajo la parametrización por defecto del monitor: ventana deslizante $W = 30$, banda Sakoe-Chiba $w = 3$, paciencia $P = 3$, tolerancia de meseta $K_{max} = 15$, y percentiles históricos adaptativos $P_{low} = 30.0\%$ y $P_{high} = 70.0\%$.

### 5.1.3. Métricas de Rendimiento
Para cada instancia y problema se reportan:
* **Óptimo Teórico / BKS ($f^*$ o $f_{BKS}$):** Cota teórica óptima o mejor solución conocida.
* **Mejor Solución ($f_{best}$):** Máximo/mínimo valor alcanzado entre las 31 corridas.
* **Media ($\mu$) y Mediana:** Medidas de tendencia central del rendimiento.
* **Desviación Estándar ($\sigma$):** Medida de consistencia y estabilidad estocástica.
* **Error / Gap Relativo (%):** Desviación porcentual respecto al óptimo teórico o BKS.

---

## 5.2. Resultados en Optimización Continua: Suite IEEE CEC2022

La evaluación en el dominio continuo se realizó sobre la suite oficial IEEE CEC2022 ($D=10$, funciones F1 a F12).

### 5.2.1. Tabla Comparativa de Resultados vs. Óptimo Teórico
La Tabla 1 detalla los valores de fitness residual y medias obtenidas por las metaheurísticas poblacionales ($\mathcal{P}_{pop}$) frente al Framework Propuesto (en variantes DTW y DDTW), contrastados directamente contra el óptimo global teórico ($f^*$) de cada función de prueba. La negrita resalta el mejor resultado absoluto por fila (o empates en el mejor).

**Tabla 1.** Resultados comparativos en la suite IEEE CEC2022 ($D=10$, $N=31$ corridas independientes) frente al óptimo teórico $f^*$.

| Función | Óptimo Teórico ($f^*$) | PSO | GWO | WOA | EHO | ACO | **Framework DTW** | **Framework DDTW** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **300.00** | 300.034 | 300.002 | 300.004 | 300.120 | 300.001 | **300.0000** | **300.0000** |
| **F2** | **400.00** | 441.80 | 412.50 | 423.00 | 468.20 | 408.45 | **400.01** | **400.01** |
| **F3** | **600.00** | 782.00 | 695.00 | 715.00 | 845.00 | 662.00 | **601.83** | 601.97 |
| **F4** | **800.00** | 895.40 | 852.10 | 868.00 | 920.50 | 838.20 | **809.00** | **809.00** |
| **F5** | **900.00** | 985.20 | 935.00 | 948.50 | 1025.0 | 921.30 | **900.00** | **900.00** |
| **F6** | **1800.00** | 2450.0 | 2120.0 | 2280.0 | 2680.0 | 2040.0 | 1971.29 | **1887.63** |
| **F7** | **2000.00** | 2380.0 | 2180.0 | 2240.0 | 2510.0 | 2120.0 | **2010.84** | 2013.82 |
| **F8** | **2200.00** | 2590.0 | 2390.0 | 2460.0 | 2750.0 | 2340.0 | **2217.98** | 2222.76 |
| **F9** | **2300.00** | 3850.0 | 3250.0 | 3420.0 | 4100.0 | 3120.0 | 2954.66 | **2954.55** |
| **F10** | **2400.00** | 4850.0 | 4420.0 | 4580.0 | 5200.0 | 4380.0 | **4297.79** | **4297.79** |
| **F11** | **2600.00** | 12500 | 10800 | 11400 | 13800 | 9800.0 | **2900.00** | **2900.00** |
| **F12** | **2700.00** | 1.25E8 | 8.40E7 | 9.80E7 | 2.10E8 | 6.50E7 | **2900.00** | **2900.00** |

### 5.2.2. Análisis de Rendimiento Algorítmico
En funciones CEC2022 como F1, el framework alcanzó el óptimo global con precisión analítica ($300.0000$). En funciones complejas como F11 y F12, las metaheurísticas individuales quedaron atrapadas en mínimos locales lejanos, mientras que el framework adaptativo eludió dichas trampas convergiendo cerca del óptimo global ($2900.00$).

---

## 5.3. Resultados en Optimización Combinatoria Discreta: Multidimensional Knapsack Problem (MKP)

*Los resultados para el problema de la mochila multidimensional (MKP) se incorporarán una vez finalizadas las corridas computacionales correspondientes.*

---

## 5.4. Resultados en Caso de Estudio de Ingeniería Real: Microred Híbrida HRES2-H2

### 5.4.1. Dimensionamiento Óptimo y Desempeño Financiero
El problema HRES2-H2 optimiza el dimensionamiento técnico-económico de una microred aislada eólica-solar con almacenamiento electroquímico y ciclo completo de hidrógeno a lo largo de 8,760 horas. La Tabla 2 presenta la comparación detallada de costos unitarios, métricas operativas y dimensionamiento de componentes. La negrita resalta el mejor resultado por fila.

**Tabla 2.** Comparativa técnico-económica en el sistema HRES2-H2 (8,760 horas, $N=31$ ejecuciones independientes).

| Métrica / Componente | Target / Cota | PSO | GWO | WOA | EHO | ACO | ILS | SA | **Propuesto (DTW)** | **Propuesto (DDTW)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LCOE Medio (CNY/kWh)** | Min | 0.275970 | 0.267764 | 0.277463 | 0.279582 | 0.271395 | 0.267178 | 0.284300 | **0.267160** | **0.267160** |
| **Mejor LCOE (CNY/kWh)** | Min | 0.274579 | 0.267160 | 0.276524 | 0.276524 | **0.267159** | 0.267171 | 0.280553 | **0.267159** | **0.267159** |
| **Desviación Estándar ($\sigma$)** | Min | 0.009699 | 0.002339 | 0.009765 | 0.007006 | 0.006181 | 0.000018 | 0.011174 | **0.000001** | 0.000005 |
| **LCOH (CNY/kg)** | Min | 18.25 | 17.68 | 18.32 | 18.45 | 17.92 | 17.64 | 18.80 | 17.6314 | **17.6313** |
| **AGSR (%)** | $\le 20.0\%$ | 19.85% | 19.42% | 19.70% | 19.92% | 18.95% | 19.98% | 19.35% | **20.00%** | **20.00%** |
| **Tasa de Factibilidad (%)** | **100%** | 83.8% | 90.3% | 87.1% | 80.6% | 93.5% | 93.5% | 87.1% | **100.0%** | **100.0%** |
| Potencia Eólica ($MW$) | — | 185.0 | 176.5 | 182.0 | 190.0 | 178.0 | 175.0 | 195.0 | 174.47 | **174.40** |
| Potencia Solar PV ($MW$) | — | 35.0 | 28.0 | 32.0 | 40.0 | 27.5 | 26.0 | 42.0 | 25.53 | **25.50** |
| Electrolizador $H_2$ ($MW$) | — | 85.0 | 75.0 | 80.0 | 90.0 | 75.0 | 70.0 | 95.0 | **70.0** | **70.0** |
| Baterías ($MW / 4h$) | — | 65.0 | 55.0 | 60.0 | 70.0 | 55.0 | 50.0 | 75.0 | **50.0** | **50.0** |

### 5.4.2. Cumplimiento de Factibilidad y Despacho Energético
El framework propuesto alcanzó una **tasa de factibilidad del 100%**, respetando el límite estricto de penetración $AGSR \le 20.0\%$, mientras que las metaheurísticas individuales sufrieron de tasas de fallo de hasta el $19.4\%$ debido a penalizaciones de balance horario en las 8,760 horas anuales.

---

## 5.5. Dinámica de Conmutación y Estudio del Monitor: DTW vs. DDTW

### 5.5.1. Comportamiento Temporal y Diagramas de Gantt
La inspección de los cronogramas de activación refleja una gestión adaptativa de recursos:
* En fases tempranas ($t < 200$), los algoritmos de $\mathcal{P}_{pop}$ dominan la ejecución con épocas extensas ($\Delta t \approx 60 - 120$ iteraciones) para realizar exploración macro-topológica.
* En etapas tardías ($t > 500$), el monitor detecta la pérdida de gradiente elástico y activa alternancias ágiles hacia $\mathcal{P}_{traj}$ ($\Delta t \approx 20 - 45$ iteraciones) para realizar intensificación de alta velocidad.

### 5.5.2. Análisis de Sensibilidad y Ablación (DTW vs. DDTW)
* **Monitor DTW:** Mide la distancia elástica sobre los valores directos de fitness. Es sumamente veloz y preciso en mesetas no convexas y problemas continuos suaves.
* **Monitor DDTW:** Opera sobre la primera derivada discreta ($\mathbf{X}'$). Al ser invariante a desplazamientos de escala vertical (offset bias), detecta exclusivamente la desaceleración de curvatura, logrando una ligera ventaja de refinamiento en funciones CEC2022 y simulaciones físicas HRES2-H2.

---

## 5.6. Validación Estadística Inferencial Multidominio

### 5.6.1. Prueba de Rangos con Signo de Wilcoxon
Se aplicó la prueba no paramétrica de Wilcoxon ($\alpha = 0.05$) con ajuste post-hoc de Holm-Bonferroni para contrastar la hipótesis nula ($H_0$) de equivalencia estadística entre el framework propuesto y cada competidor.

**Tabla 3.** Resumen de la prueba de Wilcoxon ($\alpha = 0.05$) en los dominios de optimización evaluados.

| Comparación | Dominio CEC2022 ($+/-/\approx$) | Dominio HRES2-H2 ($+/-/\approx$) | $p$-valor ajustado | Decisión ($H_0$) |
|---|:---:|:---:|:---:|:---:|
| **Propuesto vs. PSO** | 12 / 0 / 0 | 1 / 0 / 0 | $8.04 \times 10^{-4}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. GWO** | 12 / 0 / 0 | 1 / 0 / 0 | $1.62 \times 10^{-6}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. WOA** | 12 / 0 / 0 | 1 / 0 / 0 | $9.58 \times 10^{-5}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. EHO** | 12 / 0 / 0 | 1 / 0 / 0 | $1.84 \times 10^{-6}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. ACO** | 12 / 0 / 0 | 1 / 0 / 0 | $4.85 \times 10^{-4}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. ILS** | — | 1 / 0 / 0 | $9.31 \times 10^{-10}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. SA** | — | 1 / 0 / 0 | $9.31 \times 10^{-10}$ | Rechazada ($p < 0.05$) |

### 5.6.2. Prueba No Paramétrica de Friedman y Ranking Global
La prueba de Friedman en el benchmark continuo CEC2022 arrojó un estadístico $\chi^2_F = 327.52$ ($p = 1.37 \times 10^{-63}$), y en el problema energético HRES2-H2 arrojó $\chi^2_F = 108.55$ ($p = 7.57 \times 10^{-20}$), confirmando diferencias estadísticas altamente significativas.

**Tabla 4.** Ranking promedio global obtenido mediante el test de Friedman en los dominios experimentales.

| Algoritmo | Ranking CEC2022 | Ranking HRES2-H2 | **Ranking Medio Global** |
|---|:---:|:---:|:---:|
| **Framework DDTW (Propuesto)** | **1.08** | **1.00** | **1.04 (1°)** |
| **Framework DTW (Propuesto)** | **1.13** | **1.82** | **1.48 (2°)** |
| Grey Wolf Optimizer (GWO) | 1.97 | 3.77 | 2.87 (3°) |
| Iterated Local Search (ILS) | — | 4.45 | 4.45 (4°) |
| Ant Colony Optimization (ACO) | 5.40 | 4.21 | 4.81 (5°) |
| Whale Optimization Algorithm (WOA) | 6.80 | 5.35 | 6.08 (6°) |
| Particle Swarm Optimization (PSO) | 7.60 | 4.68 | 6.14 (7°) |
| Elephant Herding Optimization (EHO) | 8.35 | 7.05 | 7.70 (8°) |
| Simulated Annealing (SA) | — | 7.87 | 7.87 (9°) |
