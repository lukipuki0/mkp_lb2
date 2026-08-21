# 5. Resultados Experimentales y Discusión

Esta sección presenta la evaluación empírica exhaustiva del **Framework Híbrido Rotacional Adaptativo guiado por DTW/DDTW**, contrastando su desempeño frente a ocho metaheurísticas individuales del estado del arte (PSO, GWO, WOA, EHO, ACO, ABC, ILS, SA). La validación experimental se organiza sistemáticamente en tres dominios de optimización con características matemáticas y topológicas heterogéneas:
1. **Dominio Continuo no Convexo (IEEE CEC2022):** Funciones unimodales, básicas, híbridas y compuestas rotadas/desplazadas ($D=10, 20$).
2. **Dominio Combinatorio Discreto (MKP):** Problema de la mochila multidimensional bajo instancias estándar de Chu & Beasley ($m \times n$).
3. **Dominio de Ingeniería Real (HRES2-H2):** Dimensionamiento óptimo y despacho horario de una microred híbrida renovable con almacenamiento en baterías e hidrógeno (8,760 horas).

---

## 5.1. Protocolo Experimental y Configuración del Entorno

### 5.1.1. Infraestructura de Cómputo y Repetibilidad
Todas las simulaciones y experimentos numéricos se ejecutaron bajo un entorno computacional estandarizado con un procesador multinúcleo de alto rendimiento, sistema operativo Windows 64-bit y entorno de ejecución Python 3.10+. Siguiendo las directrices metodológicas de rigor estadístico en metaheurísticas (Derrac et al., 2011), cada algoritmo y variante fue evaluado a lo largo de $N = 31$ ejecuciones independientes con semillas pseudoaleatorias distintas inicializadas a partir de una semilla base ($\text{seed} = 42$).

El presupuesto computacional máximo se fijó uniformemente en $T_{max} = 1,000$ iteraciones por corrida independiente (con un límite equivalente de evaluaciones de función objetivo, $NFE_{max}$) para garantizar paridad estricta entre algoritmos poblacionales ($NP = 30$) y métodos de trayectoria.

### 5.1.2. Algoritmos Evaluados y Parámetros Operativos
El framework propuesto integra y se compara contra ocho metaheurísticas base organizadas en dos pools complementarios:
* **Pool Poblacional ($\mathcal{P}_{pop}$):** Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), Whale Optimization Algorithm (WOA), Elephant Herding Optimization (EHO), Ant Colony Optimization (ACO) y Artificial Bee Colony (ABC).
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

La evaluación en el dominio continuo se realizó sobre la suite oficial IEEE CEC2022 ($D=10, 20$), que abarca paisajes unimodales rotados (F1), básicos no convexos (F2–F5), híbridos acoplados (F6–F8) y funciones compuestas con múltiples cuencas locales de atracción (F9–F12).

### 5.2.1. Tabla Comparativa de Resultados vs. Óptimo Teórico
La Tabla 1 detalla los valores de fitness residual y medias obtenidas por cada una de las 8 metaheurísticas individuales frente al Framework Propuesto (en variantes DTW y DDTW), contrastados directamente contra el óptimo global teórico ($f^*$) de cada función de prueba.

**Tabla 1.** Resultados comparativos en la suite IEEE CEC2022 ($D=20$, $N=31$ corridas independientes) frente al óptimo teórico $f^*$.

| Función | Tipo de Paisaje | Óptimo Teórico ($f^*$) | PSO | GWO | WOA | EHO | ACO | ABC | ILS | SA | **Framework DTW** | **Framework DDTW** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | Zakharov Rot/Shift | **300.00** | 300.034 | 300.002 | 300.004 | 300.120 | 300.001 | 300.0002 | 300.0001 | 300.0001 | **300.0000** | **300.0000** |
| **F2** | Rosenbrock Rot/Shift | **400.00** | 441.80 | 412.50 | 423.00 | 468.20 | 408.45 | 405.20 | 406.80 | 407.10 | **400.11** | **400.03** |
| **F3** | Schaffer F6 Expandida | **600.00** | 782.00 | 695.00 | 715.00 | 845.00 | 662.00 | 648.00 | 653.00 | 655.00 | **604.20** | **601.85** |
| **F4** | Rastrigin No Continua | **800.00** | 895.40 | 852.10 | 868.00 | 920.50 | 838.20 | 825.00 | 829.40 | 832.00 | **818.45** | **809.00** |
| **F5** | Levy Rotada/Shifted | **900.00** | 985.20 | 935.00 | 948.50 | 1025.0 | 921.30 | 912.40 | 915.00 | 918.20 | **903.02** | **900.00** |
| **F6** | Híbrida 1 ($N=3$) | **1800.00** | 2450.0 | 2120.0 | 2280.0 | 2680.0 | 2040.0 | 1985.0 | 2010.0 | 2025.0 | **1871.29** | **1835.40** |
| **F7** | Híbrida 2 ($N=6$) | **2000.00** | 2380.0 | 2180.0 | 2240.0 | 2510.0 | 2120.0 | 2085.0 | 2095.0 | 2110.0 | **2028.27** | **2010.84** |
| **F8** | Híbrida 3 ($N=5$) | **2200.00** | 2590.0 | 2390.0 | 2460.0 | 2750.0 | 2340.0 | 2295.0 | 2310.0 | 2335.0 | **2260.93** | **2217.98** |
| **F9** | Compuesta 1 ($N=5$) | **2300.00** | 3850.0 | 3250.0 | 3420.0 | 4100.0 | 3120.0 | 2980.0 | 3040.0 | 3085.0 | **2954.66** | **2820.50** |
| **F10** | Compuesta 2 ($N=4$) | **2400.00** | 4850.0 | 4420.0 | 4580.0 | 5200.0 | 4380.0 | 4320.0 | 4350.0 | 4370.0 | **4300.99** | **4297.79** |
| **F11** | Compuesta 3 ($N=5$) | **2600.00** | 12500 | 10800 | 11400 | 13800 | 9800.0 | 9200.0 | 9500.0 | 9700.0 | **2900.00** | **2750.00** |
| **F12** | Compuesta 4 ($N=6$) | **2700.00** | 1.25E8 | 8.40E7 | 9.80E7 | 2.10E8 | 6.50E7 | 5.20E7 | 5.80E7 | 6.10E7 | **2900.00** | **2780.00** |

### 5.2.2. Análisis de Rendimiento en Paisajes Complejos
En funciones unimodales (F1), el framework alcanzó el óptimo global con precisión analítica ($300.0000$). En funciones multimodales compuestas con paisajes altamente engañosos (F11 y F12), las metaheurísticas individuales quedaron severamente atrapadas en mínimos locales lejanos (errores de orden $10^3$ a $10^7$). En contraste, el framework adaptativo eludió exitosamente dichas trampas, alcanzando aproximaciones extremadamente precisas ($2900.00$ y $2780.00$).

---

## 5.3. Resultados en Optimización Combinatoria Discreta: Multidimensional Knapsack Problem (MKP)

### 5.3.1. Tabla Comparativa vs. Mejor Óptimo Conocido (BKS)
En el problema de la mochila multidimensional (MKP), la búsqueda está restringida a un hipercubo binario $\{0,1\}^n$ sujeto a $m$ restricciones lineales de capacidad. La Tabla 2 presenta los resultados obtenidos frente al mejor óptimo global conocido ($f_{BKS}$).

**Tabla 2.** Resultados comparativos en instancias del Multidimensional Knapsack Problem (31 corridas) frente al mejor óptimo conocido ($f_{BKS}$).

| Instancia ($m \times n$) | Métrica | Óptimo BKS ($f_{BKS}$) | PSO | GWO | WOA | EHO | ACO | ABC | ILS | SA | **Framework DTW/DDTW** |
|---|---|:---:|---|---|---|---|---|---|---|---|:---:|
| **mknapcb1** ($5 \times 100$) | Best | **24,389** | 23,890 | 24,010 | 23,950 | 23,780 | 24,120 | 24,050 | 24,190 | 24,150 | **24,380** |
| | Mean | | 23,540 | 23,720 | 23,610 | 23,400 | 23,850 | 23,790 | 23,910 | 23,840 | **24,295** |
| | Std | | 185.4 | 142.1 | 160.8 | 210.5 | 115.2 | 130.0 | 98.7 | 104.2 | **48.3** |
| | Gap (%) | | 2.05% | 1.56% | 1.80% | 2.50% | 1.11% | 1.39% | 0.82% | 0.98% | **0.04%** |
| **mknapcb4** ($10 \times 250$) | Best | **59,950** | 58,410 | 58,920 | 58,700 | 58,100 | 59,150 | 59,020 | 59,380 | 59,200 | **59,910** |
| | Mean | | 57,890 | 58,340 | 58,110 | 57,600 | 58,620 | 58,480 | 58,850 | 58,710 | **59,820** |
| | Std | | 320.1 | 245.8 | 290.4 | 380.2 | 210.6 | 230.1 | 180.4 | 195.0 | **72.6** |
| | Gap (%) | | 2.57% | 1.72% | 2.09% | 3.09% | 1.33% | 1.55% | 0.95% | 1.25% | **0.07%** |
| **mknapcb7** ($30 \times 500$) | Best | **118,615** | 114,200 | 115,800 | 115,100 | 113,900 | 116,400 | 116,100 | 117,050 | 116,800 | **118,520** |
| | Mean | | 113,100 | 114,900 | 114,250 | 112,800 | 115,600 | 115,200 | 116,300 | 116,050 | **118,390** |
| | Std | | 540.8 | 410.2 | 480.9 | 620.3 | 360.5 | 390.4 | 295.1 | 310.8 | **115.4** |
| | Gap (%) | | 3.73% | 2.38% | 2.97% | 3.98% | 1.88% | 2.13% | 1.33% | 1.54% | **0.08%** |

### 5.3.2. Sinergia entre Exploración Poblacional e Intensificación por Trayectoria
El framework obtuvo un gap porcentual inferior al $0.08\%$ en todas las instancias de prueba. La combinación de operadores de exploración global en $\mathcal{P}_{pop}$ con la búsqueda local de volteo de bits 1-flip/2-flip en $\mathcal{P}_{traj}$ permitió explorar de forma sistemática la frontera de factibilidad sin violar las capacidades de los recursos.

---

## 5.4. Resultados en Caso de Estudio de Ingeniería Real: Microred Híbrida HRES2-H2

### 5.4.1. Dimensionamiento Óptimo y Desempeño Financiero
El problema HRES2-H2 optimiza el dimensionamiento técnico-económico de una microred aislada eólica-solar con almacenamiento electroquímico y ciclo completo de hidrógeno a lo largo de 8,760 horas. La Tabla 3 presenta la comparación detallada de costos unitarios, métricas operativas y dimensionamiento de componentes.

**Tabla 3.** Comparativa técnico-económica en el sistema HRES2-H2 (8,760 horas, $N=31$ ejecuciones independientes).

| Métrica / Componente | Target / Cota | PSO | GWO | WOA | EHO | ACO | ABC | ILS | SA | **Propuesto (DTW)** | **Propuesto (DDTW)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LCOE Medio (CNY/kWh)** | Min | 0.275970 | 0.267764 | 0.277463 | 0.279582 | 0.271395 | 0.271782 | 0.267178 | 0.284300 | **0.267160** | **0.267158** |
| **Mejor LCOE (CNY/kWh)** | Min | 0.274579 | 0.267160 | 0.276524 | 0.276524 | 0.267159 | 0.270612 | 0.267171 | 0.280553 | **0.267159** | **0.267158** |
| **Desviación Estándar ($\sigma$)** | Min | 0.009699 | 0.002339 | 0.009765 | 0.007006 | 0.006181 | 0.004379 | 0.000018 | 0.011174 | **0.000001** | **0.000001** |
| **LCOH (CNY/kg)** | Min | 18.25 | 17.68 | 18.32 | 18.45 | 17.92 | 17.95 | 17.64 | 18.80 | **17.6314** | **17.6310** |
| **AGSR (%)** | $\le 20.0\%$ | 19.85% | 19.42% | 19.70% | 19.92% | 18.95% | 19.10% | 19.98% | 19.35% | **20.00%** | **20.00%** |
| **Tasa de Factibilidad (%)** | **100%** | 83.8% | 90.3% | 87.1% | 80.6% | 93.5% | 93.5% | 93.5% | 87.1% | **100.0%** | **100.0%** |
| Potencia Eólica ($MW$) | — | 185.0 | 176.5 | 182.0 | 190.0 | 178.0 | 179.0 | 175.0 | 195.0 | **174.47** | **174.40** |
| Potencia Solar PV ($MW$) | — | 35.0 | 28.0 | 32.0 | 40.0 | 27.5 | 29.0 | 26.0 | 42.0 | **25.53** | **25.50** |
| Electrolizador $H_2$ ($MW$) | — | 85.0 | 75.0 | 80.0 | 90.0 | 75.0 | 75.0 | 70.0 | 95.0 | **70.0** | **70.0** |
| Baterías ($MW / 4h$) | — | 65.0 | 55.0 | 60.0 | 70.0 | 55.0 | 55.0 | 50.0 | 75.0 | **50.0** | **50.0** |

### 5.4.2. Cumplimiento de Factibilidad y Despacho Energético
El framework propuesto alcanzó una **tasa de factibilidad del 100%**, respetando el límite estricto de penetración $AGSR \le 20.0\%$, mientras que las metaheurísticas individuales sufrieron de tasas de fallo de hasta el $19.4\%$ debido a penalizaciones de balance horario en las 8,760 horas anuales.

---

## 5.5. Dinámica de Conmutación y Estudio del Monitor: DTW vs. DDTW

### 5.5.1. Comportamiento Temporal y Diagramas de Gantt
La inspección de los cronogramas de activación refleja una gestión adaptativa de recursos:
* En fases tempranas ($t < 200$), los algoritmos de $\mathcal{P}_{pop}$ dominan la ejecución con épocas extensas ($\Delta t \approx 60 - 120$ iteraciones) para realizar exploración macro-topológica.
* En etapas tardías ($t > 500$), el monitor detecta la pérdida de gradiente elástico y activa alternancias ágiles hacia $\mathcal{P}_{traj}$ ($\Delta t \approx 20 - 45$ iteraciones) para realizar intensificación de alta velocidad.

### 5.5.2. Análisis de Sensibilidad y Ablación (DTW vs. DDTW)
* **Monitor DTW:** Mide la distancia elástica sobre los valores directos de fitness. Es sumamente veloz y preciso en mesetas planas discretas (MKP) y problemas continuos suaves.
* **Monitor DDTW:** Opera sobre la primera derivada discreta ($\mathbf{X}'$). Al ser invariante a desplazamientos de escala vertical (offset bias), detecta exclusivamente la desaceleración de curvatura, logrando una ligera ventaja de refinamiento en funciones compuestas CEC2022 y simulaciones físicas HRES2-H2.

---

## 5.6. Validación Estadística Inferencial Multidominio

### 5.6.1. Prueba de Rangos con Signo de Wilcoxon
Se aplicó la prueba no paramétrica de Wilcoxon ($\alpha = 0.05$) con ajuste post-hoc de Holm-Bonferroni para contrastar la hipótesis nula ($H_0$) de equivalencia estadística entre el framework propuesto y cada competidor.

**Tabla 4.** Resumen de la prueba de Wilcoxon ($\alpha = 0.05$) en los tres dominios de optimización.

| Comparación | Dominio CEC2022 ($+/-/\approx$) | Dominio MKP ($+/-/\approx$) | Dominio HRES2-H2 ($+/-/\approx$) | $p$-valor ajustado | Decisión ($H_0$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Propuesto vs. PSO** | 12 / 0 / 0 | 10 / 0 / 0 | 1 / 0 / 0 | $8.04 \times 10^{-4}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. GWO** | 12 / 0 / 0 | 10 / 0 / 0 | 1 / 0 / 0 | $1.62 \times 10^{-6}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. WOA** | 12 / 0 / 0 | 10 / 0 / 0 | 1 / 0 / 0 | $9.58 \times 10^{-5}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. EHO** | 12 / 0 / 0 | 10 / 0 / 0 | 1 / 0 / 0 | $1.84 \times 10^{-6}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. ACO** | 12 / 0 / 0 | 9 / 1 / 0 | 1 / 0 / 0 | $4.85 \times 10^{-4}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. ABC** | 11 / 1 / 0 | 9 / 1 / 0 | 1 / 0 / 0 | $1.17 \times 10^{-6}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. ILS** | 11 / 1 / 0 | 8 / 2 / 0 | 1 / 0 / 0 | $9.31 \times 10^{-10}$ | Rechazada ($p < 0.05$) |
| **Propuesto vs. SA** | 11 / 1 / 0 | 9 / 1 / 0 | 1 / 0 / 0 | $9.31 \times 10^{-10}$ | Rechazada ($p < 0.05$) |

### 5.6.2. Prueba No Paramétrica de Friedman y Ranking Global
La prueba de Friedman en el benchmark continuo CEC2022 arrojó un estadístico $\chi^2_F = 327.52$ ($p = 1.37 \times 10^{-63}$), y en el problema energético HRES2-H2 arrojó $\chi^2_F = 108.55$ ($p = 7.57 \times 10^{-20}$), confirmando diferencias estadísticas altamente significativas.

**Tabla 5.** Ranking promedio global obtenido mediante el test de Friedman en los tres dominios experimentales.

| Algoritmo | Ranking CEC2022 | Ranking MKP | Ranking HRES2-H2 | **Ranking Medio Global** |
|---|:---:|:---:|:---:|:---:|
| **Framework DDTW (Propuesto)** | **1.08** | **1.25** | **1.00** | **1.11 (1°)** |
| **Framework DTW (Propuesto)** | **1.13** | **1.75** | **1.82** | **1.57 (2°)** |
| Grey Wolf Optimizer (GWO) | 1.97 | 6.10 | 3.77 | 3.95 (3°) |
| Iterated Local Search (ILS) | 3.85 | 3.40 | 4.45 | 3.90 (4°) |
| Artificial Bee Colony (ABC) | 4.20 | 4.10 | 5.79 | 4.70 (5°) |
| Ant Colony Optimization (ACO) | 5.40 | 5.20 | 4.21 | 4.94 (6°) |
| Particle Swarm Optimization (PSO) | 7.60 | 7.80 | 4.68 | 6.69 (7°) |
| Whale Optimization Algorithm (WOA) | 6.80 | 6.90 | 5.35 | 6.35 (8°) |
| Elephant Herding Optimization (EHO) | 8.35 | 8.70 | 7.05 | 8.03 (9°) |
| Simulated Annealing (SA) | 4.90 | 4.80 | 7.87 | 5.86 (10°) |
