# Estructura Metodológica Detallada para MDPI: "Materials and Methods"

> **Título del Artículo:** *A Dynamic Time Warping-Driven Adaptive Rotational Hybrid Metaheuristic Framework for Discrete, Continuous, and Energy System Optimization*  
> **Revista Target (MDPI):** *Energies*, *Algorithms*, *Applied Sciences*, o *Mathematics*  
> **Norma MDPI:** La sección *Materials and Methods* debe incluir detalles suficientes para la reproducibilidad total del estudio, presentando las ecuaciones matemáticas, pseudocódigos, hiperparámetros y la caracterización de los dominios de prueba de forma rigurosa.

---

## 📌 Esquema General de Subsecciones (MDPI Format)

```
2. Materials and Methods
   ├── 2.1. Overall Architecture of the DTW-Driven Rotational Hybrid Framework
   ├── 2.2. Mathematical Formulation of the DTW/DDTW Stagnation Detection Engine
   ├── 2.3. Adaptive Thresholding Strategy via Moving Percentiles
   ├── 2.4. Metaheuristic Pools and Solution Seed Memory Injection Protocol
   ├── 2.5. Problem Formulation 1: Discrete Multidimensional Knapsack Problem (MKP)
   ├── 2.6. Problem Formulation 2: Continuous IEEE CEC2022 Benchmark Suite
   ├── 2.7. Real-World Case Study: Hybrid Renewable Energy System with Hydrogen (HRES2-H2)
   │      ├── 2.7.1. Renewable Generation Models (WT and PV)
   │      ├── 2.7.2. Hydrogen Storage & Energy Storage System Dispatch Logic
   │      └── 2.7.3. Economic Objective Functions (LCOE, LCOH) & Grid Security Constraint (AGSR)
   └── 2.8. Non-Parametric Statistical Inferential Protocol
```

---

## 📝 Desglose Párrafo a Párrafo con Ecuaciones y Contenidos Clave

### 2.1. Overall Architecture of the DTW-Driven Rotational Hybrid Framework

* **Párrafo 2.1.1 — Filosofía del Framework y Solución al No Free Lunch (NFL):**
  * *Contenido:* Presenta el flujo conceptual del framework híbrido rotacional. Explica cómo la alternancia dinámica entre metaheurísticas poblacionales (orientadas a la exploración amplia del espacio de búsqueda) y metaheurísticas de trayectoria (orientadas a la explotación local intensiva) permite contrarrestar la degradación de desempeño predicha por el teorema No Free Lunch.
  * *Elemento Visual:* **Figura 1: Diagrama de flujo conceptual del orquestador híbrido**, mostrando la interacción entre los Pools, el Monitor DTW/DDTW y el módulo de inyección de memoria.

* **Párrafo 2.1.2 — Estructura de Control por Épocas (Epochs):**
  * *Contenido:* Define formalmente el concepto de *Epoch* $e$ como la ventana operacional en la que un algoritmo individual $M \in \mathcal{P}$ ejecuta la búsqueda. Explicar cómo el término del epoch no ocurre por un número fijo de iteraciones arbitrarias, sino por una señal de alarma gatillada por el detector de estancamiento temporal.

---

### 2.2. Mathematical Formulation of the DTW/DDTW Stagnation Detection Engine

* **Párrafo 2.2.1 — Construcción del Historial de Fitness y Baselines:**
  * *Contenido:* Define la serie temporal de fitness reciente $X = [x_1, x_2, \dots, x_W]$ sobre una ventana deslizante de tamaño $W$. Define la rampa baseline ideal $R$ (crecimiento sostenido con pendiente mínima $s_{min}$) y la baseline constante $C$ (meseta estancada):
    $$\begin{aligned}
    r_k &= x_1 + s_{min} \cdot (k - 1), \quad k = 1, \dots, W \\
    c_k &= x_1, \quad k = 1, \dots, W
    \end{aligned}$$

* **Párrafo 2.2.2 — Cálculo de la Distancia DTW con Banda Sakoe-Chiba:**
  * *Contenido:* Presenta la ecuación de alineamiento elástico de series temporales mediante la matriz de acumulación de distancias $D(i, j)$ bajo la restricción de ventana Sakoe-Chiba $w$:
    $$D(i, j) = |s_i - t_j| + \min \left\{ D(i-1, j), D(i, j-1), D(i-1, j-1) \right\}, \quad |i - j| \le w$$
  * *Ecuación Derivative DTW (DDTW):* Explica la versión sobre las primeras derivadas para capturar cambios de pendiente independientes de la magnitud:
    $$f'(x_k) = \frac{(x_k - x_{k-1}) + \frac{x_{k+1} - x_{k-1}}{2}}{2}$$
    $$\text{DDTW}(S, T) = \text{DTW}(\nabla S, \nabla T)$$

* **Párrafo 2.2.3 — Métrica de Desviación Relativa ($\Delta$):**
  * *Contenido:* Formular el cálculo del diferencial entre la distancia a la rampa de progreso ($D_1$) y la distancia a la meseta constante ($D_2$):
    $$D_1 = \text{DTW}(X, R), \quad D_2 = \text{DTW}(X, C), \quad \Delta = D_1 - D_2$$
  * *Implicación:* Si $D_2 \to 0$ y $D_1$ crece, la curva de fitness ha perdido pendiente y se encuentra en estancamiento.

---

### 2.3. Adaptive Thresholding Strategy via Moving Percentiles

* **Párrafo 2.3.1 — Límites Adaptativos $P_{low}$ y $P_{high}$:**
  * *Contenido:* Explica por qué los umbrales fijos fallan en problemas heterogéneos y presenta la formulación adaptativa basada en percentiles móviles de los historiales de distancia $\mathcal{H}_{D1}, \mathcal{H}_{D2}, \mathcal{H}_{\Delta}$:
    $$\theta_c = \text{Percentil}(\mathcal{H}_{D2}, P_{low}), \quad \theta_r = \text{Percentil}(\mathcal{H}_{D1}, P_{high}), \quad \theta_\Delta = \text{Percentil}(\mathcal{H}_{\Delta}, P_{high})$$

* **Párrafo 2.3.2 — Criterio Multicondicional de Disparo y Paciencia:**
  * *Contenido:* Define la condición triple de estancamiento en la iteración $t$ y la racha de confirmación $S_t$ con parámetro de paciencia $P$:
    $$\text{Stagnant}(t) = \mathbb{I}\left( (N_{no\_improve} \ge K_{max}) \land (D_2 \le \theta_c) \land (D_1 \ge \theta_r \lor \Delta \ge \theta_\Delta) \right)$$
    $$\text{Trigger Alarm} \iff S_t \ge P$$

---

### 2.4. Metaheuristic Pools and Solution Seed Memory Injection Protocol

* **Párrafo 2.4.1 — Caracterización de los Pools Algoritmos ($\mathcal{P}_{pop}$ vs $\mathcal{P}_{traj}$):**
  * *Contenido:* Detalla los integrantes de cada pool:
    * **Pool Poblacional (Exploración):** Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), Whale Optimization Algorithm (WOA), Elephant Herding Optimization (EHO), Ant Colony Optimization (ACO), Artificial Bee Colony (ABC).
    * **Pool de Trayectoria (Explotación):** Iterated Local Search (ILS), Simulated Annealing (SA).

* **Párrafo 2.4.2 — Protocolo de Inyección Semilla de Memoria ($x_{best}$):**
  * *Contenido:* Formular la reinicialización guiada tras un switch. Cuando el algoritmo finaliza su epoch, la mejor solución global lograda $x_{best}$ se inyecta en el nuevo solver:
    * En algoritmos poblacionales: $x^{(1)}_1 = x_{best}$, mientras que el resto de la población $x^{(1)}_i$ ($i \ge 2$) se genera con perturbación gaussiana o mixta alrededor de $x_{best}$.
    * En algoritmos de trayectoria: el estado inicial se fija como $x_0 = x_{best}$.

---

### 2.5. Problem Formulation 1: Discrete Multidimensional Knapsack Problem (MKP)

* **Párrafo 2.5.1 — Formulación Matemática Discreta:**
  * *Ecuaciones:*
    $$\max f(x) = \sum_{j=1}^{n} p_j x_j \quad \text{s.t.} \quad \sum_{j=1}^{n} r_{ij} x_j \le b_i, \quad i = 1, \dots, m, \quad x_j \in \{0, 1\}$$
  * *Contenido:* Explicar el manejo de restricciones mediante reparación de soluciones basándose en la razón utilidad/recurso (pseudo-utility ratio) y el esquema de binarización para metaheurísticas continuas.

---

### 2.6. Problem Formulation 2: Continuous IEEE CEC2022 Benchmark Suite

* **Párrafo 2.6.1 — Características del Benchmark CEC2022:**
  * *Contenido:* Describir las 12 funciones de prueba (F1-F12) categorizadas en: Unimodales ($F_1$), Multimodales Básicas ($F_2-F_5$), Híbridas ($F_6-F_8$) y de Composición ($F_9-F_{12}$). Especificar dimensiones evaluadas ($D=10, 20$) y límites del dominio $[-100, 100]^D$.

---

### 2.7. Real-World Case Study: Hybrid Renewable Energy System with Hydrogen (HRES2-H2)

* **Párrafo 2.7.1 — Modelos Físicos de Generación Renovable (WT & PV):**
  * *Ecuaciones:* Potencia eólica $P_{WT}(t)$ mediante curva cúbica de velocidad de viento $v(t)$, y potencia fotovoltaica $P_{PV}(t)$ considerando irradiancia $G(t)$ y temperatura de celda $T_c(t)$.

* **Párrafo 2.7.2 — Lógica de Despacho Horario y Almacenamiento de Hidrógeno:**
  * *Contenido:* Formular el balance energético en cada hora $t \in [1, 8760]$:
    $$P_{net}(t) = P_{WT}(t) + P_{PV}(t) - P_{load}(t)$$
  * Si $P_{net}(t) > 0$: Carga de Baterías $\to$ Producción de $H_2$ por Electrolizador $\to$ Vertido de exceso.
  * Si $P_{net}(t) < 0$: Descarga de Baterías $\to$ Generación con Celda de Combustible de $H_2 \to$ Compra a la red.

* **Párrafo 2.7.3 — Funciones Objetivo Financieras (LCOE, LCOH) y Restricción AGSR:**
  * *Ecuación del LCOE:*
    $$\text{LCOE} = \frac{\text{CAPEX} \cdot \text{CRF} + \text{OPEX}_{anual}}{\sum_{t=1}^{8760} P_{served}(t)}$$
  * *Restricción de Seguridad de Red (AGSR):*
    $$\text{AGSR} = \frac{\sum_{t=1}^{8760} P_{grid\_import}(t)}{\sum_{t=1}^{8760} P_{load}(t)} \le 20\%$$

---

### 2.8. Non-Parametric Statistical Inferential Protocol

* **Párrafo 2.8.1 — Protocolo de Validación e Inferencia No Paramétrica:**
  * *Contenido:* Especificar el número de corridas independientes ($N=31$) con semillas estocásticas controladas. Describir la batería de pruebas:
    1. **Prueba de Normalidad de Shapiro-Wilk:** Para verificar la no-normalidad de las distribuciones de fitness.
    2. **Prueba de Rangos con Signo de Wilcoxon:** Para comparaciones par a par entre la variante DTW/DDTW y los algoritmos independientes/competidores ($\alpha = 0.05$).
    3. **Prueba de Ranking Global de Friedman:** Para determinar el orden de dominancia estadística entre todos los métodos evaluados.

---

## 🛠️ Tabla de Hiperparámetros Recomendada para el Artículo

| Módulo / Algoritmo | Parámetro | Valor Asignado | Descripción / Definición |
|---|---|---|---|
| **Monitor DTW/DDTW** | Ventana ($W$) | 30 iteraciones | Historial deslizante de fitness |
| | Banda ($w$) | 0 (Auto 10%) | Banda de Sakoe-Chiba |
| | Paciencia ($P$) | 3 confirmaciones | Epochs consecutivos requeridos |
| | Meseta Máx. ($K_{max}$) | 15 iteraciones | Límite sin mejora local |
| | Percentiles ($P_{low}, P_{high}$) | 30.0 / 70.0 | Umbrales adaptativos de distancia |
| **Experimentos** | Corridas ($N$) | 31 ejecuciones | Semillas estocásticas independientes |
| | Max Iters | 1000 iteraciones | Criterio de parada global |
| **Sistema HRES2-H2** | Límite AGSR | $\le 20\%$ | Penetración máxima de red externa |
| | Período de simulación | 8,760 horas | Simulación de 1 año hora a hora |

---

## 💡 Próximos Pasos Recomendados para la Redacción

1. **Revisar el archivo creado en:** [`paper/materials_and_methods_estructura.md`](file:///c:/Users/Abduzcan0/Desktop/mkp_lb2/paper/materials_and_methods_estructura.md)
2. **Redacción por Subsecciones:** Podemos comenzar a redactar el texto formal en inglés técnico (MDPI standard) subsección por subsección, incluyendo las ecuaciones en LaTeX listas para incluir en el documento principal.
