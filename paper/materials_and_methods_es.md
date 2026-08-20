# Sección 2: Materiales y Métodos (Materials and Methods)

> **Manuscrito:** *A Dynamic Time Warping-Driven Adaptive Rotational Hybrid Metaheuristic Framework for Discrete, Continuous, and Energy System Optimization*  
> **Formato:** MDPI Standard (Español Académico Formal de Alto Impacto)

---

## 2. Materiales y Métodos

### 2.1. Arquitectura General del Framework Híbrido Rotacional Guiado por DTW

Para resolver de manera integral las deficiencias de estancamiento prematuro, pérdida de diversidad y falta de generalización multidominio identificadas en las metaheurísticas tradicionales, este trabajo propone un **Framework Híbrido Rotacional Adaptativo gobernado por un Monitor de Estancamiento basado en Alineamiento Temporal Dinámico (DTW/DDTW)**.

La filosofía central de la arquitectura se fundamenta en la explotación de la complementariedad algorítmica. De acuerdo con el Teorema del No Free Lunch (NFL), ninguna metaheurística individual posee una estrategia de búsqueda capaz de superar holgadamente a sus competidoras en todo el espacio de soluciones de problemas complejos. En consecuencia, el framework organiza las metaheurísticas en dos pools operativos disjuntos: un **Pool Poblacional ($\mathcal{P}_{pop}$)** enfocado en la exploración global del espacio de soluciones mediante dinámicas colectivas, y un **Pool de Trayectoria ($\mathcal{P}_{traj}$)** enfocado en la intensificación y explotación local acelerada.

```
+-----------------------------------------------------------------------------------+
|                        ORQUESTADOR HÍBRIDO ROTACIONAL DTW                         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         | Inicialización de Parámetros  |
                         |  y Solución Semilla Initial   |
                         +-------------------------------+
                                         |
                                         v
            +---------------------------------------------------------+
            |  Selección Aleatoria de Solver M ∈ P_pop (Exploración)  |
            +---------------------------------------------------------+
                                         |
                                         v
                        +---------------------------------+
                        |  Ejecución de Iteración t en M  |
                        +---------------------------------+
                                         |
                                         v
                        +---------------------------------+
                        | Monitor DTW/DDTW: Evaluador de  |
                        | Historial Reciente de Fitness X  |
                        +---------------------------------+
                                         |
                       ¿Se detecta estancamiento? (Fire)
                               /                   \
                        Sí    /                     \  No
                             /                       \
                            v                         v
           +----------------------------------+   +----------------------+
           | 1. Congelar Solver Actual M      |   | Continuar Época en M |
           | 2. Extraer Mejor Solución x_best  |   +----------------------+
           | 3. Conmutar a Pool P_traj        |
           | 4. Inyectar x_best en Nuevo Solver|
           +----------------------------------+
```

El flujo de control se estructura en **Épocas ($e$)**. A diferencia de los métodos de conmutación tradicionales que alternan metaheurísticas tras un número de iteraciones fijo o arbitrario, en este framework la duración de cada época es completamente dinámica. Una época finaliza únicamente cuando el **Monitor DTW/DDTW** detecta que la curva de convergencia instantánea ha perdido pendiente y ha entrado en una meseta de estancamiento no productiva. En ese instante, el orquestador congela la metaheurística activa, extrae la mejor solución global alcanzada hasta el momento ($x_{best}$) y conmuta la ejecución hacia una metaheurística del pool opuesto, inyectando $x_{best}$ como semilla inicial para evitar reinicios a ciegas.

---

### 2.2. Formulación Matemática del Motor de Detección de Estancamiento DTW y DDTW

El motor de detección evalúa continuamente el perfil de mejoría del algoritmo analizando la serie temporal del mejor valor de fitness registrado en una ventana deslizante de tamaño $W$, denotada como $X = [x_1, x_2, \dots, x_W] \in \mathbb{R}^W$.

#### 2.2.1. Definición de Series Baseline de Reference

Para determinar cuantitativamente si la trayectoria $X$ corresponde a una búsqueda progresiva o a una meseta de inactividad, el monitor construye dinámicamente dos series temporales de referencia de longitud $W$:

1. **Rampa de Progreso Ideal ($R = [r_1, r_2, \dots, r_W]$):** Representa una trayectoria de mejora constante con pendiente mínima $s_{min}$:
   $$r_k = x_1 + s_{min} \cdot (k - 1), \quad \forall k \in \{1, 2, \dots, W\}$$
   donde $s_{min}$ se configura de forma estática o se calcula automáticamente como el $1\%$ del rango de valores de fitness presentes en la ventana:
   $$s_{min} = 0.01 \cdot \frac{|x_W - x_1|}{W}$$

2. **Meseta Constante de Estancamiento ($C = [c_1, c_2, \dots, c_W]$):** Representa una ausencia total de mejora a partir del estado inicial de la ventana:
   $$c_k = x_1, \quad \forall k \in \{1, 2, \dots, W\}$$

#### 2.2.2. Alineamiento Temporal elástico vía Dynamic Time Warping (DTW)

La distancia DTW entre la serie observada $X$ y una serie de referencia $Y \in \{R, C\}$ se calcula evaluando la matriz de costo acumulado $D \in \mathbb{R}^{(W+1) \times (W+1)}$. La celda $(i, j)$ de la matriz se actualiza mediante la relación de recurrencia:

$$D(i, j) = d(x_i, y_j) + \min \left\{ D(i-1, j), D(i, j-1), D(i-1, j-1) \right\}$$

donde $d(x_i, y_j) = |x_i - y_j|$ representa la distancia absoluta punto a punto. La búsqueda de la ruta de alineamiento óptimo se restringe mediante la **Banda de Sakoe-Chiba** de ancho $w$, garantizando un límite superior a la deformación temporal y reduciendo la complejidad computacional a $\mathcal{O}(W \cdot w)$:

$$|i - j| \le w, \quad w = \max(1, \lfloor \gamma \cdot W \rfloor), \quad \gamma \in (0, 0.2]$$

Las condiciones de frontera e inicialización de la matriz son $D(0, 0) = 0$ y $D(i, 0) = D(0, j) = \infty$ para $i, j > 0$. La distancia DTW final corresponde al valor acumulado $D(W, W)$.

#### 2.2.3. Variante basada en Derivadas (DDTW)

Para hacer la detección inmune a desplazamientos en el eje vertical (magnitude shifts) y enfocarse exclusivamente en la forma y cambio de pendiente de la curva de convergencia, se implementa la variante **Derivative Dynamic Time Warping (DDTW)**. En DDTW, las series $X$ y $Y$ se transforman previamente mediante la primera diferencia finita:

$$\nabla x_k = x_k - x_{k-1}, \quad \text{con } \nabla x_1 = 0$$

La distancia DDTW resulta de aplicar el procedimiento DTW sobre las series de derivadas:

$$\text{DDTW}(X, Y) = \text{DTW}(\nabla X, \nabla Y)$$

#### 2.2.4. Métrica de Desviación Relativa ($\Delta$)

En cada iteración $t \ge W$, el monitor calcula dos distancias elásticas: la distancia a la rampa $D_1 = \text{DTW}(X, R)$ y la distancia a la constante $D_2 = \text{DTW}(X, C)$. Se define la métrica de desviación relativa $\Delta$ como:

$$\Delta = D_1 - D_2$$

Un valor cercano a cero o negativo en $D_2$, acompañado de un incremento en $D_1$ y $\Delta$, indica que la trayectoria observada es morfológicamente idéntica a una meseta inactiva.

---

### 2.3. Estrategia de Umbralización Adaptativa mediante Percentiles Móviles

El uso de umbrales fijos para clasificar distancias DTW es altamente ineficiente debido a las diferencias de escala entre distintas funciones objetivo. Por ello, el framework implementa un mecanismo de umbrales adaptativos basados en percentiles históricos.

Se mantienen tres memorias circulares de largo plazo que almacenan los valores recientes de las métricas: $\mathcal{H}_{D1}$, $\mathcal{H}_{D2}$ y $\mathcal{H}_{\Delta}$. Cuando el historial cuenta con al menos 10 registros, los umbrales dinámicos se recalculan como:

$$\theta_c = \text{Percentil}(\mathcal{H}_{D2}, P_{low})$$
$$\theta_r = \text{Percentil}(\mathcal{H}_{D1}, P_{high})$$
$$\theta_\Delta = \text{Percentil}(\mathcal{H}_{\Delta}, P_{high})$$

donde $P_{low} \in [20, 40]$ y $P_{high} \in [60, 80]$ son percentiles configurables (predeterminados en $P_{low} = 30.0$ y $P_{high} = 70.0$).

La condición de estancamiento instantánea $\text{Stagnant}(t)$ se satisface si se cumplen simultáneamente tres criterios:

$$\text{Stagnant}(t) = \left( N_{no\_improve} \ge K_{max} \right) \land \left( D_2 \le \theta_c \right) \land \left( (D_1 \ge \theta_r) \lor (\Delta \ge \theta_\Delta) \right)$$

donde $N_{no\_improve}$ es el número de iteraciones consecutivas sin mejora del óptimo local y $K_{max}$ es el límite de tolerancia de meseta. Para evitar falsos positivos provocados por fluctuaciones estocásticas de corto plazo, se introduce el parámetro de **paciencia ($P$)**. La alarma de conmutación ($\text{Trigger Alarm}$) se dispara únicamente cuando la condición $\text{Stagnant}(t)$ se sostiene durante $P$ iteraciones consecutivas:

$$S_t = \begin{cases} S_{t-1} + 1, & \text{si } \text{Stagnant}(t) = \text{Verdadero} \\ 0, & \text{si } \text{Stagnant}(t) = \text{Falso} \end{cases}$$

$$\text{Trigger Alarm} \iff S_t \ge P$$

---

### 2.4. Pools Metaheurísticos y Protocolo de Inyección de Memoria Semilla

#### 2.4.1. Definición de los Pools Algorítmicos

El framework integra ocho metaheurísticas bien consolidadas en la literatura, clasificadas según su filosofía de búsqueda:

1. **Pool Poblacional ($\mathcal{P}_{pop}$ - Exploración Global):**
   * *Particle Swarm Optimization (PSO):* Simulación de bandadas basada en vectores de velocidad, cognición individual y atracción social.
   * *Grey Wolf Optimizer (GWO):* Modelado de jerarquía de caza (lobos $\alpha, \beta, \delta$).
   * *Whale Optimization Algorithm (WOA):* Maniobra de caza con red de burbujas en espiral y búsqueda aleatoria.
   * *Elephant Herding Optimization (EHO):* Operadores de clan y separación de elefantes macho.
   * *Ant Colony Optimization (ACO):* Optimización basada en depósitos y evaporación de feromonas.
   * *Artificial Bee Colony (ABC):* Búsqueda dividida entre abejas empleadas, observadoras y exploradoras.

2. **Pool de Trayectoria ($\mathcal{P}_{traj}$ - Explotación Local):**
   * *Iterated Local Search (ILS):* Perturbación sistemática y descenso local estocástico.
   * *Simulated Annealing (SA):* Aceptación probabilística de soluciones de inferior calidad mediante esquema de enfriamiento exponencial $T_k = T_0 \cdot \alpha^k$.

#### 2.4.2. Protocolo de Inyección de Memoria Semilla ($x_{best}$)

Cuando la alarma DTW fuerza una rotación de solver en la época $e$, se ejecuta el protocolo de transferencia de conocimiento:

1. **Extracción:** Se recupera la mejor solución global válida encontrada desde el inicio de la ejecución, denotada como $x_{best}^* \in \mathbb{R}^D$.
2. **Inyección en Metaheurísticas Poblacionales:** El nuevo solver poblacional $M \in \mathcal{P}_{pop}$ inicializa su primer individuo como el vector exacto $x_1^{(1)} = x_{best}^*$. Los $N_{pop} - 1$ individuos restantes se generan mediante una inyección mixta: un $50\%$ mediante perturbación gaussiana centrada en $x_{best}^*$:
   $$x_i^{(1)} = x_{best}^* + \mathcal{N}(0, \sigma^2) \cdot (U - L), \quad \sigma = 0.05$$
   y el $50\%$ restante mediante muestreo uniforme aleatorio en el espacio de búsqueda $[L, U]^D$ para preservar la diversidad.
3. **Inyección en Metaheurísticas de Trayectoria:** El nuevo solver de trayectoria $M \in \mathcal{P}_{traj}$ establece su punto de partida inicial de manera determinista como $x_0 = x_{best}^*$.

---

### 2.5. Formulación del Problema 1: Problema de la Mochila Multidimensional Discreta (MKP)

El Problema de la Mochila Multidimensional es un problema de optimización combinatoria NP-hard formulado como:

$$\max f(x) = \sum_{j=1}^{n} p_j x_j$$

$$\text{sujeto a} \quad \sum_{j=1}^{n} r_{ij} x_j \le b_i, \quad \forall i \in \{1, 2, \dots, m\}$$

$$x_j \in \{0, 1\}, \quad \forall j \in \{1, 2, \dots, n\}$$

donde $n$ es el número de objetos, $m$ es el número de recursos restringidos, $p_j > 0$ es el beneficio del objeto $j$, $r_{ij} \ge 0$ es la cantidad de recurso $i$ consumida por el objeto $j$, y $b_i > 0$ es la capacidad máxima del recurso $i$.

#### Reparación de Soluciones e Infactibilidad

Para garantizar que todas las soluciones evaluadas sean factibles, se aplica un operador de reparación heurístico basado en el ratio de pseudo-utilidad $u_j$:

$$u_j = \frac{p_j}{\sum_{i=1}^{m} \frac{r_{ij}}{b_i}}$$

Si una solución infringe alguna restricción ($\exists i : \sum r_{ij} x_j > b_i$), los objetos incluidos ($x_j = 1$) se remueven en orden creciente de $u_j$ hasta restablecer la factibilidad. Posteriormente, se realiza una fase de codicia (*greedy addition*) insertando objetos excluidos ($x_j = 0$) en orden decreciente de $u_j$ mientras no se violen las capacidades.

---

### 2.6. Formulación del Problema 2: Suite de Funciones Continuas IEEE CEC2022

Para evaluar la capacidad de generalización en espacios continuos no convexos de alta dimensión, se adopta la suite oficial **IEEE CEC2022**, compuesta por 12 funciones de minimización definidas en el dominio $[-100, 100]^D$ con dimensiones $D \in \{10, 20\}$:

$$\min f(x), \quad x \in [-100, 100]^D, \quad \text{con óptimo global } f(x^*) = F_i^*$$

```
+-----------------------------------------------------------------------------------+
|                     SUITE DE FUNCIONES DE PRUEBA IEEE CEC2022                     |
+-----------------------------------------------------------------------------------+
| ID  | Nombre de la Función            | Tipo de Paisaje de Fitness    | Óptimo F_i*|
+-----+---------------------------------+-------------------------------+-----------+
| F1  | Shifted Sphere Function         | Unimodal Básica               | 300       |
| F2  | Shifted Weighted Rosenbrock     | Multimodal Básica (No convexa)| 400       |
| F3  | Shifted Lunacek Bi-Rastrigin    | Multimodal Básica (Mesetas)   | 600       |
| F4  | Expanded Non-Continuous Ackley | Multimodal Básica             | 800       |
| F5  | Shifted Levy Function           | Multimodal Básica             | 900       |
| F6  | Hybrid Function 1 (N=3)         | Híbrida (Sub-espacios)        | 1800      |
| F7  | Hybrid Function 2 (N=6)         | Híbrida (Sub-espacios)        | 2000      |
| F8  | Hybrid Function 3 (N=5)         | Híbrida (Sub-espacios)        | 2200      |
| F9  | Composition Function 1 (N=5)    | Composición Gaussiana         | 2300      |
| F10 | Composition Function 2 (N=4)    | Composición Gaussiana         | 2400      |
| F11 | Composition Function 3 (N=5)    | Composición Gaussiana         | 2600      |
| F12 | Composition Function 4 (N=6)    | Composición Gaussiana         | 2700      |
+-----+---------------------------------+-------------------------------+-----------+
```

---

### 2.7. Estudio de Caso en Ingeniería Real: Sistema Energético Renovable Híbrido con Hidrógeno (HRES2-H2)

Como aplicación práctica en ingeniería de descarbonización, el framework se aplica a la optimización dimensional y de despacho de una microred industrial híbrida (HRES2-H2) ubicada en Baotou, China ($41.70^\circ \text{N}, 110.43^\circ \text{E}$). El sistema integra turbinas eólicas ($P_{WT}$), paneles fotovoltaicos ($P_{PV}$), banco de baterías BESS ($P_{bat}$), electrolizador PEM ($P_{el}$), celda de combustible ($P_{fc}$) y tanque de almacenamiento de $H_2$.

#### 2.7.1. Modelos de Generación Renovable

1. **Aerogenerador Eólico:** La potencia eólica horaria $P_{WT}(t)$ se calcula mediante la curva característica de velocidad del viento $v(t)$:
   $$P_{WT}(t) = \begin{cases} 0, & v(t) < v_{in} \text{ o } v(t) \ge v_{out} \\ P_{rated} \cdot \frac{v(t)^3 - v_{in}^3}{v_{rated}^3 - v_{in}^3}, & v_{in} \le v(t) < v_{rated} \\ P_{rated}, & v_{rated} \le v(t) < v_{out} \end{cases}$$
   con $v_{in} = 2.5 \text{ m/s}$, $v_{rated} = 10.5 \text{ m/s}$, $v_{out} = 25.0 \text{ m/s}$ y $P_{rated} = 5.0 \text{ MW}$.

2. **Arreglo Fotovoltaico:** La potencia solar $P_{PV}(t)$ en función de la irradiancia $G(t)$ y la temperatura ambiente $T_a(t)$ es:
   $$T_c(t) = T_a(t) + \frac{G(t)}{800} \cdot (NOCT - 20)$$
   $$P_{PV}(t) = P_{PV,STC} \cdot \frac{G(t)}{1000} \cdot \left[ 1 + \gamma_T \cdot (T_c(t) - 25) \right] \cdot f_{derating}$$
   con $NOCT = 47^\circ\text{C}$, coeficient $\gamma_T = -0.45\%/^\circ\text{C}$ y factor de pérdida $f_{derating} = 0.90$.

#### 2.7.2. Lógica de Despacho Horario (8,760 horas)

En cada hora $t \in \{1, 2, \dots, 8760\}$, se determina el balance neto de potencia $P_{net}(t) = P_{WT}(t) + P_{PV}(t) - P_{load}(t)$:

* **Superávit ($P_{net}(t) > 0$):** El exceso de energía se destina en prioridad a: (1) Cargar la batería hasta su límite $SOC_{max}$, (2) Alimentar el electrolizador PEM para producir $H_2$ comprimido a razón de $\eta_{el} = 75\%$, y (3) Inyectar el remanente a la red eléctrica externa.
* **Deficit ($P_{net}(t) < 0$):** El déficit se cubre mediante: (1) Descarga del banco de baterías hasta $SOC_{min}$, (2) Generación de energía limpia vía celda de combustible alimentada por $H_2$, y (3) Importación de energía desde la red eléctrica externa.

#### 2.7.3. Funciones Objetivo Financieras y Restricción AGSR

El problema busca minimizar el Costo Nivelado de la Energía (LCOE) expresado en CNY/kWh:

$$\min f(x) = \text{LCOE} = \frac{\text{CAPEX} \cdot \text{CRF}(r, N_{proj}) + \text{OPEX}_{anual} + \text{REP}_{anual} - \text{REV}_{grid}}{\sum_{t=1}^{8760} P_{load}(t)}$$

donde la tasa de recuperación de capital es $\text{CRF}(r, N) = \frac{r(1+r)^N}{(1+r)^N - 1}$ con $r = 4.35\%$ y $N_{proj} = 25$ años.

**Restricción de Seguridad y Autonomía Energética de Red (AGSR):**
Para evitar la dependencia excesiva de la red externa, la proporción anual de energía importada respecto a la demanda total no debe superar el límite estricto del $20\%$:

$$\text{AGSR} = \frac{\sum_{t=1}^{8760} P_{grid\_import}(t)}{\sum_{t=1}^{8760} P_{load}(t)} \le 0.20$$

---

### 2.8. Protocolo de Inferencia Estadística No Paramétrica

Dado que la naturaleza estocástica de las metaheurísticas produce distribuciones de resultados no necesariamente gaussianas, la evaluación experimental se diseña bajo un riguroso protocolo de inferencia no paramétrica considerando $N = 31$ ejecuciones independientes por problema.

1. **Prueba de Normalidad de Shapiro-Wilk:** Se evalúa la hipótesis nula $H_0$ de normalidad sobre los vectores de resultados finales. Si el $p$-valor resulta $p < 0.05$, se rechaza $H_0$, justificando el uso exclusivo de pruebas no paramétricas.
2. **Prueba de Rangos con Signo de Wilcoxon:** Se realiza una prueba emparejada par a par entre el framework propuesto (DTW/DDTW) y cada metaheurística competidora con un nivel de significancia del $\alpha = 0.05$. Se reportan los estadísticos $W^+$, $W^-$ y el $p$-valor asociado.
3. **Prueba de Rangos de Friedman:** Para obtener un ranking global de dominancia estadística entre todos los algoritmos evaluados sobre el conjunto de instancias, se calcula el estadístico de Friedman $\chi_F^2$ y los rangos promedio (*Mean Ranks*).

---

## 🛠️ Resumen de Parámetros de Simulación y Ejecución

| Dominio | Parámetro | Valor Asignado | Unidad / Descripción |
|---|---|---|---|
| **Monitor DTW/DDTW** | Ventana ($W$) | 30 | Iteraciones de historial reciente |
| | Banda Sakoe-Chiba ($w$) | 0 (Auto 10%) | Restricción de deformación temporal |
| | Paciencia ($P$) | 3 | Confirmaciones consecutivas para trigger |
| | Meseta Máx ($K_{max}$) | 15 | Iteraciones sin mejora local |
| | Percentiles ($P_{low}, P_{high}$) | 30.0 / 70.0 | Percentiles para umbrales móviles |
| **Simulación HRES2-H2** | Período evaluado | 8,760 | Horas continuas (1 año completo) |
| | Tasa de Descuento ($r$) | 4.35% | Tasa real de descuento anual |
| | Vida del Proyecto | 25 | Años de vida útil del sistema |
| | Límite AGSR | $\le 20.0\%$ | Máxima importación de red permitida |
| **Protocolo Estadístico** | Ejecuciones ($N$) | 31 | Corridas independientes con semillas fija (42) |
| | Nivel de Significación ($\alpha$) | 0.05 | Umbral de rechazo para $p$-valores |
