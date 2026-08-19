---
# 2. Materiales y Métodos

## 2.1. Arquitectura General del Framework Híbrido Rotacional Guiado por DTW

Para resolver de manera integral las deficiencias de estancamiento prematuro, pérdida de diversidad y falta de generalización multidominio identificadas en las metaheurísticas convencionales, este trabajo propone un **Framework Híbrido Rotacional Adaptativo** gobernado por un **Monitor de Estancamiento basado en Alineamiento Temporal Dinámico (DTW/DDTW)**. A diferencia de los enfoques híbridos existentes que alternan algoritmos mediante reglas estáticas —como umbrales de paciencia fijos o periodos de conmutación predefinidos—, el presente framework emplea la distancia DTW sobre la curva de convergencia reciente como criterio dinámico y fundamentado matemáticamente para detectar pérdidas de progreso con alta precisión y sin falsos positivos prematuros.

La filosofía central de la arquitectura se fundamenta en la explotación de la complementariedad algorítmica. De acuerdo con el Teorema del No Free Lunch [Wolpert & Macready, 1997], ninguna metaheurística individual posee una estrategia de búsqueda capaz de superar consistentemente a sus competidoras en todo el universo de problemas de optimización. En consecuencia, el framework organiza las metaheurísticas en dos pools operativos disjuntos y especializados: un **Pool Poblacional** ($\mathcal{P}_{pop}$) que agrupa algoritmos orientados a la exploración global mediante dinámicas colectivas de gran escala, y un **Pool de Trayectoria** ($\mathcal{P}_{traj}$) que agrupa algoritmos orientados a la intensificación y explotación local acelerada mediante perturbaciones guiadas.

El flujo de control del orquestador se estructura en **Épocas** ($e = 1, 2, \dots$). En cada época, un único solver $M^{(e)}$ seleccionado aleatoriamente del pool correspondiente ejecuta iteraciones de búsqueda sobre el problema objetivo. A diferencia de los métodos tradicionales de conmutación que determinan el fin de la época por un contador de iteraciones fijo, en este framework la duración de cada época es enteramente **dinámica y adaptativa**: la época termina únicamente cuando el Monitor DTW/DDTW detecta que el perfil reciente de fitness ha degenerado en una meseta de estancamiento estadísticamente significativa. En ese instante, el orquestador ejecuta el **Protocolo de Conmutación y Transferencia de Memoria**, extrayendo la mejor solución global lograda hasta el momento ($x_{best}$), conmutando hacia el pool alternante e inyectando $x_{best}$ como semilla de inicialización del nuevo solver para evitar la pérdida total del progreso acumulado.

**Tabla 1.** Resumen de parámetros operativos del framework híbrido DTW.

| Parámetro | Símbolo | Valor | Descripción |
|---|---|---|---|
| Tamaño de ventana | $W$ | 30 | Historial reciente de fitness analizado por el monitor |
| Banda Sakoe-Chiba | $w$ | $\lfloor 0.1 \cdot W \rfloor = 3$ | Restricción de deformación temporal en DTW |
| Paciencia | $P$ | 3 | Confirmaciones consecutivas requeridas para disparar la alarma |
| Tolerancia de meseta | $K_{max}$ | 15 | Iteraciones consecutivas sin mejora para activar la detección |
| Percentil inferior | $P_{low}$ | 30.0 | Percentil para el umbral adaptativo de distancia a constante |
| Percentil superior | $P_{high}$ | 70.0 | Percentil para los umbrales adaptativos de rampa y delta |
| Max. iteraciones | $T_{max}$ | 1,000 | Criterio de parada global por iteraciones |
| Corridas independientes | $N$ | 31 | Ejecuciones estocásticas con semilla base 42 |

---

## 2.2. Formulación Matemática del Monitor de Estancamiento DTW/DDTW

El monitor de estancamiento constituye el núcleo algorítmico del framework. En cada iteración $t$, el monitor recibe el mejor valor de fitness registrado hasta ese momento por el solver activo y mantiene un historial deslizante $\mathbf{X} = [x_{t-W+1}, \dots, x_t] \in \mathbb{R}^W$ de los últimos $W$ valores. El análisis se activa únicamente cuando se han acumulado al menos $W$ observaciones ($t \geq W$).

### 2.2.1. Series de Referencia Baseline

Para cuantificar el grado de progreso presente en la ventana $\mathbf{X}$, el monitor construye en tiempo real dos series de referencia de longitud $W$ ancladas en el valor inicial de la ventana $x_1 = x_{t-W+1}$:

1. **Rampa de Progreso Ideal** $\mathbf{R} = [r_1, r_2, \dots, r_W]$: representa la trayectoria de convergencia esperada bajo mejora sostenida con pendiente mínima $s_{min}$:
$$r_k = x_1 + s_{min} \cdot (k-1), \quad k = 1, \dots, W$$
La pendiente mínima se calcula de forma adaptativa como el $1\%$ del rango de variación de la ventana:
$$s_{min} = 0.01 \cdot \frac{|x_W - x_1|}{W}$$

2. **Meseta Constante de Estancamiento** $\mathbf{C} = [c_1, c_2, \dots, c_W]$: representa la ausencia total de mejora, es decir, un valor de fitness completamente inmovilizado:
$$c_k = x_1, \quad k = 1, \dots, W$$

### 2.2.2. Distancia DTW con Banda de Sakoe-Chiba

La distancia entre la trayectoria observada $\mathbf{X}$ y cualquier serie de referencia $\mathbf{Y} \in \{\mathbf{R}, \mathbf{C}\}$ se calcula mediante el algoritmo de **Dynamic Time Warping (DTW)**, que encuentra la correspondencia óptima entre elementos de las dos series a través de una matriz de costo acumulado $\mathbf{D} \in \mathbb{R}^{(W+1) \times (W+1)}$. La actualización de cada celda $(i, j)$ sigue la relación de recurrencia:

$$D(i,j) = |x_i - y_j| + \min\bigl\{D(i-1,j),\ D(i,j-1),\ D(i-1,j-1)\bigr\}$$

con condiciones de frontera $D(0,0) = 0$ y $D(i,0) = D(0,j) = +\infty$ para $i,j > 0$. Para limitar la deformación temporal admisible y reducir la complejidad de $\mathcal{O}(W^2)$ a $\mathcal{O}(W \cdot w)$, se aplica la **Banda de Sakoe-Chiba** de ancho $w$, restringiendo la búsqueda a celdas que satisfacen $|i-j| \leq w$, donde $w = \max(1, \lfloor 0.1 \cdot W \rfloor)$. La distancia DTW final corresponde al valor acumulado en la esquina opuesta: $\text{DTW}(\mathbf{X}, \mathbf{Y}) = D(W, W)$.

### 2.2.3. Variante DDTW basada en Derivadas

Para hacer el monitor **invariante a desplazamientos en magnitud** (magnitude-shift invariance) y sensible exclusivamente a la forma y pendiente de la curva de convergencia, se implementa la variante **Derivative Dynamic Time Warping (DDTW)**. Bajo DDTW, tanto la serie observada como la de referencia se transforman mediante la primera diferencia finita antes de calcular la distancia DTW:

$$\nabla x_k = x_k - x_{k-1}, \quad \text{con } \nabla x_1 = 0$$
$$\text{DDTW}(\mathbf{X}, \mathbf{Y}) = \text{DTW}(\nabla \mathbf{X},\ \nabla \mathbf{Y})$$

Esta transformación convierte la comparación morfológica de niveles absolutos en una comparación de **tasas de cambio**, lo que permite detectar mesetas incluso cuando la magnitud del fitness varía considerablemente entre diferentes instancias o funciones de prueba. El parámetro `use_ddtw` en `StagnationConfig` selecciona el modo de operación del monitor.

### 2.2.4. Métricas de Diagnóstico de Estancamiento

En cada iteración $t \geq W$, el monitor calcula las dos distancias DTW/DDTW de diagnóstico:

$$D_1 = \text{DTW/DDTW}(\mathbf{X}, \mathbf{R}), \qquad D_2 = \text{DTW/DDTW}(\mathbf{X}, \mathbf{C})$$

y la métrica de desviación relativa:

$$\Delta = D_1 - D_2$$

La interpretación geométrica de estas métricas es directa: si la trayectoria $\mathbf{X}$ está progresando, $D_1$ será pequeña (similar a la rampa) y $D_2$ grande (diferente a la constante), resultando en $\Delta < 0$. Si la trayectoria está estancada, $D_2 \approx 0$, $D_1$ crece, y $\Delta > 0$ con valor positivo creciente.

---

## 2.3. Umbralización Adaptativa mediante Percentiles Móviles

El empleo de umbrales fijos para clasificar distancias DTW es fundamentalmente inadecuado en optimización multidominio, debido a que las escalas de fitness varían en varios órdenes de magnitud entre problemas discretos (MKP, escala de decenas de miles), continuos (CEC2022, escala de centenas) e industriales (HRES2-H2, escala sub-unitaria de CNY/kWh). Para garantizar la portabilidad del monitor sin re-ajuste manual de hiperparámetros, el framework implementa un mecanismo de **umbralización dinámica basada en percentiles móviles históricos**.

Se mantienen tres buffers circulares acumulativos: $\mathcal{H}_{D_1}$, $\mathcal{H}_{D_2}$ y $\mathcal{H}_{\Delta}$, que registran el historial de los valores de $D_1$, $D_2$ y $\Delta$ de todas las iteraciones anteriores desde el inicio de la ejecución. Una vez que cada buffer contiene al menos 10 registros, los umbrales de clasificación se recalculan en cada iteración como:

$$\theta_c = \text{Perc}(\mathcal{H}_{D_2},\ P_{low}), \qquad \theta_r = \text{Perc}(\mathcal{H}_{D_1},\ P_{high}), \qquad \theta_\Delta = \text{Perc}(\mathcal{H}_\Delta,\ P_{high})$$

donde $P_{low} = 30.0$ y $P_{high} = 70.0$ son los percentiles predeterminados. Si el historial es insuficiente (menos de 10 registros), se emplean umbrales estáticos de arranque: $\theta_c = 0.1W$, $\theta_r = 0.5W$, $\theta_\Delta = 0.3W$.

La **condición triple de estancamiento** $\text{Stagnant}(t)$ se satisface cuando convergen simultáneamente tres criterios independientes:

$$\text{Stagnant}(t) = \underbrace{\left(N_{no\_improve} \geq K_{max}\right)}_{\text{meseta de paciencia}} \land \underbrace{\left(D_2 \leq \theta_c\right)}_{\text{morfología constante}} \land \underbrace{\left(D_1 \geq \theta_r \lor \Delta \geq \theta_\Delta\right)}_{\text{desviación de progreso}}$$

donde $N_{no\_improve}$ es el contador de iteraciones consecutivas sin mejora en el mejor valor de fitness del solver activo. Para filtrar disparos espurios provocados por fluctuaciones estocásticas de corto plazo, se aplica un mecanismo de **racha de confirmación** con parámetro de paciencia $P$. El contador de racha $S_t$ se actualiza como:

$$S_t = \begin{cases} S_{t-1} + 1 & \text{si } \text{Stagnant}(t) \\ 0 & \text{en otro caso} \end{cases}$$

La **alarma de conmutación** se dispara únicamente cuando $S_t \geq P$, requiriendo que el estancamiento persista durante al menos $P = 3$ iteraciones consecutivas antes de forzar la rotación.

---

## 2.4. Pools Metaheurísticos y Protocolo de Inyección de Memoria Semilla

### 2.4.1. Composición de los Pools Algorítmicos

El framework integra nueve metaheurísticas consolidadas en la literatura de computación evolutiva, seleccionadas para maximizar la diversidad de mecanismos de búsqueda y minimizar la correlación de comportamiento entre sus operadores. Se organizan en dos pools disjuntos:

**Pool Poblacional** $\mathcal{P}_{pop}$ (exploración global — 5 algoritmos):

- **Particle Swarm Optimization (PSO)** [Kennedy & Eberhart, 1995]: modela el comportamiento social de bandadas. La velocidad de cada partícula $i$ en la dimensión $d$ se actualiza como $v_{i,d}^{t+1} = w \cdot v_{i,d}^t + c_1 r_1 (p_{i,d} - x_{i,d}^t) + c_2 r_2 (g_d - x_{i,d}^t)$, donde $w$ es la inercia, $c_1$ y $c_2$ los coeficientes de aceleración cognitivo y social, y $p_{i,d}$, $g_d$ las mejores posiciones individual y global. Se emplea inercia lineal decreciente $w_{max} \to w_{min}$ durante la época.
- **Grey Wolf Optimizer (GWO)** [Mirjalili et al., 2014]: simula la jerarquía de caza de lobos grises ($\alpha, \beta, \delta$). La posición de actualización pondera los vectores de dirección de los tres líderes con coeficientes $A$ y $C$ variables.
- **Whale Optimization Algorithm (WOA)** [Mirjalili & Lewis, 2016]: alterna entre la estrategia de búsqueda aleatoria y la maniobra de ataque en red de burbujas en espiral según un parámetro de probabilidad $p$ por iteración.
- **Elephant Herding Optimization (EHO)** [Wang et al., 2015]: implementa dos operadores complementarios —actualización de clan que atrae individuos hacia el matriarca de cada subgrupo, y separación del macho dominante hacia la periferia del espacio de búsqueda.
- **Ant Colony Optimization (ACO)** [Dorigo & Stützle, 2004]: en su variante para espacios continuos, actualiza un archivo de soluciones ponderadas con feromonas que sirve como distribución de muestreo gaussiano para la exploración del vecindario.

**Pool de Trayectoria** $\mathcal{P}_{traj}$ (explotación local — 4 algoritmos):

- **Iterated Local Search (ILS)** [Lourenço et al., 2003]: aplica perturbaciones sistemáticas sobre la mejor solución conocida seguidas de una fase de descenso local estocástico, iterando para escapar de óptimos locales mediante diversificación controlada.
- **Variable Neighborhood Search (VNS)** [Mladenović & Hansen, 1997]: explora dinámicamente un conjunto jerárquico de $k_{max}$ estructuras de vecindario $\mathcal{N}_k$, alternando fases de perturbación sistemática (shaking) y refinamiento local por descenso, expandiendo el radio de búsqueda ante estancamientos y retornando al vecindario primario tras cada éxito.
- **Tabu Search (TS)** [Glover, 1986]: implementa memoria adaptativa de corto plazo mediante regiones tabú (bolas de exclusión con radio $r_{tabu}$) que restringen el ciclado sobre óptimos locales previamente visitados, complementado con un criterio de aspiración que invalida la prohibición si la solución supera al mejor valor global.
- **Simulated Annealing (SA)** [Kirkpatrick et al., 1983]: implementa la aceptación probabilística de soluciones de menor calidad mediante el criterio de Metropolis $P(accept) = \exp\bigl(-\Delta f / T_k\bigr)$, con temperatura decreciente según el esquema de enfriamiento geométrico $T_k = T_0 \cdot \alpha^k$.

### 2.4.2. Protocolo de Inyección de Memoria Semilla ($x_{best}^*$)

El protocolo de transferencia de conocimiento inter-época es el mecanismo que diferencia este framework de un portafolio de algoritmos ejecutados de forma independiente. Cuando el Monitor DTW/DDTW dispara la alarma de conmutación al final de la época $e$, se ejecutan los siguientes pasos:

1. **Extracción del Conocimiento Acumulado:** Se recupera la mejor solución global válida encontrada desde el inicio de la ejecución, $x_{best}^* \in \mathbb{R}^D$, y su valor de fitness asociado $f^* = f(x_{best}^*)$.

2. **Inyección en Algoritmos Poblacionales:** Para el nuevo solver $M^{(e+1)} \in \mathcal{P}_{pop}$ con población de tamaño $N_{pop}$, la inicialización se realiza mediante una estrategia **mixta** que preserva simultáneamente la guía elitista y la diversidad exploratoria:
   - El primer individuo recibe exactamente la semilla: $\mathbf{x}_1^{(0)} = x_{best}^*$.
   - El $\lfloor N_{pop}/2 \rfloor$ de los individuos restantes se generan mediante perturbación gaussiana centrada en $x_{best}^*$: $\mathbf{x}_i^{(0)} = x_{best}^* + \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I}) \cdot (U - L)$, con $\sigma = 0.05$.
   - El resto se inicializa mediante muestreo uniforme en el dominio $[L, U]^D$ para garantizar cobertura exploratoria.

3. **Inyección en Algoritmos de Trayectoria:** Para el nuevo solver $M^{(e+1)} \in \mathcal{P}_{traj}$, el punto de partida se fija de forma determinista como $x_0 = x_{best}^*$, permitiendo que el algoritmo de explotación local refine la mejor región del espacio de búsqueda identificada hasta el momento.

---

## 2.5. Problema de Optimización 1: Mochila Multidimensional Discreta (MKP)

El **Problema de la Mochila Multidimensional** (Multidimensional Knapsack Problem, MKP) es un problema de optimización combinatoria NP-hard que generaliza la mochila 0-1 clásica a $m$ restricciones de capacidad simultáneas. Su formulación matemática es:

$$\max \quad f(\mathbf{x}) = \sum_{j=1}^{n} p_j x_j$$

$$\text{sujeto a} \quad \sum_{j=1}^{n} r_{ij} x_j \leq b_i, \quad \forall\, i \in \{1, \dots, m\}$$

$$x_j \in \{0, 1\}, \quad \forall\, j \in \{1, \dots, n\}$$

donde $n$ denota el número de objetos candidatos, $p_j > 0$ el beneficio del objeto $j$, $r_{ij} \geq 0$ el consumo del recurso $i$ por el objeto $j$, y $b_i > 0$ la capacidad máxima del recurso $i$. La dificultad computacional del MKP radica en la combinación de la naturaleza binaria de las variables de decisión con la necesidad de satisfacer simultáneamente $m$ restricciones de desigualdad entrelazadas.

Los experimentos se realizan sobre las instancias de referencia del conjunto **OR-Library MKNAPCB** [Chu & Beasley, 1998], que proporciona instancias con $n \in \{100, 250, 500\}$ objetos y $m \in \{5, 10, 30\}$ restricciones, con soluciones óptimas conocidas que permiten calcular el gap de optimización porcentual.

#### Mecanismo de Binarización y Reparación de Soluciones

Para operar metaheurísticas de espacio continuo sobre el espacio discreto $\{0,1\}^n$ del MKP, se aplica la función de transferencia sigmoide para mapear velocidades PSO o posiciones continuas a probabilidades de selección binaria. Dado que las perturbaciones estocásticas pueden generar soluciones infactibles, se aplica el **operador de reparación heurístico basado en pseudo-utilidad** $u_j$:

$$u_j = \frac{p_j}{\displaystyle\sum_{i=1}^{m} \dfrac{r_{ij}}{b_i}}$$

Si la solución viola alguna restricción de capacidad, se eliminan objetos incluidos ($x_j = 1$) en orden creciente de $u_j$ hasta restablecer la factibilidad. Tras la reparación, se ejecuta una fase de **adición codiciosa** (*greedy addition*) que inserta objetos excluidos ($x_j = 0$) en orden decreciente de $u_j$ mientras no se violen las restricciones, maximizando el aprovechamiento de la capacidad residual.

---

## 2.6. Problema de Optimización 2: Suite Continua IEEE CEC2022

Para evaluar la capacidad de generalización del framework en el dominio de la optimización continua no convexa de alta dimensión, se adopta la **Suite IEEE CEC2022** [Abhishek et al., 2022], que comprende 12 funciones de minimización distribuidas en cuatro categorías de complejidad creciente:

$$\min \quad f(\mathbf{x}), \quad \mathbf{x} \in [-100, 100]^D, \quad D \in \{10, 20\}$$

**Tabla 2.** Funciones de prueba de la Suite IEEE CEC2022.

| ID | Función | Categoría | Óptimo $F_i^*$ |
|:---:|:---|:---:|:---:|
| $F_1$ | Shifted Sphere | Unimodal básica | 300 |
| $F_2$ | Shifted Weighted Rosenbrock | Multimodal básica | 400 |
| $F_3$ | Shifted Lunacek Bi-Rastrigin | Multimodal básica | 600 |
| $F_4$ | Expanded Non-Continuous Ackley | Multimodal básica | 800 |
| $F_5$ | Shifted Lévy | Multimodal básica | 900 |
| $F_6$ | Hybrid Function 1 ($N=3$ subfunciones) | Híbrida | 1800 |
| $F_7$ | Hybrid Function 2 ($N=6$ subfunciones) | Híbrida | 2000 |
| $F_8$ | Hybrid Function 3 ($N=5$ subfunciones) | Híbrida | 2200 |
| $F_9$ | Composition Function 1 ($N=5$ componentes) | Composición gaussiana | 2300 |
| $F_{10}$ | Composition Function 2 ($N=4$ componentes) | Composición gaussiana | 2400 |
| $F_{11}$ | Composition Function 3 ($N=5$ componentes) | Composición gaussiana | 2600 |
| $F_{12}$ | Composition Function 4 ($N=6$ componentes) | Composición gaussiana | 2700 |

Las funciones unimodales ($F_1$) permiten evaluar la eficiencia de convergencia pura; las multimodales básicas ($F_2$–$F_5$) introducen múltiples óptimos locales y mesetas; las híbridas ($F_6$–$F_8$) dividen el espacio de búsqueda en sub-espacios con funciones de naturaleza diferente; y las de composición ($F_9$–$F_{12}$) combinan múltiples funciones base con pesos gaussianos, creando paisajes de fitness altamente irregulares, multimodales y con engaños de gradiente.

El error de aproximación al óptimo global se reporta como $\epsilon_i = f(\mathbf{x}_{best}) - F_i^*$ para cada función $i$, donde un valor $\epsilon_i \approx 0$ indica convergencia al óptimo conocido.

---

## 2.7. Estudio de Caso en Ingeniería Real: Sistema Energético Renovable Híbrido con Almacenamiento de Hidrógeno (HRES2-H2)

Como demostración de la capacidad del framework para resolver problemas de ingeniería de alto impacto tecnológico y complejidad físico-económica, se aplica a la **optimización dimensional y de despacho** de un sistema de energía renovable híbrida con generación y almacenamiento de hidrógeno (HRES2-H2). El caso de estudio extiende el modelo WPEB (Wind-Photovoltaic-Electrolyzer-Battery) propuesto en [Li et al., 2024] para la ubicación de Baotou, Mongolia Interior, China ($41.70°\text{N}$, $110.43°\text{E}$), introduciendo cuatro extensiones metodológicas que transforman el espacio de búsqueda original de cuasi-continuo a mixto entero-continuo.

### 2.7.1. Descripción y Extensiones del Modelo WPEB

El sistema HRES2-H2 integra cinco tecnologías complementarias con una capacidad total de generación renovable fijada en $P_{total} = 200$ MW (limitación de parcela):

$$P_{WT} + P_{PV} = P_{total} = 200 \text{ MW}$$

Las cuatro variables de decisión del problema de optimización son:

$$\mathbf{x} = \left[x_1,\ x_2,\ x_3,\ x_4\right] = \left[P_{WT},\ N_{el},\ P_{bat},\ \tau_{bat}\right]$$

**Tabla 3.** Variables de decisión y sus dominios de búsqueda.

| Variable | Símbolo | Tipo | Rango / Conjunto |
|:---|:---:|:---:|:---|
| Capacidad eólica instalada | $P_{WT}$ | Continua | $[0,\ 200]$ MW |
| Capacidad fotovoltaica | $P_{PV}$ | Derivada | $200 - P_{WT}$ MW |
| N° de módulos de electrolizador | $N_{el}$ | Entera | $\{10, 11, \dots, 20\}$ unid. $\times 5$ MW |
| Potencia del banco de baterías | $P_{bat}$ | Discreta | $\{0, 5, 10, \dots, 50\}$ MW |
| Duración del almacenamiento | $\tau_{bat}$ | Discreta | $\{1.0,\ 2.0,\ 4.0\}$ h |

La extensión del modelo original introduce la discretización de $N_{el}$ (10 a 20 módulos de 5 MW cada uno, resultando en capacidades de electrolizador de 50 a 100 MW), la independencia de la potencia de batería respecto a la capacidad del electrolizador (en el modelo original $P_{bat}$ estaba fijada como el 30\% de $P_{el}$), y la elección entre tres duraciones de almacenamiento.

### 2.7.2. Modelos Físicos de Generación Renovable

#### Modelo Eólico

La potencia horaria del parque eólico $P_{WT}(t)$ se calcula en función de la velocidad del viento a 50 metros de altura $v_{50m}(t)$ mediante la curva de potencia cúbica con zona de operación a potencia nominal y umbrales de corte:

$$P_{WT}(t) = \begin{cases}
0, & v(t) < v_{in} \text{ o } v(t) \geq v_{out} \\[4pt]
P_{rated} \cdot \dfrac{v(t)^3 - v_{in}^3}{v_{rated}^3 - v_{in}^3}, & v_{in} \leq v(t) < v_{rated} \\[6pt]
P_{rated}, & v_{rated} \leq v(t) < v_{out}
\end{cases}$$

con $v_{in} = 2.5$ m/s (velocidad de arranque), $v_{rated} = 10.5$ m/s (velocidad nominal), $v_{out} = 25.0$ m/s (velocidad de corte) y $P_{rated} = 5.0$ MW por turbina. Los perfiles de velocidad de viento horarios se obtienen de la base de datos climática NASA POWER para el año de referencia 2008 (seleccionado como Año Meteorológico Típico).

#### Modelo Fotovoltaico

La potencia del arreglo fotovoltaico $P_{PV}(t)$ se determina a partir de la irradiancia global horizontal $G(t)$ [kWh/m²] y la temperatura ambiente $T_a(t)$ [°C], incorporando el efecto térmico sobre la eficiencia de las celdas mediante la temperatura normal de operación de celda (NOCT):

$$T_c(t) = T_a(t) + \frac{G(t)}{800} \cdot (NOCT - 20)$$

$$P_{PV}(t) = P_{PV,STC} \cdot \frac{G(t)}{1000} \cdot \bigl[1 + \gamma_T \cdot (T_c(t) - 25)\bigr] \cdot f_{dera}$$

donde $NOCT = 47°$C, $\gamma_T = -0.45\%/°$C es el coeficiente de temperatura, $f_{dera} = 0.90$ es el factor de pérdidas globales (suciedad, cableado, sombras parciales), y $P_{PV,STC}$ es la potencia instalada bajo condiciones estándar de prueba (STC). Los perfiles de irradiancia y temperatura se vectorizan para las 8,760 horas del año mediante precalculación de perfiles unitarios normalizados, reduciendo el costo computacional de cada evaluación de fitness.

### 2.7.3. Lógica de Despacho Horario (8,760 horas)

La simulación física del sistema se realiza hora a hora para el año de referencia completo. En cada hora $t \in \{1, 2, \dots, 8760\}$, se computa la potencia neta disponible:

$$P_{net}(t) = P_{WT}(t) + P_{PV}(t) - P_{load}(t)$$

y se aplica la siguiente **jerarquía de prioridad de despacho**:

**Caso Superávit** ($P_{net}(t) > 0$): La energía excedente se asigna siguiendo las prioridades:
1. Alimentar el electrolizador PEM si $P_{gen}(t) \geq P_{el,min} = 0.30 \cdot P_{el}$ (condición de carga mínima del 30\%).
2. Cargar el banco de baterías con la potencia sobrante hasta $SOC_{max} = P_{bat} \cdot \tau_{bat}$ MWh.
3. Inyectar el remanente a la red eléctrica externa (contabilizado como $P_{grid\_sales}(t)$).

**Caso Déficit** ($P_{net}(t) < 0$): El déficit energético se cubre mediante:
1. Descarga del banco de baterías con rendimiento $\eta_{dis} = \sqrt{\eta_{RT}}$.
2. Si la batería no es suficiente, usar energía de la celda de combustible ($H_2$ comprimido).
3. Importar energía desde la red eléctrica externa (contabilizado como $P_{grid\_import}(t)$).

La eficiencia del ciclo completo de carga-descarga de baterías es $\eta_{RT} = 0.90$ (rendimiento de ida y vuelta), con $\eta_{ch} = \eta_{dis} = \sqrt{0.90} \approx 0.9487$ para carga y descarga respectivamente. El electrolizador convierte electricidad en hidrógeno con una eficiencia de $\eta_{el} = 0.75$, y el poder calorífico superior (HHV) del hidrógeno es $HHV_{H_2} = 39.4$ kWh/kg.

La producción total anual de hidrógeno en kilogramos se calcula como:

$$m_{H_2} = \frac{E_{el,anual} \times 1000 \times \eta_{el}}{HHV_{H_2}} \quad [\text{kg/año}]$$

### 2.7.4. Funciones Objetivo Financieras y Restricción AGSR

La función objetivo del problema de optimización busca minimizar el **Costo Nivelado de Energía (LCOE)** en CNY/kWh durante la vida útil del proyecto de $N_{proj} = 25$ años:

$$\min_{\mathbf{x}}\quad \text{LCOE} = \frac{NPC \cdot CRF(r, N_{proj})}{E_{servida,anual}}$$

donde el **Costo Presente Neto (NPC)** se descompone como la suma de los costos de inversión inicial (CAPEX), operación y mantenimiento (O\&M) y reemplazos de componentes de ciclo de vida corto, actualizados a valor presente para cada tecnología $k \in \{WT, PV, EL, BAT\}$:

$$NPC = \sum_{k} \left[ CAPEX_k + \sum_{y \in \mathcal{Y}_{rep,k}} \frac{REP_k}{(1+r)^y} + \sum_{y=1}^{N_{proj}} \frac{O\&M_k}{(1+r)^y} \right]$$

La **Tasa de Recuperación de Capital (CRF)** anualiza el NPC a lo largo de la vida del proyecto con tasa de descuento real $r = 4.35\%$:

$$CRF(r, N) = \frac{r(1+r)^N}{(1+r)^N - 1}$$

**Tabla 4.** Parámetros económicos de referencia (adaptados de Li et al., 2024).

| Tecnología | CAPEX [CNY/kW] | O\&M [CNY/kW·año] | Reemplazo [CNY/kW] | Vida útil [años] |
|:---|:---:|:---:|:---:|:---:|
| Eólica (WT) | 5,917 | 40.2 | — | 25 |
| Fotovoltaica (PV) | 4,633 | 17.6 | — | 25 |
| Electrolizador PEM | 6,964 | 208.9 | 5,969 | 15 |
| Batería BESS | 2,549 | 10.0 | 500 | 10 |

**Restricción de Seguridad de Suministro de Red (AGSR):** Para garantizar la autonomía energética mínima del sistema y evitar una dependencia excesiva de la red eléctrica externa, el cociente anual entre la energía importada de la red y la demanda total atendida está acotado estrictamente por:

$$AGSR = \frac{\displaystyle\sum_{t=1}^{8760} P_{grid\_import}(t)}{\displaystyle\sum_{t=1}^{8760} P_{load}(t)} \leq 0.20$$

Las soluciones que violan esta restricción ($AGSR > 0.20$) son catalogadas como **infactibles** y penalizadas con una función objetivo $f(\mathbf{x}) = 100 + 10 \cdot AGSR$, garantizando que el optimizador evite estas regiones del espacio de búsqueda.

El **LCOH (Costo Nivelado del Hidrógeno)** en CNY/kg se calcula complementariamente como:

$$LCOH = \frac{NPC \cdot CRF(r, N_{proj})}{m_{H_2,anual}}$$

---

## 2.8. Protocolo de Evaluación y Validación Estadística No Paramétrica

Dado que la naturaleza estocástica intrínseca de las metaheurísticas —derivada de la inicialización aleatoria de la población, los operadores probabilísticos y la selección de soluciones— produce distribuciones de resultados cuya gaussianidad no puede asumirse a priori, la evaluación experimental adopta un **protocolo de inferencia estadística no paramétrica** siguiendo las directrices de [Derrac et al., 2011; García et al., 2010].

Para cada algoritmo evaluado sobre cada instancia o función de prueba, se realizan **$N = 31$ ejecuciones estadísticamente independientes** con semillas pseudo-aleatorias distintas (semilla base: 42) para garantizar la reproducibilidad y la estimación fiable de la variabilidad del desempeño.

### Protocolo de Pruebas Estadísticas

1. **Prueba de Normalidad de Shapiro-Wilk:** Se evalúa la hipótesis nula $H_0$ de que la distribución de los $N = 31$ valores finales sigue una distribución normal. Si el $p$-valor resultante satisface $p < 0.05$, se rechaza $H_0$ y se confirma la necesidad de emplear pruebas no paramétricas.

2. **Prueba de Rangos con Signo de Wilcoxon (Comparaciones Binarias):** Para cada par de algoritmos (framework propuesto vs. algoritmo competidor $k$), se realiza la prueba emparejada de Wilcoxon con nivel de significancia $\alpha = 0.05$. Los resultados se clasifican como:
   - $p < 0.05$: diferencia estadísticamente significativa (el framework propuesto es superior o inferior con confianza del 95\%).
   - $p \geq 0.05$: ausencia de diferencia estadísticamente significativa.

3. **Prueba de Ranking Global de Friedman:** Para obtener un ranking de dominancia estadística global entre todos los algoritmos evaluados simultáneamente sobre el conjunto completo de instancias o funciones, se calcula el estadístico de Friedman:
$$\chi_F^2 = \frac{12}{N_p \cdot K(K+1)} \sum_{j=1}^{K} \bar{R}_j^2 - 3 N_p (K+1)$$
donde $N_p$ es el número de instancias/funciones de prueba, $K$ es el número de algoritmos comparados y $\bar{R}_j$ es el rango promedio del algoritmo $j$. Un valor $p < 0.05$ en la prueba global confirma que al menos un par de algoritmos presenta diferencias estadísticamente significativas en su desempeño.

---

## 2.9. Configuración del Entorno Computacional

Todos los experimentos se ejecutaron sobre una plataforma de cómputo con Python 3.11.x, empleando NumPy para las operaciones vectorizadas de alta velocidad y, opcionalmente, la compilación JIT de Numba para el bucle de despacho horario del modelo HRES2-H2 (`_fast_dispatch_simulation`). La semilla pseudo-aleatoria global de NumPy y del módulo `random` de Python se fija al valor 42 al inicio de cada ejecución independiente para garantizar reproducibilidad completa. El código fuente completo del framework, incluyendo todos los módulos de metaheurísticas, el monitor DTW/DDTW y los scripts de benchmark, está disponible públicamente en el repositorio del proyecto.
