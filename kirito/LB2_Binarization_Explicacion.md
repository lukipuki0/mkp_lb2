# Binarización en el Framework LB2 para MKP con PSO

Este documento detalla el funcionamiento interno de la binarización propuesta por el framework LB2 cuando se aplica al algoritmo Particle Swarm Optimization (PSO) para resolver el Problema de la Mochila Multidimensional (MKP).

## 1. El Problema Base: Continuo vs Binario

El algoritmo PSO clásico fue diseñado para optimización en espacios continuos, donde las posiciones y velocidades son números reales. Sin embargo, el problema MKP es estrictamente binario (se selecciona un objeto o no). 

La pregunta fundamental es: **¿Cómo mapear una velocidad continua a una decisión binaria efectiva?**

### El Enfoque Clásico (BPSO - Kennedy & Eberhart, 1997)

En el BPSO clásico, el enfoque es directo pero limitado:
1. Se toma la velocidad continua de la partícula.
2. Se pasa por una **función sigmoide** $S(v) = 1 / (1 + e^{-v})$ para convertirla en una probabilidad $[0, 1]$.
3. Se genera un número aleatorio; si es menor a la probabilidad, la dimensión correspondiente de la nueva solución es 1, de lo contrario es 0.

**Desventaja:** Este método **descarta por completo la posición actual** de la partícula y genera una nueva solución desde cero en cada iteración. Además, la función de transferencia (sigmoide) es **estática** y única.

---

## 2. La Innovación del LB2: Funciones de Transferencia Dinámicas (L1 y L2)

El framework LB2 aborda las deficiencias del BPSO clásico introduciendo dos conceptos fundamentales: **probabilidades de flip** en lugar de reemplazo directo, y el uso de **dos funciones de transferencia dinámicas**.

### A. "Flips" en lugar de reemplazo desde cero

A diferencia del BPSO clásico que construye la solución de la nada, LB2 toma la solución actual (que ya tiene un grado de bondad/fitness) y evalúa la **probabilidad de invertir (flip)** cada bit.

### B. Las dos funciones: L1 (Conservadora) y L2 (Agresiva)

LB2 genera **dos soluciones candidatas** en cada iteración por partícula, utilizando dos funciones lineales de probabilidad de flip:

*   **Función L1 (Pendiente negativa):** Genera una probabilidad de flip que disminuye a medida que la magnitud de la velocidad "sugiere" el cambio. Esto la hace más **conservadora**, tendiendo a mantener el estado actual de los bits incluso frente a velocidades altas. 
*   **Función L2 (Pendiente positiva):** Genera una probabilidad de flip proporcional a la "fuerza" de la velocidad. Es más **agresiva**, favoreciendo cambios cuando la velocidad así lo indica.

Para ambas candidatas, se aplica el operador de reparación (para garantizar la factibilidad frente a las restricciones del MKP), y la partícula solo se actualiza si la mejor de estas dos candidatas supera su fitness histórico (selección greedy).

---

## 3. Los Parámetros de Control: G1, G2 y G3

Las formas exactas de las funciones L1 y L2 están controladas por tres parámetros, que actúan como "perillas" de ajuste:

*   **`G1` (Pendiente):** Controla qué tan "empinadas" son las rectas.
    *   *Bajo:* Probabilidades uniformes independientemente de la velocidad. Fomenta la exploración.
    *   *Alto:* Respuesta muy sensible a la magnitud de la velocidad. Fomenta la explotación.
*   **`G2` (Compresión de señal):** Aparece como divisor ($(V_{max} - G2)$).
    *   *Bajo:* Atenúa la magnitud de las velocidades. Fomenta exploración.
    *   *Alto:* Amplifica las diferencias en las velocidades. Fomenta explotación selectiva.
*   **`G3` (Offset/Piso):** Desplaza ambas funciones verticalmente.
    *   *Alto (ej. 0.5):* Establece un "piso" alto de probabilidad de flip base (cercano a lanzar una moneda) sin importar la velocidad. Extrema exploración.
    *   *Bajo (0.0):* Elimina el piso aleatorio; los flips dependen únicamente de la directriz de la velocidad. Explotación pura.

---

## 4. Evolución Temporal: Exploración hacia Explotación

La verdadera potencia del LB2 original radica en cómo estos parámetros cambian a lo largo de las iteraciones. Generalmente se interpolan linealmente desde valores iniciales (`_i`) a valores finales (`_f`):

| Iteración | Fase | $G1$ | $G2$ | $G3$ | Comportamiento Resultante |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Iniciales** | Exploración | Bajo (0.5) | Bajo (0.5) | Alto (0.5) | Curvas L1 y L2 planas con offset alto. Altas probabilidades base de *flip*. La búsqueda es cuasi-aleatoria, priorizando saltos grandes en el espacio de búsqueda. |
| **Finales** | Explotación | Alto (1.0) | Alto (ej. 7.2) | Bajo (0.0) | Curvas empinadas sin offset. Los bits solo se invierten si la velocidad indica una señal direccional fuerte. Búsqueda muy refinada alrededor de óptimos locales. |

---

## 5. Nuestro Contexto Actual: Integración con DTW (Dynamic Time Warping)

En el planteamiento original del LB2, la transición de los parámetros G1, G2 y G3 es **lineal y ciega**. Asume que la convergencia ocurrirá gradualmente a lo largo del total de iteraciones predefinidas.

**El objetivo de la autoadaptación que estamos desarrollando es reemplazar esa transición ciega.**

Utilizamos el monitor de estancamiento basado en **DTW** para evaluar la forma real de la curva de convergencia:

1.  **Detección de Delta ($\Delta = D_1 - D_2$):** Medimos si el comportamiento reciente se parece más a un progreso continuo ($D_1$ bajo, $\Delta$ negativo/bajo) o a una meseta plana ($D_2$ bajo, $\Delta$ alto).
2.  **Adaptación Continua:** En lugar de avanzar ciegamente los parámetros $G$ con la iteración $t$, mapeamos el valor del $\Delta$ a un factor de interpolación $\alpha \in [0, 1]$.
    *   Si $\Delta$ sube (estancamiento): $\alpha$ se empuja hacia 0 (forzando valores de $G$ exploratorios).
    *   Si $\Delta$ baja (progreso): $\alpha$ se acerca a 1 (permitiendo valores de $G$ de explotación).

Esta es la fusión del poder de búsqueda guiada binaria selectiva del LB2 con el análisis de forma de serie temporal del DTW.
