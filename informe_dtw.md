# Informe: Monitor de Estancamiento basado en DTW

Este informe explica en detalle cada componente, función y parámetro del archivo `dtw_stagnation.py`, cuyo propósito principal es detectar cuándo un algoritmo de optimización (como Simulated Annealing para el problema MKP) ha dejado de mejorar y se encuentra "estancado".

Para lograr esto, utiliza **DTW (Dynamic Time Warping)**, una técnica matemática que mide la similitud entre dos secuencias temporales que pueden variar en velocidad.

---

## 1. Funciones Centrales (DTW)

### `dtw_distance(s, t, window=None)`
Calcula la distancia DTW entre dos series numéricas `s` y `t`.
- **¿Qué hace?**: Encuentra el "alineamiento óptimo" entre dos secuencias para calcular su distancia (qué tan diferentes son). En lugar de comparar punto a punto en el mismo instante (como la distancia Euclidiana), DTW permite "deformar" el tiempo, asociando un punto de una serie con múltiples puntos de la otra si eso minimiza el error total.
- **Parámetros**:
  - `s` y `t`: Las dos series de tiempo (arreglos numéricos) que se quieren comparar.
  - `window`: Un número entero opcional. Representa la **banda de Sakoe-Chiba**. Sirve para limitar cuánto se puede "doblar" el tiempo. Si `window` es pequeño, la comparación es más estricta (casi lineal).
- **Retorno**: Un número (`float`) que representa la "distancia" o "costo" entre las dos secuencias. **A menor distancia, más similares son.**

### `_first_diff(x)`
Calcula la primera derivada discreta (la diferencia entre puntos consecutivos de un arreglo).
- **¿Qué hace?**: En lugar de analizar los valores absolutos, obtiene cuánto cambió la serie en cada paso. Por ejemplo, si la serie es `[2, 4, 7]`, la diferencia es `[2, 2, 3]` (el primer elemento se copia para mantener el tamaño).

### `ddtw_distance(s, t, window=None)`
Es la versión de la Derivada de DTW (Derivative Dynamic Time Warping).
- **¿Qué hace?**: Llama a `dtw_distance` pero pasándole `_first_diff(s)` y `_first_diff(t)`.
- **¿Por qué es útil?**: A veces, dos series pueden tener "formas" (tendencias) idénticas pero estar en alturas (valores en Y) muy diferentes. Analizar la derivada ayuda a enfocarse puramente en "cuánto suben o bajan", ignorando el valor absoluto de la serie.

---

## 2. Líneas Base (Baselines)

El script funciona comparando el historial real de mejoras de tu algoritmo contra dos **escenarios ideales**:

### `ramp_baseline(start_value, s_min, length)`
- **¿Qué hace?**: Crea una línea recta ascendente. Representa el escenario de un algoritmo que **sigue progresando y mejorando de manera constante**.
- **Parámetros**:
  - `start_value`: El valor en Y donde comienza.
  - `s_min`: La pendiente (cuánto crece en cada paso).
  - `length`: El tamaño de la serie.

### `constant_baseline(start_value, length)`
- **¿Qué hace?**: Crea una línea plana horizontal. Representa el escenario de un algoritmo que **se ha estancado por completo** y no mejora.

### `moving_percentile(buffer, p)`
- **¿Qué hace?**: Calcula el percentil `p` de una lista de valores. Se usa para adaptar dinámicamente el comportamiento del algoritmo basándose en su propio historial.

---

## 3. Configuración (`StagnationConfig`)

Es un contenedor de hiperparámetros (las "perillas" de ajuste):

- `window` (int, default=30): El tamaño de la "ventana" de tiempo. Cuántos de los últimos valores observamos para tomar la decisión.
- `band` (int, default=0): Tamaño de la banda de Sakoe-Chiba. Si se deja en 0, se auto-asigna al 10% de la ventana.
- `min_slope` (float, default=0.0): La pendiente de nuestra "rampa" ideal. Si es 0.0, calcula automáticamente un 1% del rango de los datos evaluados.
- `plateau_max` (int, default=15): Iteraciones máximas permitidas sin mejora absoluta de la aptitud (fitness) antes de empezar a considerarlo una meseta.
- `patience` (int, default=3): Épocas consecutivas que el algoritmo debe dictaminar "estancamiento" antes de disparar la alarma real. Evita "falsos positivos".
- `use_ddtw` (bool, default=False): Si está activado, analiza las pendientes (derivada) en vez de los valores originales.
- `adapt_thresholds` (bool, default=True): Si es True, los umbrales de detección no son fijos, sino que se adaptan usando percentiles históricos.
- `p_low` / `p_high` (float): Percentiles usados para definir los umbrales. (30% y 70% por defecto).

---

## 4. El Motor Central: `StagnationMonitor`

Esta clase mantiene la memoria del progreso del algoritmo en el tiempo, y dictamina si está estancado o no a través del método `update()`.

### Variables que memoriza:
- `best_so_far`: Toda la historia de los mejores valores encontrados.
- `no_improve_len`: Contador numérico de cuántas rondas seguidas llevamos sin que el valor global supere su propio récord.
- `trigger_streak`: Contador de racha de las "alarmas de estancamiento" internas.

### ¿Cómo funciona internamente la función `update(new_best)`?

Se manda a llamar en cada ciclo pasándole el nuevo mejor valor descubierto (`new_best`). Su flujo es el siguiente:

1. **Acondicionamiento**: Guarda el nuevo valor. Si el nuevo valor no supera al último récord, el contador `no_improve_len` aumenta. Si todavía no tenemos suficientes datos para llenar una `window` (ej. menos de 30 datos), simplemente devuelve `{"ready": False}` y espera más iteraciones.
2. **Cálculo de Distancias**:
   - Toma los últimos `W` valores del historial (`X`).
   - Genera la Rampa ideal progresando (`r`).
   - Genera la Constante plana estancada (`c`).
   - Mide **D1**: Qué tan lejos está la serie real (`X`) de la Rampa progresiva.
   - Mide **D2**: Qué tan lejos está la serie real (`X`) de la Constante plana.
   - Calcula **Delta**: La resta `D1 - D2`. Un Delta positivo significa que la serie se parece MÁS a la constante plana que a la rampa de progreso.
3. **Cálculo de Umbrales**:
   Si está en modo adaptativo, se calculan tres topes (`theta_c`, `theta_r`, `theta_delta`) basados en la mediana histórica de qué tanto suele alejarse el algoritmo en esta ejecución.
4. **Condiciones (El Jurado)**:
   Se hacen 3 preguntas matemáticas clave:
   - **`cond_plateau`**: ¿Llevamos estancados un tiempo prudencial? (`no_improve_len >= plateau_max`).
   - **`cond_constant`**: ¿La distancia D2 es tan pequeña que somos prácticamente una línea plana? (`D2 <= theta_c`).
   - **`cond_ramp`**: ¿La distancia D1 es lo suficientemente grande como para descartar que estemos subiendo por la rampa? (`D1 >= theta_r` o el delta es muy alto).
   
   **Si las 3 son ciertas al mismo tiempo**, se suma 1 al contador de alarmas internas (`trigger_streak`).
5. **El Disparo final (`fire`)**:
   Si el contador de alarmas supera la "paciencia" definida (ej. `patience >= 3`), la variable `fire` se activa en `True`. Esto se devuelve al usuario para que sepa que debe reiniciar su metaheurística, aumentar la temperatura, hacer una perturbación de las soluciones, o cualquier técnica de evasión de óptimos locales.
