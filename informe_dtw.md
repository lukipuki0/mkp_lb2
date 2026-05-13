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

### Equivalencia en el Orquestador (`rotating_benchmark.py`)
Estos parámetros se configuran como constantes globales en tu script principal:

```python
# DTW Stagnation params
STAG_WINDOW      = 30       # Tamaño de la ventana deslizante (últimas N iteraciones evaluadas)
STAG_BAND        = 0        # Banda Sakoe-Chiba para DTW. 0 = auto (10% de la ventana)
STAG_MIN_SLOPE   = 0.0      # Pendiente de la rampa ideal. 0.0 = auto (1% del progreso en la ventana)
STAG_PLATEAU_MAX = 15       # Iteraciones máximas permitidas sin mejora absoluta (fitness plano)
STAG_PATIENCE    = 3        # Alarmas consecutivas requeridas para confirmar el estancamiento (evita falsos positivos)
STAG_USE_DDTW    = False    # Usar derivadas (DDTW) en vez de valores absolutos (DTW)
STAG_ADAPT       = True     # Si es True, adapta los umbrales dinámicamente usando el historial
STAG_P_LOW       = 30.0     # Percentil bajo para umbral de línea plana (qué tan estricto es D2)
STAG_P_HIGH      = 70.0     # Percentil alto para umbral de rampa/delta (qué tan estricto es D1)
```

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

### Variables de Salida (Los indicios o señales de estancamiento)

Al finalizar cada actualización, el monitor devuelve un diccionario con **variables clave** que sirven como señales e indicios del estado del algoritmo:

- **`D1_vs_ramp` (Distancia al escenario ideal):** Distancia DTW entre cómo va tu metaheurística y una línea que sube constantemente. Si este valor es **muy grande**, significa que tu algoritmo se está alejando del comportamiento ideal y ya no mejora a buen ritmo.
- **`D2_vs_const` (Distancia al estancamiento total):** Distancia DTW a una línea completamente plana. Si es **muy pequeño** (cercano a 0), la curva se parece muchísimo a una línea plana. Es la señal más fuerte de estar atascado en una meseta (plateau).
- **`delta` (La diferencia):** Es la resta `D1 - D2`. Si es positivo y alto, indica que la curva tira más hacia la línea plana que a la rampa de progreso. Es una confirmación matemática del estancamiento.
- **`theta_c`, `theta_r`, `theta_delta` (Los Umbrales):** Son los límites calculados dinámicamente utilizando el historial (percentiles). Le indican al código: *"Si D2 es menor que theta_c y D1 es mayor que theta_r, enciende la alarma"*.
- **`fire` (La señal definitiva):** Es el resultado final (`True` o `False`). Se activa si el algoritmo cumple de manera sostenida las condiciones de estancamiento definidas por los umbrales durante las iteraciones indicadas en la "paciencia".

---

## 5. Ejemplos Prácticos y Detalles Matemáticos

### Ejemplo 1: ¿Por qué usar DDTW en lugar de DTW Normal?
Imagina que comparamos el historial de fitness de tu algoritmo contra una curva de referencia. Creamos dos curvas que tienen **exactamente la misma forma**, pero empiezan en números distintos.

*   **Curva A (Tu algoritmo):** `[10, 20, 30, 20]` *(Empieza en 10)*
*   **Curva B (Referencia):** `[100, 110, 120, 110]` *(Empieza en 100)*

**Con DTW Normal (Valores absolutos):**
El DTW normal mide la distancia entre los números tal cual. La distancia entre 10 y 100 es 90. Al sumar el error de todos los puntos, la distancia total es **360**. El DTW concluye erróneamente que las curvas son completamente diferentes por estar a distintas "alturas".

**Con DDTW (Derivadas/Diferencias):**
El DDTW ignora la altura y mira los saltos:
*   Saltos Curva A: `[+10, +10, -10]`
*   Saltos Curva B: `[+10, +10, -10]`
El DDTW compara los saltos y encuentra que la distancia es **0**. Se da cuenta de que las tendencias son idénticas. Esto es clave en MKP: nos importa si la tendencia es "plana" (estancamiento), sin importar si el fitness está trabado en 5,000 o 150,000.

### Ejemplo 2: El concepto de la "Ventana Deslizante" y las Líneas Base
Las líneas base (la rampa ideal y la línea plana) **no tienen memoria** desde la iteración 0. Se "destruyen" y se vuelven a dibujar desde cero en cada iteración, utilizando exclusivamente los datos de la ventana actual.

Si llevas 10 iteraciones y tu ventana (`W`) es de 5:
*   Historial completo: `[10, 20, 30, 40, 50, 60, 70, 80, 80, 80]`
*   **Ventana (X):** Recorta los últimos 5 datos: `[60, 70, 80, 80, 80]`
*   **Punto de inicio (`start_value`):** Es el primer dato de la ventana, es decir, `60`.

Con este punto de inicio, se dibujan las dos líneas contra las que competirá tu ventana `X`:
1.  **La nueva Línea Plana:** Repite el 60 cinco veces `[60, 60, 60, 60, 60]`.
2.  **La nueva Rampa Ideal:** Empieza en 60 y le suma la pendiente calculada en cada paso.

En la iteración 11, la ventana avanza un paso (ej. `[70, 80, 80, 80, 80]`), el nuevo inicio es `70`, y se vuelven a dibujar una nueva línea plana de 70s y una nueva rampa. Así, el monitor evalúa el estancamiento siempre en el "aquí y ahora".

### Ejemplo 3: Cálculo Automático de la Pendiente (`s_min`)
En el código, la pendiente de la Rampa Ideal se calcula automáticamente en cada iteración si `min_slope = 0.0`, usando la fórmula: `s_min = 0.01 * rng / W`

Usando la ventana anterior (`X = [60, 70, 80, 80, 80]` con `W = 5`):
1.  **Rango (`rng`):** Distancia entre el último y primer valor de la ventana (`abs(80 - 60) = 20`).
2.  **Fórmula:** `s_min = 0.01 * 20 / 5 = 0.04`.

La **Rampa Ideal** se construye como `y = start_value + (s_min * iteración)`:
*   Paso 0: 60 + 0 = `60.00`
*   Paso 1: 60 + 0.04 = `60.04`
*   Paso 2: 60 + 0.08 = `60.08`
*   Paso 3: 60 + 0.12 = `60.12`
*   Paso 4: 60 + 0.16 = `60.16`
Arreglo Final Rampa: `[60.00, 60.04, 60.08, 60.12, 60.16]`

**¿El multiplicador `0.01` cambia?**
No, es una constante ("hardcodeada"). Representa una regla del **1%** del rango. El `s_min` total sí cambia en cada iteración porque el rango (`rng`) cambia conforme la ventana avanza. Además, el código usa `max(1.0, rng)` para asegurar que el rango jamás sea 0. Si la ventana se llena de ochos (`[80, 80, 80, 80, 80]`), el rango forzado será `1.0`, generando una pendiente de `0.002`. Esto evita que la rampa sea completamente plana; siempre exigirá un mínimo avance.
Si se desea cambiar esta regla del 1%, se debe desactivar el cálculo automático asignando un valor fijo en la configuración (ej. `min_slope = 0.5`).
