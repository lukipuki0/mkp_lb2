# Dynamic Time Warping (DTW) — Fundamentos, Algoritmo y Parámetros

> **Referencia base**: Sakoe & Chiba (1978) · Keogh & Pazzani (2001)  
> **Uso en este proyecto**: Sensor de similitud de formas para curvas de convergencia

---

## 1. El Problema que Resuelve el DTW

### 1.1 Comparar secuencias no es trivial

Supongamos dos secuencias temporales:

```
A: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
B: [1, 1, 2, 2, 3, 3, 4, 4, 5,  5]
```

Si las comparamos punto a punto (distancia euclidiana), la diferencia es enorme: A crece consistentemente, B crece a trompicones. Pero **visualmente** son similares: ambas son crecientes, solo que B "se detiene" a veces. La distancia euclidiana castiga este desfase temporal como si fuera una diferencia estructural.

El **Dynamic Time Warping** resuelve exactamente esto: compara dos secuencias **alineándolas de forma no lineal**, permitiendo que un punto de una secuencia se compare con múltiples puntos de la otra. Esto captura la **forma** de las curvas, no solo sus valores en instantes coincidentes.

### 1.2 La intuición geométrica

Imaginá dos firmas manuscritas de la misma palabra. Una fue escrita más rápido que la otra. Punto a punto, no coinciden en ninguna coordenada. Pero si "estirás" o "comprimís" el eje temporal de una para alinearla con la otra, las formas coinciden. Eso es DTW: encuentra la **deformación temporal óptima** que minimiza la distancia entre dos secuencias.

```
Secuencia A:  /‾‾‾‾\___/‾‾‾\      (sube, meseta, baja, sube)
Secuencia B:  /‾‾\______/‾‾‾\     (sube rápido, meseta larga, sube)

Distancia euclidiana: ENORME (no coinciden en ningún punto)
Distancia DTW:        PEQUEÑA (la forma es similar, solo cambia el timing)
```

---

## 2. El Algoritmo: Programación Dinámica

### 2.1 La matriz de acumulación

DTW construye una matriz `D` de tamaño `(n+1) × (m+1)` donde `n` y `m` son las longitudes de las dos secuencias. Cada celda `D[i, j]` representa el **costo mínimo acumulado** de alinear el prefijo `A[0..i-1]` con el prefijo `B[0..j-1]`.

La recursión fundamental es:

```
D[i, j] = |A[i-1] − B[j-1]| + min( D[i-1, j  ],    ← inserción (estirar A)
                                    D[i  , j-1],    ← eliminación (comprimir A)
                                    D[i-1, j-1] )   ← correspondencia directa
```

Cada paso elige la operación más barata entre:
- **Estirar** A: el punto `A[i-1]` se alinea con múltiples puntos de B
- **Comprimir** A: un punto de B se alinea con múltiples puntos de A
- **Correspondencia directa**: alineación uno a uno

La distancia DTW final es `D[n, m]`: el costo mínimo de alinear las dos secuencias completas.

### 2.2 ¿Por qué programación dinámica?

La cantidad de alineamientos posibles entre dos secuencias es **exponencial** en sus longitudes. La programación dinámica resuelve este problema en **O(n·m)** aprovechando la subestructura óptima: el mejor alineamiento de las secuencias completas se construye a partir de los mejores alineamientos de sus prefijos.

### 2.3 Las restricciones del alineamiento

Para que el resultado sea significativo, el alineamiento debe cumplir:

1. **Monotonicidad**: el tiempo nunca retrocede. Si `A[i]` se alinea con `B[j]`, el siguiente punto `A[i+1]` solo puede alinearse con `B[j]`, `B[j+1]`, `B[j+2]`, etc. — nunca con un punto anterior de B.

2. **Continuidad**: no se pueden saltar puntos. El alineamiento avanza de a un paso por vez (diagonal, horizontal o vertical en la matriz).

3. **Condiciones de frontera**: el primer punto de A se alinea con el primero de B, y el último con el último.

Estas restricciones garantizan que el alineamiento preserve el orden temporal, una propiedad esencial para comparar secuencias que representan evolución en el tiempo.

---

## 3. La Banda Sakoe-Chiba: Reducción de Costo y Ruido

### 3.1 El problema del DTW completo

El DTW sin restricciones (`band = 0`) compara **todas las celdas** de la matriz, con costo O(n·m). Para secuencias largas, esto es costoso. Además, permitir alineamientos muy "estirados" (donde un solo punto de A se alinea con docenas de puntos de B) puede producir distancias engañosamente bajas: el algoritmo "fuerza" un alineamiento incluso entre secuencias fundamentalmente distintas.

### 3.2 La restricción de banda

La banda Sakoe-Chiba (Sakoe & Chiba, 1978) impone una **ventana de alineamiento**: el punto `A[i]` solo puede alinearse con puntos `B[j]` donde `|i − j| ≤ band`. Esto restringe los alineamientos válidos a una franja diagonal de ancho `2·band + 1` alrededor de la diagonal principal.

```
Matriz DTW sin banda (band=0):          Matriz con banda (band=2):
  B  B  B  B  B  B                       B  B  B  B  B  B
A ■  ■  ■  ■  ■  ■                    A ■  ■  ■  .  .  .
A ■  ■  ■  ■  ■  ■                    A ■  ■  ■  ■  .  .
A ■  ■  ■  ■  ■  ■                    A .  ■  ■  ■  ■  .
A ■  ■  ■  ■  ■  ■                    A .  .  ■  ■  ■  ■
A ■  ■  ■  ■  ■  ■                    A .  .  .  ■  ■  ■
A ■  ■  ■  ■  ■  ■                    A .  .  .  .  ■  ■

Todas las celdas evaluadas            Solo la banda diagonal evaluada
Costo: O(n·m)                          Costo: O(n·band)
```

### 3.3 Efecto en la calidad de la medición

| `band` | Comportamiento |
|---|---|
| **band = 1** | Alineamiento casi uno a uno. Cada punto de A se alinea con a lo sumo 3 puntos de B. Muy sensible a diferencias **locales** de forma. Captura estancamiento inmediato. |
| **band = 2** | Leve flexibilidad. Permite pequeños desfases temporales. Buen balance entre sensibilidad local y tolerancia al ruido. |
| **band = 5** | Flexibilidad moderada. El DTW puede absorber desfases más grandes. Más tolerante, menos reactivo. |
| **band = 0** (sin restricción) | DTW completo. Máxima flexibilidad pero máximo riesgo de **over-warping**: forzar correspondencias donde no las hay. Costo O(n²). |

**La intuición**: cuánto más angosta la banda, más "exigente" es el DTW. Con `band = 1`, incluso pequeñas irregularidades producen distancias altas. Con `band = 0`, el DTW encuentra la mínima distancia posible, potencialmente subestimando diferencias reales.

---

## 4. DDTW — Derivative Dynamic Time Warping

### 4.1 El problema de la escala

El DTW estándar compara **valores absolutos**. Si una secuencia varía entre 0 y 100 y otra entre 1000 y 50000, las distancias absolutas son incomparables. Esto es un problema para cualquier aplicación donde la magnitud de las secuencias cambia entre ejecuciones (como el fitness de distintas instancias MKP).

Keogh & Pazzani (2001) propusieron el **Derivative DTW (DDTW)**: en lugar de comparar los valores originales, compara sus **primeras derivadas** (diferencias entre puntos consecutivos).

```
DDTW(A, B) = DTW( diff(A), diff(B) )

donde diff(X) = [X[0], X[1]−X[0], X[2]−X[1], ..., X[n−1]−X[n−2]]
```

### 4.2 ¿Qué gana el DDTW?

La derivada captura la **forma local** de la curva —su pendiente— independientemente de la magnitud absoluta:

| Secuencia original | Derivada |
|---|---|
| `[10000, 10050, 10100, 10150]` | `[10000, +50, +50, +50]` |
| `[100, 150, 200, 250]` | `[100, +50, +50, +50]` |

Ambas secuencias tienen la misma derivada aunque sus magnitudes difieran en un factor de 100. El DDTW las consideraría **idénticas** porque su forma (pendiente constante) es la misma. El DTW estándar las consideraría **muy distintas** por la diferencia de magnitud.

### 4.3 Implicaciones para detección de estancamiento

Para detectar si una curva de fitness es "plana" (estancada) o "inclinada" (progreso), el DDTW es superior porque:

- Una **meseta** tiene derivada ≈ 0 en cualquier escala de fitness
- Una **rampa** tiene derivada positiva constante en cualquier escala
- El DDTW compara estas **formas** sin importar si el fitness es 10 o 10,000

Esto hace que el sistema sea **invariante a la escala del problema**: la misma configuración de DTW funciona para instancias MKP con n=100 (fitness ~5,000) y n=500 (fitness ~25,000).

### 4.4 La primera posición

El DDTW incluye el primer valor original como primer elemento de la derivada (`diff(X) = [X[0], X[1]−X[0], ...]`). Esto ancla la secuencia: sin este anclaje, dos secuencias idénticas en forma pero desplazadas verticalmente serían indistinguibles para el DDTW.

---

## 5. Parámetros del DTW

### 5.1 `window` — La ventana de observación

Define **cuántos puntos** de cada secuencia se comparan. En el contexto de una serie temporal que se construye incrementalmente (como el fitness iteración a iteración), `window` determina cuánta historia se considera.

| `window` | Efecto |
|---|---|
| **Pequeño (5-10)** | Muy reactivo. Captura cambios inmediatos. Ruidoso: fluctuaciones puntuales afectan la medición. |
| **Mediano (15-30)** | Balance entre reactividad y estabilidad. Suficiente historia para identificar tendencias sin ser demasiado lento. |
| **Grande (50+)** | Muy estable. Captura tendencias de largo plazo. Lento para reaccionar: necesita acumular mucha evidencia antes de que la métrica cambie. |

**El trade-off fundamental**: una ventana grande produce mediciones más suaves y confiables, pero introduce **latencia**: el sistema tarda `window` iteraciones en empezar a funcionar (período de warm-up) y reacciona con retraso a los cambios.

### 5.2 `band` — La restricción de alineamiento

Define el **ancho de la banda Sakoe-Chiba**: cuánto puede desviarse el alineamiento de la diagonal.

| `band` | Comportamiento del alineamiento | Costo computacional |
|---|---|---|
| **1** | Muy restrictivo. Alineamiento casi uno a uno. | O(n) |
| **2** | Leve flexibilidad. Permite desfases de ±2 posiciones. | O(2n) |
| **0** (auto) | Sin restricción. DTW completo. | O(n²) |

**¿Cuándo usar cada uno?** Una banda angosta es preferible cuando las secuencias son cortas y se espera que estén aproximadamente alineadas temporalmente (como iteraciones consecutivas de una MH). Una banda ancha es necesaria cuando las secuencias pueden tener desfases temporales grandes (como comparar señales de audio donde una palabra se pronunció más rápido que otra).

### 5.3 `min_slope` — La pendiente de la rampa de referencia

Define la **rampa ideal** contra la cual se compara la secuencia observada. La rampa es una línea recta que comienza en el primer valor de la ventana y crece con pendiente `min_slope` en cada paso.

| `min_slope` | Rampa resultante | Efecto en la distancia D₁ |
|---|---|---|
| **0.0 (auto)** | Se auto-calcula como 1% del rango ÷ window | Rampa muy suave. Casi cualquier mejora produce D₁ bajo. |
| **Bajo (~0.01-0.1)** | Rampa casi plana | Poco exigente. Hasta mejoras mínimas se consideran "progreso". Riesgo: falsos negativos (no detecta estancamiento real). |
| **Alto (~2.0+)** | Rampa muy empinada | Muy exigente. La secuencia debe mejorar mucho y rápido para que D₁ sea bajo. Riesgo: falsos positivos (ve estancamiento donde hay progreso lento). |

**Calibración**: el `min_slope` ideal depende de la tasa de mejora esperada de la secuencia que se está monitoreando. Debe calibrarse para que refleje lo que constituye "progreso satisfactorio" en el contexto específico.

### 5.4 `use_ddtw` — DTW estándar vs. derivativo

| `use_ddtw` | Algoritmo | Qué compara | Cuándo usarlo |
|---|---|---|---|
| **True** | DDTW (Keogh & Pazzani, 2001) | Derivadas (pendientes, forma local) | Cuando la magnitud absoluta varía entre ejecuciones o no es relevante. Comparación de **formas**. |
| **False** | DTW estándar (Sakoe & Chiba, 1978) | Valores absolutos | Cuando la magnitud es significativa (ej: "cuánto mejoró", no solo "si mejoró"). Comparación de **magnitudes**. |

---

## 6. Las Métricas que Produce el DTW

Cuando el DTW compara una secuencia observada contra **dos patrones de referencia** (una rampa y una meseta), produce tres métricas fundamentales:

### 6.1 D₁ — Distancia a la rampa (progreso ideal)

```
D₁ = DTW(secuencia_observada, rampa_ideal)
```

Mide **cuán lejos está la secuencia de representar progreso constante**. Un D₁ bajo significa que la secuencia se parece a una rampa: hay mejora sostenida. Un D₁ alto significa que la secuencia **no** se parece a progreso.

### 6.2 D₂ — Distancia a la meseta (estancamiento total)

```
D₂ = DTW(secuencia_observada, meseta)
```

Mide **cuán lejos está la secuencia de ser completamente plana**. Un D₂ bajo significa que la secuencia se parece a una meseta: no hay cambio. Un D₂ alto significa que hay actividad (mejora, empeoramiento, oscilación).

### 6.3 Δ (delta) — Balance entre progreso y estancamiento

```
Δ = D₁ − D₂
```

Sintetiza ambas distancias en un solo número con signo:
- **Δ < 0**: la secuencia está más cerca de la rampa que de la meseta → **progreso neto**
- **Δ ≈ 0**: la secuencia es equidistante → **zona ambigua**
- **Δ > 0**: la secuencia está más cerca de la meseta que de la rampa → **estancamiento neto**

Delta es más informativo que D₁ o D₂ por separado porque captura el **contraste** entre ambas fuerzas: no solo "¿es plano?" (D₂) sino "¿es más plano que progresivo?" (Δ).

---

## 7. Umbrales Adaptativos: Auto-Calibración por Percentiles

Un problema fundamental al usar DTW como sensor es que los valores de D₁, D₂ y Δ dependen de la **escala** de las secuencias, que varía entre ejecuciones, problemas y contextos. Un D₂ = 5 puede ser "muy bajo" en un contexto y "muy alto" en otro.

La solución es calcular **umbrales adaptativos** como percentiles móviles del historial de cada métrica:

| Umbral | Cálculo | Interpretación |
|---|---|---|
| **θ_c** | Percentil `p_low` del historial de D₂ | "¿Qué es anormalmente bajo para D₂ en esta ejecución?" |
| **θ_r** | Percentil `p_high` del historial de D₁ | "¿Qué es anormalmente alto para D₁ en esta ejecución?" |
| **θ_δ** | Percentil `p_high` del historial de Δ | "¿Qué es un balance anormalmente sesgado en esta ejecución?" |

### 7.1 ¿Por qué percentiles y no valores fijos?

Los percentiles se **auto-calibran** a la distribución real de cada métrica en cada ejecución. Si una metaheurística naturalmente produce D₂ entre 10 y 50, el percentil 30 podría ser ~18. Si otra produce D₂ entre 100 y 500, el percentil 30 sería ~180. En ambos casos, θ_c representa "el 30% más bajo del historial", que es una referencia **relativa** consistente.

### 7.2 Los parámetros `p_low` y `p_high`

| Parámetro | Rol | Valor típico |
|---|---|---|
| **`p_low`** | Percentil para el umbral **inferior** (θ_c). Define qué fracción del historial se considera "anormalmente bajo". | 30 |
| **`p_high`** | Percentil para los umbrales **superiores** (θ_r, θ_δ). Define qué fracción del historial se considera "anormalmente alto". | 70 |

- `p_low` más bajo → θ_c más estricto (solo D₂ muy extremos se consideran "meseta")
- `p_high` más alto → θ_r y θ_δ más permisivos (se necesita D₁ o Δ más extremos para considerarlos "anormales")

---

## 8. El Problema del Warm-Up

El DTW requiere al menos `window` puntos en la secuencia observada para producir una medición significativa. Durante este **período de warm-up**, las métricas D₁, D₂ y Δ no están disponibles o no son confiables.

Esto introduce una restricción de diseño: con `window = 20` y un presupuesto total de 50 iteraciones, el 40% del tiempo el sensor está "ciego". La elección de `window` debe balancear:
- **Calidad de la medición** (ventana más grande = mejor)
- **Cobertura temporal** (ventana más chica = más iteraciones activas)

---

## 9. Resumen: DTW como Herramienta de Análisis de Formas

El Dynamic Time Warping es fundamentalmente una **medida de similitud entre secuencias temporales** que:

1. **Alinea de forma no lineal**, permitiendo comparar formas independientemente de diferencias de velocidad o timing
2. **Preserva el orden temporal**, gracias a las restricciones de monotonicidad y continuidad
3. **Se puede restringir** con la banda Sakoe-Chiba para controlar el trade-off entre flexibilidad y especificidad
4. **Se puede hacer invariante a escala** mediante el DDTW, que compara derivadas en vez de valores absolutos
5. **Produce mediciones interpretables** (distancias) que pueden usarse como entradas para sistemas de decisión

Su aplicación como sensor de estancamiento se basa en una observación simple pero poderosa: **las curvas de convergencia de algoritmos iterativos son series temporales**, y las herramientas de análisis de series temporales pueden extraer información sobre su comportamiento que los contadores simples (como "iteraciones sin mejora") no capturan.

---

## 10. Referencias Clave

- **DTW original**: Sakoe, H., & Chiba, S. (1978). "Dynamic Programming Algorithm Optimization for Spoken Word Recognition." *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49.
- **Banda Sakoe-Chiba**: Sakoe, H., & Chiba, S. (1971). "A Dynamic Programming Approach to Continuous Speech Recognition." *Proceedings of the 7th International Congress on Acoustics*, 65-68.
- **DDTW**: Keogh, E.J., & Pazzani, M.J. (2001). "Derivative Dynamic Time Warping." *Proceedings of the 2001 SIAM International Conference on Data Mining*, 1-11.
- **Survey de DTW**: Müller, M. (2007). "Dynamic Time Warping." *Information Retrieval for Music and Motion*, Springer, 69-84.
- **DTW con banda**: Ratanamahatana, C.A., & Keogh, E. (2004). "Everything You Know About Dynamic Time Warping is Wrong." *3rd Workshop on Mining Temporal and Sequential Data*, ACM SIGKDD.
- **FastDTW**: Salvador, S., & Chan, P. (2007). "Toward Accurate Dynamic Time Warping in Linear Time and Space." *Intelligent Data Analysis*, 11(5), 561-580.
- **Alineamiento temporal**: Rabiner, L., & Juang, B.H. (1993). *Fundamentals of Speech Recognition*. Prentice Hall. (Capítulos sobre DTW).
