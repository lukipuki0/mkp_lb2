# Binary-Simple — Estrategia de Adaptación por Umbral D₂

> **Clasificación**: Estrategia binaria A3 — Fire D2 puro  
> **Filosofía**: "La pregunta más directa posible: ¿está plana la curva?"

---

## 1. Posición en el Espacio de Diseño

Dentro de la familia de estrategias de adaptación basadas en DTW, Binary-Simple ocupa el extremo de **máxima simplicidad con señal informada**. Es más simple que la estrategia baseline de 3 condiciones (Binary-Complex, A4), pero más sofisticada que una decisión puramente basada en conteo de iteraciones sin mejora.

En el espectro de las estrategias booleanas definidas en el marco teórico del proyecto:

| Estrategia | Señal usada | Complejidad |
|---|---|---|
| A1 — Delta puro | `δ > 0` | Mínima |
| A2 — Delta + theta | `δ ≥ θ_δ` | Baja |
| **A3 — D2 puro** | **`D₂ ≤ θ_c`** | **Baja** |
| A4 — 3 condiciones + patience | D₂, D₁, δ, plateau, patience | Alta |
| A5 — Ratio normalizado + patience | `δ/θ_δ > 1` con patience | Media |

Binary-Simple implementa la estrategia **A3**, que se distingue por hacer exactamente una pregunta: **¿la curva de fitness se parece a una meseta?**

---

## 2. Fundamento Teórico

### 2.1 La señal D₂ como detector directo de estancamiento

El monitor DTW computa dos distancias fundamentales en cada iteración:

- **D₁**: distancia DTW entre la ventana de fitness y una **rampa ideal** (progreso constante). Un D₁ alto significa que la curva NO se parece a progreso.
- **D₂**: distancia DTW entre la ventana de fitness y una **meseta** (línea plana). Un D₂ bajo significa que la curva SÍ se parece a estancamiento.

Mientras que la estrategia baseline (A4) exige evidencia simultánea de tres fenómenos distintos (meseta prolongada + curva plana + ausencia de rampa), A3 se enfoca en la señal más fundamental: **D₂ bajo**.

La intuición es directa: si la curva de convergencia es indistinguible de una línea plana según la métrica DTW, la metaheurística está estancada, independientemente de cuánto tiempo lleve así o de qué tan lejos esté de una rampa ideal.

### 2.2 ¿Por qué D₂ y no delta?

La métrica `δ = D₁ − D₂` combina dos señales en una sola. Aunque es informativa, introduce ambigüedad: un delta positivo puede deberse a D₁ alto (lejos de la rampa) o a D₂ bajo (cerca de la meseta), o a una combinación de ambos. Esto requiere umbrales cuidadosamente calibrados para distinguir las causas.

D₂, en cambio, responde una sola pregunta sin ambigüedad: **¿cuán plana es la curva?**. Es una señal más pura, más directa, y requiere menos calibración.

### 2.3 El rol del DDTW (Derivative DTW)

Binary-Simple opera con **DDTW activado** (`use_ddtw=True`). Esto es crucial porque:

- El DTW estándar compara valores absolutos del fitness. Si el fitness salta de 10,000 a 50,000 entre ejecuciones (diferentes instancias MKP), las distancias absolutas no son comparables.
- El DDTW aplica DTW sobre la **primera derivada** de las secuencias. Compara pendientes, no magnitudes. Una meseta tiene derivada ≈ 0 en cualquier escala.
- Esto hace que D₂ sea **invariante a la escala del fitness**, permitiendo que la misma configuración funcione en instancias MKP de distintos tamaños (n = 100 a n = 500) sin recalibración.

---

## 3. Mecanismo de Decisión

### 3.1 Regla de disparo

La regla es mínima:

```
fire = D₂ ≤ θ_c
```

Donde:
- **D₂** es la distancia DTW (derivative) entre la ventana de fitness actual y una meseta ideal
- **θ_c** es un umbral adaptativo que se auto-calibra durante la ejecución

Cuando `fire = True`, la metaheurística recibe la señal de cambiar sus parámetros al modo **explore**. Cuando `fire = False`, retorna al modo **exploit**.

### 3.2 El umbral adaptativo θ_c

θ_c no es un valor fijo. Se calcula como el **percentil móvil** del historial de D₂:

```
θ_c = percentile(D₂_hist, p_low)
```

Con `p_low = 30`, esto significa que θ_c es el valor por debajo del cual cae el 30% de las observaciones históricas de D₂. En otras palabras: **el sistema aprende qué significa "plano" en el contexto de esta ejecución específica**.

Esta adaptación es fundamental por tres razones:

1. **Invarianza a la escala del problema**: instancias con n=100 y n=500 producen rangos de fitness diferentes, y por tanto valores de D₂ en diferentes escalas. El percentil se adapta automáticamente.

2. **Invarianza a la metaheurística**: PSO, GA, GWO y DE tienen dinámicas de convergencia distintas. Lo que es "plano" para PSO puede no serlo para DE. El percentil captura el comportamiento específico de cada MH.

3. **Invarianza a la fase de búsqueda**: al inicio, cuando la MH explora, D₂ tiende a ser alto (la curva no es plana). En fases tardías, cuando converge, D₂ tiende a ser bajo. El percentil se ajusta dinámicamente a esta evolución.

### 3.3 Período de warm-up

El monitor DTW requiere una ventana de `W = 20` iteraciones antes de producir métricas válidas. Durante este warm-up, no se emiten señales de fire y la MH opera en su modo por defecto (exploit). Esto implica que, con un presupuesto de 50 iteraciones, solo 30 iteraciones son "activas" para el DTW. Con 200 iteraciones, 180 son activas — una relación más favorable.

---

## 4. Comportamiento Esperado

### 4.1 Ventajas frente a la baseline (A4)

| Aspecto | Binary-Simple (A3) | Binary-Complex (A4) |
|---|---|---|
| Señales requeridas | 1 (D₂) | 3 + patience |
| Hiperparámetros de decisión | 0 (θ_c es auto-adaptativo) | 2 (plateau_max, patience) |
| Latencia de detección | Baja (reacciona en cuanto D₂ cruza θ_c) | Alta (necesita acumular plateau + confirmaciones) |
| Riesgo de falsos positivos | Moderado (una fluctuación de D₂ puede disparar) | Bajo (múltiples barreras) |
| Riesgo de falsos negativos | Bajo (detecta cualquier meseta) | Moderado (puede no disparar si plateau_max no se alcanza) |

La hipótesis central es que **la simplicidad de A3 no debería sacrificar rendimiento significativo** frente a A4, porque la señal D₂ ya captura la esencia del estancamiento. Las condiciones adicionales de A4 (plateau prolongado, ramp check) son redundantes cuando D₂ ya está midiendo directamente la planitud de la curva.

### 4.2 Limitaciones conocidas

1. **Ignora el progreso**: A diferencia de A4, A3 no consulta D₁ (distancia a la rampa). Esto significa que no distingue entre una meseta por convergencia al óptimo y una meseta por trampa local. En teoría, podría disparar exploración innecesaria cuando la MH ya encontró el óptimo global.

2. **Sin filtro temporal**: A3 no tiene mecanismo de patience. Si D₂ fluctúa alrededor de θ_c, la MH podría oscilar entre modos exploit/explore. Sin embargo, el uso de `plateau_max=4` en el monitor interno (que opera independientemente de la regla de decisión) proporciona un amortiguamiento parcial.

3. **Sensibilidad al ruido**: en problemas donde el fitness tiene varianza alta entre iteraciones, D₂ puede ser ruidoso. El DDTW mitiga parcialmente esto al operar sobre derivadas, pero no elimina el problema por completo.

---

## 5. Relación con las Metaheurísticas

Binary-Simple es agnóstico a la MH subyacente. La misma señal D₂ ≤ θ_c se aplica a PSO, GA, GWO y DE sin modificaciones. Lo que cambia es **cómo responde cada MH** a la señal:

| MH | Efecto de fire=True (→ explore) |
|---|---|
| **PSO** | w↑ (0.729→0.9), c₁↑ (1.49→2.5), c₂↓ (1.49→0.5). Las partículas confían más en su historia personal y menos en el enjambre. |
| **GA** | crossover↓ (0.9→0.6), mutación↑ (0.01→0.15). Se reduce la herencia de los padres y se aumenta la perturbación aleatoria. |
| **GWO** | a↑ (0.5→2.0). Los lobos se alejan de los líderes alfa/beta/delta, explorando regiones más lejanas. |
| **DE** | F↑ (0.5→0.9), CR↓ (0.9→0.3). Mayor perturbación diferencial y menor herencia del padre. |

La transición es **discreta y global**: todos los individuos de la población cambian sus parámetros simultáneamente. No hay interpolación ni estados intermedios.

---

## 6. Rol en el Paper

Binary-Simple cumple tres funciones en la narrativa científica del estudio:

1. **Validación de la señal D₂**: demuestra que una sola métrica bien elegida (D₂) puede ser suficiente para guiar la adaptación, sin necesidad de las 3 condiciones de A4.

2. **Puente conceptual**: ocupa el punto medio entre la simplicidad ingenua (A1: `fire = δ > 0`) y la complejidad completa (A4). Si A3 funciona comparablemente a A4, se fortalece el argumento de que D₂ es la señal dominante.

3. **Baseline para las versiones continuas**: B3 (Continuous-Simple) es esencialmente la versión continua de A3: en vez de `fire = D₂ ≤ θ_c` (booleano), usa `intensity = 1 − clip(D₂/(θ_c × scale))` (continuo). Tener A3 permite medir cuánto aporta la continuidad por sí sola, aislando el efecto de "discreto vs. continuo" del efecto de "qué señal se usa".

---

## 7. Configuración del Monitor DTW

La configuración del sensor DTW para esta estrategia es:

| Parámetro | Valor | Justificación |
|---|---|---|
| `window` | 20 | Balance entre reactividad (ventana pequeña) y estabilidad (ventana grande) |
| `band` | 2 | Banda Sakoe-Chiba que restringe el alineamiento DTW a ±2 posiciones. Reduce el costo computacional de O(W²) a O(W) y favorece comparaciones locales |
| `min_slope` | 2.0 | Pendiente de la rampa ideal. Un valor alto hace que la rampa sea exigente (espera mejora significativa), lo que indirectamente hace que D₂ domine sobre D₁ en la detección |
| `use_ddtw` | True | Derivative DTW: compara pendientes en vez de valores absolutos. Esencial para invarianza a la escala |
| `adapt_thresholds` | True | θ_c se calcula como percentil móvil del historial de D₂, adaptándose a cada ejecución |
| `p_low` | 30 | Percentil para θ_c: el 30% inferior del historial de D₂ se considera "plano" |

---

## 8. Preguntas Abiertas para el Análisis Experimental

- ¿Es A3 más reactivo que A4? ¿Se traduce esto en más fires pero de menor duración?
- ¿En qué MHs funciona mejor? La hipótesis es que MHs con convergencia más ruidosa (GA, por su mutación) se benefician más del filtro de A4, mientras que MHs más suaves (PSO) funcionan igual de bien con A3.
- ¿El umbral adaptativo θ_c converge a un valor estable o fluctúa durante toda la ejecución?
- ¿Hay diferencia en el gap al óptimo entre A3 y A4 que justifique la complejidad adicional de A4?
