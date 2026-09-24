# Continuous-Complex — Estrategia de Intensidad Sigmoidal sobre Delta

> **Clasificación**: Estrategia continua B1 — Sigmoid Delta  
> **Filosofía**: "No todo estancamiento es igual. La respuesta debe ser proporcional, pero no lineal."

---

## 1. La Estrategia Más Sofisticada

Continuous-Complex (B1) representa la culminación de la evolución desde reglas booleanas simples hacia la adaptación continua: utiliza la **señal más rica** del monitor DTW (delta, que sintetiza D₁ y D₂), la **normaliza** respecto a su propio historial (θ_δ), y la transforma mediante una **función sigmoide** que produce una respuesta no lineal.

Es la contraparte continua de A4 (Binary-Complex): ambas usan delta y sus umbrales, pero donde A4 reduce todo a un bit (fire sí/no), B1 preserva y transforma la magnitud completa de la señal.

---

## 2. Fundamento Teórico

### 2.1 Delta como señal compuesta

A diferencia de D₂ —que solo mide cercanía a la meseta—, delta captura el **balance neto** entre las dos fuerzas que el DTW compara:

```
δ = D₁ − D₂
```

- **δ < 0**: la curva se parece más a una rampa que a una meseta → **progreso neto**
- **δ ≈ 0**: la curva es ambigua, equidistante de ambos patrones → **zona de transición**
- **δ > 0**: la curva se parece más a una meseta que a una rampa → **estancamiento neto**

Delta es inherentemente más informativa que D₂ porque considera **ambas** dimensiones del espacio de formas: no solo "¿es plano?" (D₂), sino también "¿es inclinado?" (D₁). Un D₂ bajo podría deberse a convergencia legítima (la MH ya llegó al óptimo y es correcto que la curva sea plana), pero un delta positivo en ese contexto indicaría que además la curva no muestra progreso —confirmando que la planitud no es deseable.

### 2.2 Normalización: r_balance

Para que delta sea comparable entre ejecuciones, instancias y metaheurísticas, B1 lo normaliza respecto al umbral adaptativo:

```
r_balance = δ / (θ_δ + ε)
```

Donde θ_δ es el percentil 70 del historial de delta. Esta normalización produce una escala interpretable:

| r_balance | Interpretación |
|---|---|
| < 0 | Progreso neto (δ negativo). La curva está más cerca de la rampa. |
| 0 a 1 | Estancamiento leve a moderado. δ está por debajo o cerca de lo que históricamente es "normal". |
| > 1 | Estancamiento severo. δ supera el percentil 70 histórico — está peor que el 70% de las observaciones previas. |

A diferencia de B3, donde el ratio `D₂/(θ_c × scale)` está acotado inferiormente por 0, r_balance puede ser **negativo**. Esto es crucial: permite que el sistema distinga activamente entre progreso y estancamiento, no solo entre grados de estancamiento.

### 2.3 La sigmoide: respuesta no lineal

La transformación final aplica una función sigmoide:

```
intensity = 1 / (1 + exp(−k × (r_balance − center)))
```

Con `k = 5.0` y `center = 0.5`. Esta función tiene tres regiones cualitativamente distintas:

| Región | r_balance | Intensidad | Comportamiento |
|---|---|---|---|
| **Explotación** | r_balance ≪ 0.5 | intensity ≈ 0 | La MH progresa bien. El sistema es **insensible al ruido** en esta zona: pequeñas fluctuaciones de delta no cambian la intensidad. |
| **Transición** | r_balance ≈ 0.5 | intensity ≈ 0.5 | Zona de **máxima sensibilidad**. Pequeños cambios en delta producen grandes cambios en intensidad. El sistema está "escuchando" atentamente. |
| **Exploración** | r_balance ≫ 0.5 | intensity ≈ 1 | Estancamiento severo. El sistema **satura**: más estancamiento no produce más exploración (ya está al máximo). |

La no-linealidad de la sigmoide es su rasgo definitorio frente a B3:

- **B3 (lineal)**: cada unidad adicional de D₂ produce el mismo cambio en intensidad, en todo el rango. Si D₂ es ruidoso, la intensidad también lo es.
- **B1 (sigmoide)**: crea una zona muerta alrededor del progreso (insensibilidad al ruido cuando la MH va bien) y una saturación en estancamiento extremo (evita sobre-reacción). Solo en la zona de transición es altamente sensible.

### 2.4 Los hiperparámetros k y center

| Parámetro | Valor | Efecto |
|---|---|---|
| **k** (steepness) | 5.0 | Controla la pendiente en el punto de inflexión. k alto → transición más abrupta (se aproxima a un switch binario). k bajo → transición más suave (se aproxima a una recta). k = 5.0 produce una transición nítida pero no instantánea. |
| **center** | 0.5 | Punto de inflexión: el valor de r_balance donde intensity = 0.5. center = 0.5 significa que se requiere r_balance > 0.5 (estancamiento moderado) para que la intensidad supere el 50%. |

La elección de `center = 0.5` (no 0, no 1) es deliberada: coloca el punto de máxima sensibilidad en la zona donde delta apenas comienza a ser positivo pero aún no es anormalmente alto. Esto hace que B1 sea **proactivo**: empieza a reaccionar antes de que el estancamiento sea severo, pero con intensidad moderada.

---

## 3. Comparación con B3 (Continuous-Simple)

B1 y B3 comparten la interfaz `adapt_continuous` y el mecanismo de interpolación lineal en las MHs. Difieren en la señal y su transformación:

| Dimensión | B3 (Continuous-Simple) | B1 (Continuous-Complex) |
|---|---|---|
| **Señal primaria** | D₂ (distancia a meseta) | δ = D₁ − D₂ (balance) |
| **Señal secundaria** | — | D₁ (distancia a rampa, vía δ) |
| **Normalización** | D₂ / (θ_c × scale) | δ / θ_δ |
| **Transformación** | Lineal + clip | Sigmoide |
| **Hiperparámetros** | 1 (scale) | 2 (k, center) |
| **Dominio de entrada** | [0, ∞) | (−∞, +∞) |
| **Sensibilidad al ruido** | Constante en todo el rango | Variable: baja en extremos, alta en transición |
| **Saturación** | Solo en el extremo inferior (intensity=0) | En ambos extremos (0 y 1) |
| **Información del DTW usada** | Parcial (solo D₂) | Completa (D₁ y D₂ vía δ) |

La pregunta experimental clave es: **¿justifica la complejidad adicional de B1 (2 hiperparámetros, señal compuesta, sigmoide) una mejora sobre B3 (1 hiperparámetro, señal simple, lineal)?**

---

## 4. Comparación con A4 (Binary-Complex)

B1 es el análogo continuo de A4. Ambos usan delta y sus umbrales, pero difieren radicalmente en cómo:

| Dimensión | A4 (Binary-Complex) | B1 (Continuous-Complex) |
|---|---|---|
| **Uso de delta** | Condición booleana: ¿δ ≥ θ_δ? | Señal continua: intensidad ∝ sigmoid(δ/θ_δ) |
| **Uso de θ_δ** | Umbral fijo en cada iteración (percentil) | Normalizador dinámico |
| **Uso de D₂** | Condición independiente: ¿D₂ ≤ θ_c? | Incorporado en δ (δ = D₁ − D₂) |
| **Filtro temporal** | plateau_max + patience | Inercia natural de la sigmoide (zona muerta) |
| **Latencia** | ~6 iteraciones mínimo | Inmediata pero proporcional |
| **Parámetros libres** | plateau_max, patience | k, center |

La diferencia filosófica es profunda: A4 espera a **confirmar** el estancamiento; B1 **anticipa** el estancamiento y comienza a responder antes de que sea total.

---

## 5. Comportamiento Esperado

### 5.1 Ventajas sobre B3

- **Señal bilateral**: delta distingue progreso de estancamiento, no solo grados de estancamiento. Esto debería traducirse en mejor timing: B1 no debería explorar cuando la MH está progresando (δ < 0 fuerza intensity ≈ 0).
- **No-linealidad adaptativa**: insensible al ruido cuando la MH progresa, altamente sensible en la zona crítica de transición, saturada cuando el estancamiento es extremo. B3 trata todo el rango por igual.
- **Auto-normalización más robusta**: al dividir por θ_δ (percentil de delta) en vez de θ_c (percentil de D₂), la normalización captura el balance histórico completo, no solo la historia de planitud.

### 5.2 Ventajas sobre A4

- **Sin latencia artificial**: B1 no espera plateau_max ni patience. Responde en la misma iteración en que delta cruza la zona de transición.
- **Respuesta proporcional**: un estancamiento leve recibe exploración leve. A4 responde igual a todo estancamiento que supere sus umbrales.
- **Sin falsos negativos por patience**: un estancamiento breve pero real que A4 ignoraría (por no alcanzar patience), B1 lo detecta y responde proporcionalmente.

### 5.3 Limitaciones

- **Dos hiperparámetros**: k y center requieren calibración. Aunque los valores actuales (5.0, 0.5) son defaults razonables, el rendimiento podría ser sensible a ellos.
- **La sigmoide puede saturar**: si k es muy alto, B1 se comporta como un switch binario (degenera en A2). Si k es muy bajo, se comporta como B3 lineal. El valor óptimo depende de la MH y la instancia.
- **Delta puede ser ruidoso**: al ser una diferencia (D₁ − D₂), delta amplifica el ruido de ambas señales. En MHs con convergencia muy ruidosa (GA con alta mutación), esto podría producir fluctuaciones en intensity.
- **Complejidad conceptual**: es la estrategia más difícil de explicar en el paper, combinando normalización, sigmoide, y una señal compuesta.

---

## 6. Rol en el Paper

Continuous-Complex es la **contribución principal** desde la perspectiva de sofisticación del mecanismo adaptativo:

1. **Culminación de la evolución**: la narrativa A4 → B3 → B1 cuenta una historia de progreso: de reglas booleanas complejas a regulación continua simple, y de ahí a regulación continua sofisticada con respuesta no lineal.

2. **Validación del uso de delta**: si B1 supera a B3, se valida que delta (que combina D₁ y D₂) es superior a D₂ solo. Si no, se concluye que la simplicidad de D₂ es suficiente.

3. **Demostración del valor de la no-linealidad**: la comparación B3 vs. B1 aísla el efecto de la sigmoide (no-linealidad) manteniendo constante el mecanismo de adaptación continua.

4. **Estado del arte propuesto**: B1 representa la propuesta más completa del estudio: usa toda la información del DTW, produce una respuesta continua y no lineal, y tiene solo 2 hiperparámetros libres.

---

## 7. Configuración

| Parámetro | Valor | Justificación |
|---|---|---|
| **DTW** | | |
| `window` | 20 | Balance reactividad/estabilidad |
| `band` | 2 | Banda Sakoe-Chiba estrecha |
| `min_slope` | 2.0 | Rampa exigente |
| `use_ddtw` | True | Invarianza a escala del fitness |
| `adapt_thresholds` | True | Umbrales auto-adaptativos |
| `p_low` | 30 | θ_c como percentil 30 |
| `p_high` | 70 | θ_r y θ_δ como percentil 70 |
| **B1** | | |
| `k` | 5.0 | Pendiente de la sigmoide. Transición nítida sin ser binaria |
| `center` | 0.5 | Punto de inflexión. Requiere r_balance > 0.5 para intensity > 0.5 |

---

## 8. Preguntas Abiertas para el Análisis Experimental

- ¿B1 supera a B3 de manera consistente? Si la mejora es marginal, la complejidad adicional (2 hiperparámetros vs. 1, sigmoide vs. lineal) podría no justificarse.
- ¿Cuál es la sensibilidad a k y center? ¿Existe un valor robusto que funcione bien para todas las MHs, o cada MH requiere su propia calibración?
- ¿La curva de intensidad de B1 es más "limpia" (menos ruidosa) que la de B3 gracias a la zona muerta de la sigmoide?
- ¿B1 logra mejor balance exploración-explotación que A4, medido en gap al óptimo y varianza entre epochs?
- ¿La intensidad promedio de B1 es menor que la de B3 (porque delta fuerza intensity ≈ 0 durante el progreso), y esto se traduce en mejor convergencia final?
