# Continuous-Simple — Estrategia de Intensidad Continua desde D₂

> **Clasificación**: Estrategia continua B3 — D2-Direct  
> **Filosofía**: "No prendas ni apagues. Dosifica."

---

## 1. El Salto Conceptual: De Interruptor a Regulador

Continuous-Simple (B3) representa una ruptura fundamental con las estrategias booleanas que la preceden. Mientras A3 y A4 tratan la adaptación como un **interruptor binario** —estancado o no, explore o exploit—, B3 la trata como un **regulador continuo**: el grado de exploración es proporcional al grado de estancamiento detectado.

La diferencia no es meramente cuantitativa (más niveles). Es cualitativa: cambia la naturaleza de la interacción entre el monitor DTW y la metaheurística.

| Dimensión | Estrategias Booleanas (A3, A4) | Estrategia Continua (B3) |
|---|---|---|
| Salida del monitor | `fire ∈ {True, False}` | `intensity ∈ [0, 1]` |
| Parámetros de la MH | 2 conjuntos fijos (exploit/explore) | Interpolación lineal entre los extremos |
| Transiciones | Discretas, instantáneas | Suaves, progresivas |
| Estados posibles | 2 | ∞ (continuo) |
| Información del DTW usada | 1 bit (D₂ ≤ θ_c o no) | Valor continuo de D₂ |
| Riesgo de oscilación | Presente (on/off cerca del umbral) | Eliminado (cambios graduales) |

---

## 2. Fundamento Teórico

### 2.1 La señal D₂ como intensidad

En A3, la pregunta es binaria: ¿D₂ ≤ θ_c? La respuesta es sí o no, y la MH reacciona con un cambio total de parámetros.

B3 reformula la pregunta: **¿cuán lejos está D₂ de θ_c?** La respuesta es un valor continuo que se traduce directamente en intensidad de exploración:

```
intensity = 1 − clip( D₂ / (θ_c × scale), 0, 1 )
```

Donde:
- **D₂ / (θ_c × scale)** es una medida normalizada de cuán plana es la curva, relativa al umbral adaptativo
- Cuando D₂ ≈ 0 (curva perfectamente plana) → ratio ≈ 0 → intensity ≈ 1 (explore puro)
- Cuando D₂ = θ_c × scale → ratio = 1 → intensity = 0 (exploit puro)
- Cuando D₂ > θ_c × scale → ratio > 1 → intensity = 0 (exploit puro, saturación inferior)

### 2.2 El parámetro scale: sensibilidad del regulador

`scale = 2.0` es el único hiperparámetro libre de B3. Controla **qué tan sensible es el sistema al estancamiento**:

| scale | Comportamiento |
|---|---|
| scale → 0 | intensity ≈ 1 siempre (exploración constante). Degenera en vanilla-exploración. |
| scale = 1 | D₂ = θ_c produce intensity = 0. El punto de transición coincide exactamente con el umbral adaptativo. Muy sensible. |
| **scale = 2** | D₂ debe ser la mitad de θ_c para que intensity = 0.5. **Más tolerante**: requiere estancamiento más pronunciado para explorar fuerte. |
| scale → ∞ | intensity ≈ 0 siempre (explotación constante). Degenera en vanilla-explotación. |

Con `scale = 2.0`, el sistema es **conservador**: incluso cuando D₂ = θ_c (el punto que en A3 dispararía fire), la intensidad es solo 0.5 —una exploración moderada, no total.

### 2.3 ¿Por qué D₂ y no delta para la versión continua simple?

La elección de D₂ como señal para B3 no es arbitraria. Responde a tres propiedades deseables:

1. **Dominio acotado inferiormente**: D₂ ≥ 0 siempre. Esto hace que el ratio `D₂ / (θ_c × scale)` esté naturalmente acotado, simplificando el clipping.

2. **Monotonicidad con el estancamiento**: a menor D₂, mayor estancamiento. La relación es directa y no tiene inversiones. Delta, en cambio, puede ser negativo (progreso) y su interpretación requiere contexto.

3. **Continuidad natural con A3**: B3 es la evolución directa de A3. Esto permite aislar el efecto de "binario → continuo" manteniendo constante la señal utilizada.

---

## 3. Mecanismo de Interpolación

### 3.1 La interfaz `adapt_continuous(intensity)`

A diferencia de `adapt(fire: bool)` que solo conmuta entre dos conjuntos de parámetros, `adapt_continuous(intensity: float)` interpola linealmente:

```
param_actual = param_exploit + intensity × (param_explore − param_exploit)
```

Con `intensity = 0` → parámetros de exploit puro.  
Con `intensity = 1` → parámetros de explore puro.  
Con `intensity = 0.5` → punto medio exacto entre ambos extremos.

### 3.2 Efecto en cada metaheurística

| MH | Parámetros interpolados | Efecto de intensity creciente |
|---|---|---|
| **PSO** | w (0.729→0.9), c₁ (1.49→2.5), c₂ (1.49→0.5) | Las partículas gradualmente ignoran más al enjambre y confían más en su historia personal |
| **GA** | crossover_rate (0.9→0.6), mutation_rate (0.01→0.15) | Progresivamente menos herencia de padres y más perturbación aleatoria |
| **GWO** | a (0.5→2.0) | Los lobos se alejan gradualmente de los líderes; con intensity=0.5, a=1.25 (decay natural de GWO) |
| **DE** | F (0.5→0.9), CR (0.9→0.3) | Mayor perturbación diferencial y menor herencia del padre, de forma progresiva |

### 3.3 La etiqueta de modo como convención

Aunque la adaptación es continua, cada MH mantiene la propiedad `mode` con valor `"exploit"` o `"explore"`. Esta etiqueta se determina por un umbral convencional: `intensity > 0.5 → "explore"`. Es importante entender que esto es solo una **etiqueta descriptiva** para visualización y análisis; no afecta el comportamiento, que está determinado por los valores interpolados.

---

## 4. Comparación Estructural con A3 (Binary-Simple)

B3 es la contraparte continua de A3. Compararlas permite aislar el efecto de la continuidad:

| Dimensión | A3 (Binary-Simple) | B3 (Continuous-Simple) |
|---|---|---|
| **Señal** | D₂ | D₂ |
| **Umbral** | θ_c | θ_c × scale |
| **Decisión** | D₂ ≤ θ_c → fire=True | D₂/(θ_c × scale) → intensity continua |
| **Parámetros MH** | Salto discreto: exploit ↔ explore | Interpolación lineal: ∞ puntos intermedios |
| **Hiperparámetros** | 0 | 1 (scale) |
| **Comportamiento en el borde** | Puede oscilar si D₂ ≈ θ_c | Transición suave, sin oscilación |
| **Información descartada** | Magnitud de D₂ (solo importa si cruza θ_c) | Ninguna (toda la magnitud se usa) |

La hipótesis central es que **B3 debería superar a A3** porque:
1. No desperdicia la magnitud de D₂: un estancamiento severo (D₂ ≈ 0) recibe exploración máxima; un estancamiento leve (D₂ ≈ θ_c) recibe solo exploración moderada.
2. Elimina la oscilación: no hay un punto de quiebre donde el sistema salta de un estado al otro.
3. Permite estados mixtos: la MH puede estar simultáneamente explotando y explorando en proporción a la señal.

---

## 5. Comportamiento Esperado

### 5.1 Ventajas sobre A3

- **Dosificación**: un estancamiento leve no descarrila la convergencia con exploración total. La MH recibe exactamente la dosis de exploración que necesita.
- **Estabilidad**: sin punto de discontinuidad, no hay oscilación explotar/explotar.
- **Aprovechamiento de información**: toda la precisión del DTW se conserva en la salida.

### 5.2 Ventajas sobre A4

- **Simplicidad**: 1 hiperparámetro (scale) vs. al menos 2 (plateau_max, patience) que requiere A4.
- **Proactividad**: no espera a acumular evidencia de estancamiento. Reacciona inmediatamente, pero de forma proporcional.
- **Sin falsos negativos por patience**: si el estancamiento es breve pero real, B3 lo detecta y responde (con intensidad proporcional), mientras que A4 podría ignorarlo si no alcanza el patience.

### 5.3 Limitaciones

- **Scale es crítico**: un scale mal calibrado puede hacer que el sistema sea demasiado reactivo (intensidad alta constante) o demasiado pasivo (nunca explora). Aunque θ_c es auto-adaptativo, scale es fijo y debe elegirse con criterio.
- **No usa D₁**: al igual que A3, B3 ignora la señal de progreso. No distingue entre meseta por convergencia al óptimo y meseta por trampa local.
- **Linealidad**: la interpolación lineal asume que la relación entre D₂ e intensidad óptima es lineal, lo cual es una simplificación. La versión B1 (sigmoid) explora una relación no lineal.

---

## 6. Rol en el Paper

Continuous-Simple ocupa una posición estratégica en la narrativa del estudio:

1. **Demostración del valor de la continuidad**: al compartir la misma señal (D₂) que A3 pero diferir solo en el mecanismo de salida (binario vs. continuo), la comparación A3 vs. B3 es un **experimento controlado** que aísla el efecto de "discreto vs. continuo".

2. **Puente hacia B1**: B3 es la versión más simple de la familia continua. B1 (sigmoid sobre delta) agrega no-linealidad y una señal diferente. Tener B3 permite determinar si las mejoras de B1 vienen de la continuidad (que B3 ya tiene) o de la señal delta + sigmoid.

3. **Validación de `adapt_continuous`**: B3 es la primera estrategia que ejerce la interfaz de adaptación continua. Su existencia valida el diseño arquitectónico que agregó `adapt_continuous` a `BaseMH`.

4. **Elegancia conceptual**: para el paper, B3 es quizás la estrategia más elegante: una sola señal (D₂), un solo parámetro (scale), y una respuesta proporcional. Representa la filosofía de "máxima simplicidad con máxima información".

---

## 7. Configuración

| Parámetro | Valor | Justificación |
|---|---|---|
| **DTW** | | |
| `window` | 20 | Balance reactividad/estabilidad |
| `band` | 2 | Banda Sakoe-Chiba estrecha |
| `min_slope` | 2.0 | Rampa exigente |
| `use_ddtw` | True | Invarianza a escala del fitness |
| `adapt_thresholds` | True | θ_c auto-adaptativo vía percentil |
| `p_low` | 30 | θ_c como percentil 30 de D₂ histórico |
| **B3** | | |
| `scale` | 2.0 | Factor de tolerancia. D₂ debe ser ≤ θ_c/2 para intensity ≥ 0.5 |

---

## 8. Preguntas Abiertas para el Análisis Experimental

- ¿B3 supera consistentemente a A3? Si sí, la continuidad se valida como mejora universal. Si no, hay casos donde el switch binario es preferible.
- ¿Cuál es el scale óptimo? ¿Depende de la MH? ¿De la instancia? Un análisis de sensibilidad de scale sería valioso.
- ¿La intensity promedio se correlaciona con la calidad de la solución final? ¿Existe un "punto dulce" de intensidad?
- ¿B3 exhibe menos varianza entre epochs que A3? La hipótesis es que la estabilidad de la interpolación debería traducirse en resultados más consistentes.
- Comparado con B1 (sigmoid sobre delta): ¿la señal D₂ es suficiente, o delta aporta información complementaria que mejora el rendimiento?
