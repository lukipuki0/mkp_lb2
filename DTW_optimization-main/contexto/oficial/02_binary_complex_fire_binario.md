# Binary-Complex — Estrategia Baseline de 3 Condiciones con Patience

> **Clasificación**: Estrategia binaria A4 — Fire multicriterio con filtro temporal  
> **Filosofía**: "No basta con estar estancado. Hay que demostrarlo."

---

## 1. Posición en el Espacio de Diseño

Binary-Complex (A4) es la **estrategia baseline** del estudio: la versión más completa y robusta dentro de la familia de adaptación booleana. Representa el extremo de máxima sofisticación en el espectro binario, utilizando todas las señales disponibles del monitor DTW más un mecanismo de confirmación temporal.

En el marco de estrategias del proyecto:

| Componente | A3 (Binary-Simple) | **A4 (Binary-Complex)** |
|---|---|---|
| Señal primaria | D₂ | D₁, D₂, δ (las tres) |
| Filtro estadístico | θ_c (umbral D₂) | θ_c, θ_r, θ_δ (tres umbrales) |
| Filtro temporal | — (sin filtro) | plateau_max + patience (doble barrera) |
| Señal auxiliar | — | no_improve_len (contador de meseta) |
| Parámetros de decisión | 0 (todo auto-adaptativo) | 2 (plateau_max, patience) |

A4 no es simplemente "A3 con cosas agregadas". Es un cambio de filosofía: donde A3 pregunta "¿está plana la curva?", A4 pregunta "¿hay evidencia suficiente y concordante de estancamiento?". Es un sistema **conservador por diseño**.

---

## 2. Fundamento Teórico

### 2.1 El problema de los falsos positivos en detección de estancamiento

En optimización con metaheurísticas, confundir una pausa breve con estancamiento real tiene un costo: disparar exploración cuando la MH está progresando adecuadamente desperdicia iteraciones y puede descarrilar la convergencia.

Las fuentes de falsos positivos son múltiples:

- **Ruido estocástico**: el fitness puede no mejorar durante 2-3 iteraciones por simple varianza del muestreo, no por estancamiento real.
- **Mesetas estructurales**: en MKP, el operador de reparación greedy puede producir soluciones con idéntico fitness durante varias iteraciones mientras la población se reorganiza internamente.
- **Convergencia legítima**: cuando la MH se aproxima al óptimo, las mejoras son cada vez más pequeñas y espaciadas. Esto es deseable, no un problema.

A4 aborda estas fuentes con una arquitectura de **múltiples capas de filtrado**.

### 2.2 La lógica de las 3 condiciones

La decisión de fire en A4 no se basa en una sola métrica, sino en la **conjunción** de tres condiciones independientes que deben cumplirse simultáneamente:

```
fire = cond_plateau ∧ cond_constant ∧ cond_ramp
       (durante patience iteraciones consecutivas)
```

Cada condición ataca una dimensión distinta del estancamiento:

#### Condición 1: Plateau temporal (`cond_plateau`)
```
no_improve_len ≥ plateau_max
```
Esta es la condición más intuitiva: ¿lleva la MH suficiente tiempo sin mejorar? Con `plateau_max = 4`, se requiere un período de 4 iteraciones consecutivas sin incremento en el best-so-far. Es la **primera barrera**: filtra el ruido estocástico de corto plazo.

#### Condición 2: Forma de meseta (`cond_constant`)
```
D₂ ≤ θ_c
```
¿La forma de la curva de fitness es indistinguible de una línea plana? Esta es la misma señal que usa A3, y constituye la **evidencia geométrica** de estancamiento. No basta con que el fitness no mejore (cond_plateau); la curva entera debe tener forma de meseta según la métrica DTW.

#### Condición 3: Ausencia de progreso (`cond_ramp`)
```
D₁ ≥ θ_r  ∨  δ ≥ θ_δ
```
¿La curva NO se parece a progreso? Esta condición actúa como **confirmación negativa**: incluso si la curva es plana (D₂ bajo), verificamos que tampoco se parece a una rampa de mejora. La disyunción (OR) entre D₁ y δ añade redundancia: si cualquiera de las dos señales indica "lejos de progreso", la condición se satisface.

- **D₁ ≥ θ_r**: la distancia a la rampa ideal es anormalmente alta (peor que el 70% del historial)
- **δ ≥ θ_δ**: el balance D₁−D₂ está anormalmente sesgado hacia estancamiento

### 2.3 El mecanismo de patience

Incluso con las 3 condiciones satisfechas, A4 no dispara inmediatamente. Exige que se mantengan durante `patience = 2` iteraciones consecutivas. Esto es la **segunda barrera** anti-ruido:

```
trigger_streak: contador que se incrementa cuando las 3 condiciones son True
                se resetea a 0 cuando alguna falla

fire = (trigger_streak ≥ patience)
```

El efecto del patience es similar a un **debounce** en electrónica: elimina transiciones espurias que duran una sola iteración. Para que A4 dispare, el estancamiento debe ser **sostenido**, no puntual.

### 2.4 Por qué plateau_max y patience son valores bajos (4 y 2)

Con `plateau_max = 4` y `patience = 2`, el tiempo mínimo para disparar un fire desde que comienza el estancamiento es de aproximadamente 6 iteraciones (4 para alcanzar plateau_max + 2 para satisfacer patience). Estos valores son deliberadamente **bajos** en comparación con los defaults históricos del monitor (15 y 3 respectivamente) porque:

1. El presupuesto de iteraciones es limitado (50-200). Valores altos harían que el sistema nunca dispare en ejecuciones cortas.
2. Las otras 2 condiciones (cond_constant y cond_ramp) ya proporcionan filtrado suficiente. Un plateau_max alto sería redundante.
3. La metaheurística DE, en particular, tiene convergencia rápida y se beneficia de detección temprana.

---

## 3. Los Tres Umbrales Adaptativos

A4 utiliza los tres umbrales que el monitor DTW auto-calibra:

| Umbral | Definición | Rol en A4 |
|---|---|---|
| **θ_c** | `percentile(D₂_hist, p_low=30)` | "¿Qué es anormalmente plano?" — umbral inferior de D₂ |
| **θ_r** | `percentile(D₁_hist, p_high=70)` | "¿Qué es anormalmente lejos de la rampa?" — umbral superior de D₁ |
| **θ_δ** | `percentile(δ_hist, p_high=70)` | "¿Qué es un balance anormalmente sesgado?" — umbral superior de δ |

Los tres se calculan como **percentiles móviles** sobre el historial completo de la ejecución actual. Esto significa que:

- Al inicio, con poco historial (10-20 iteraciones post-warmup), los umbrales son volátiles y se ajustan rápidamente.
- A medida que la ejecución avanza, convergen a valores estables que caracterizan el comportamiento "normal" de esa MH en esa instancia.
- Si la MH cambia de fase (ej: pasa de exploración a convergencia), los umbrales se desplazan con ella, manteniendo la referencia relativa.

Esta adaptabilidad es lo que permite que **la misma configuración de A4 funcione sin recalibración** en instancias de 100, 250 y 500 ítems, y en las cuatro metaheurísticas.

---

## 4. Ciclo de Vida de un Fire

Para entender la dinámica completa, sigamos la secuencia típica de un episodio de estancamiento:

```
Fase 1 — Progreso normal (exploit)
  La MH mejora cada pocas iteraciones.
  no_improve_len se resetea con cada mejora.
  D₂ fluctúa por encima de θ_c (la curva no es plana).
  trigger_streak = 0.

Fase 2 — Inicio de meseta
  La MH deja de mejorar. no_improve_len comienza a crecer.
  La ventana DTW empieza a aplanarse. D₂ desciende.
  Pero no_improve_len < plateau_max → cond_plateau: NO.
  trigger_streak = 0.

Fase 3 — Meseta confirmada
  no_improve_len ≥ plateau_max → cond_plateau: SÍ.
  D₂ ≤ θ_c → cond_constant: SÍ.
  D₁ ≥ θ_r → cond_ramp: SÍ.
  Las 3 condiciones se cumplen → trigger_streak = 1.

Fase 4 — Confirmación sostenida
  Siguiente iteración: las 3 condiciones se mantienen.
  trigger_streak = 2 ≥ patience → FIRE.
  La MH cambia a modo EXPLORE.

Fase 5 — En exploración
  La MH explora con parámetros agresivos.
  Si encuentra mejoras → no_improve_len se resetea.
  Las 3 condiciones eventualmente fallan → trigger_streak = 0.
  La MH retorna a EXPLOIT.
```

---

## 5. Comparación Estructural con Binary-Simple (A3)

| Dimensión | A3 (Binary-Simple) | A4 (Binary-Complex) |
|---|---|---|
| **Señales** | 1 (D₂) | 5 (D₁, D₂, δ, no_improve_len, trigger_streak) |
| **Umbrales** | 1 (θ_c) | 3 (θ_c, θ_r, θ_δ) |
| **Filtros temporales** | 0 | 2 (plateau_max, patience) |
| **Latencia típica** | ~1 iteración post-cruce de θ_c | ~6 iteraciones (4 plateau + 2 patience) |
| **Falsos positivos** | Moderados (fluctuaciones de D₂) | Bajos (triple confirmación) |
| **Falsos negativos** | Bajos (detecta cualquier meseta) | Moderados (puede ignorar mesetas breves) |
| **Oscilación** | Posible cerca de θ_c | Mínima (patience amortigua) |
| **Complejidad conceptual** | Baja | Alta |

La relación entre A3 y A4 no es de "mejor vs. peor", sino de **trade-off deliberado**: reactividad vs. precisión. A3 es un detector rápido que puede generar falsas alarmas; A4 es un detector conservador que puede llegar tarde.

---

## 6. Relación con las Metaheurísticas

A4 es particularmente relevante para MHs con diferentes perfiles de convergencia:

- **PSO**: convergencia relativamente suave. Las 3 condiciones de A4 pueden ser excesivas — A3 podría ser suficiente. El patience de 2 iteraciones es adecuado porque PSO rara vez tiene mesetas de 1 iteración.

- **GA**: convergencia ruidosa por la mutación. El plateau_max = 4 es valioso aquí porque el GA frecuentemente tiene iteraciones sin mejora que no son estancamiento real. El doble filtro (plateau + patience) protege contra falsos positivos que A3 sufriría.

- **GWO**: converge rápido hacia los líderes. Con `a` decayendo, las mejoras se espacian naturalmente. A4 con plateau_max = 4 puede detectar el estancamiento genuino sin confundirlo con la desaceleración natural.

- **DE**: la convergencia más agresiva de las cuatro. La perturbación diferencial produce saltos grandes. Cuando DE se estanca, suele ser definitivo. A4 con paciencia baja (2) es adecuado: cuando las 3 condiciones se alinean, el estancamiento es casi seguro real.

---

## 7. Rol en el Paper

Binary-Complex cumple tres funciones en la narrativa científica:

1. **Baseline principal**: Es la estrategia contra la cual se comparan todas las demás. Representa el estado del arte en detección de estancamiento con DTW: la versión más completa que usa toda la información disponible.

2. **Validación del diseño del monitor**: El hecho de que el monitor produzca D₁, D₂, δ, y los tres thetas permite que A4 exista. Si A4 supera a las versiones vanilla, valida que el enfoque DTW como sensor es útil.

3. **Punto de referencia para ablación**: Cada componente que se remueve de A4 genera una estrategia más simple:
   - Remover cond_ramp y cond_plateau → A3 (Binary-Simple)
   - Remover solo cond_ramp → A3 + plateau (no implementada)
   - Remover patience → A4 sin filtro temporal (no implementada)

   Esto permite, en el análisis, atribuir diferencias de rendimiento a componentes específicos.

---

## 8. Configuración del Monitor DTW

| Parámetro | Valor | Justificación |
|---|---|---|
| `window` | 20 | Ventana suficientemente larga para capturar tendencias, suficientemente corta para ser reactiva |
| `band` | 2 | Banda Sakoe-Chiba estrecha: el alineamiento DTW es casi uno-a-uno, favoreciendo comparaciones locales |
| `min_slope` | 2.0 | Rampa ideal exigente. Esto hace que D₁ sea naturalmente alto, forzando a que la condición cond_ramp dependa más de los umbrales adaptativos que del valor absoluto |
| `use_ddtw` | True | Derivative DTW: esencial para comparar formas independientemente de la escala del fitness |
| `adapt_thresholds` | True | Los tres θ se auto-calibran vía percentiles móviles |
| `p_low` | 30 | θ_c captura el 30% inferior del historial de D₂ |
| `p_high` | 70 | θ_r y θ_δ capturan el 30% superior del historial de D₁ y δ |
| `plateau_max` | 4 | Período mínimo de no-mejora. Valor bajo porque cond_constant y cond_ramp ya filtran |
| `patience` | 2 | Confirmaciones requeridas. Valor bajo para mantener reactividad en ejecuciones de 50-200 iteraciones |

---

## 9. Preguntas Abiertas para el Análisis Experimental

- ¿Es A4 significativamente mejor que A3 en términos de gap al óptimo, o la complejidad adicional no se traduce en mejor rendimiento?
- ¿El número de fires es menor en A4 que en A3? ¿Son fires de mayor calidad (mejor timed)?
- ¿En qué MHs la diferencia A3 vs. A4 es más pronunciada? La hipótesis es GA (más ruidoso → A4 ayuda) vs. DE (más determinista → A3 basta).
- ¿El plateau_max = 4 es óptimo o debería escalar con el presupuesto de iteraciones?
- ¿Las 3 condiciones son todas necesarias? Un análisis de ablación podría revelar que cond_plateau + cond_constant son suficientes (sin cond_ramp).
