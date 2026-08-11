# Guía de Implementación: Hibridación WOA-ABC
## Tres variantes de combinación (A, B, C) — Documento técnico completo

---

## Índice

1. [Introducción y motivación](#1-introducción-y-motivación)
2. [Fundamentos matemáticos de WOA](#2-fundamentos-matemáticos-de-woa)
3. [Fundamentos matemáticos de ABC](#3-fundamentos-matemáticos-de-abc)
4. [Variante A: Switching por parámetro `a` decreciente](#4-variante-a-switching-por-parámetro-a-decreciente)
5. [Variante B: Momentum-Guided (memoria histórica)](#5-variante-b-momentum-guided-memoria-histórica)
6. [Variante C: Switching por diversidad poblacional](#6-variante-c-switching-por-diversidad-poblacional)
7. [Variante Combinada: MDG-WABC (B + C)](#7-variante-combinada-mdg-wabc-b--c)
8. [Estructura de código recomendada](#8-estructura-de-código-recomendada)
9. [Parámetros y valores por defecto](#9-parámetros-y-valores-por-defecto)
10. [Plan de validación experimental](#10-plan-de-validación-experimental)
11. [Checklist de implementación](#11-checklist-de-implementación)

---

## 1. Introducción y motivación

### 1.1 Problema que resolvemos

| Algoritmo | Fortaleza | Debilidad |
|---|---|---|
| **WOA** | Exploración global (movimiento espiral no lineal) | Explotación débil, transición exploración→explotación rígida (basada solo en tiempo) |
| **ABC** | Explotación local fina (employed + onlooker bees), buen escape de óptimos locales (scout bees) | Solo modifica **una dimensión a la vez** → exploración lenta en alta dimensión |

**Objetivo:** construir un algoritmo híbrido donde WOA se encargue de explorar el espacio de búsqueda ampliamente en las primeras iteraciones, y ABC refine las soluciones encontradas con precisión en etapas posteriores — sin depender únicamente de un cronograma fijo (tiempo/iteración), sino también del estado real de la búsqueda.

### 1.2 Las tres variantes explicadas de forma simple

| Variante | Criterio de "cuándo usar WOA vs ABC" | Nivel de originalidad | Complejidad de código |
|---|---|---|---|
| **A** | El propio parámetro `a` de WOA (decrece linealmente con el tiempo) | Baja (reutiliza mecanismo nativo de WOA) | Baja |
| **B** | Memoria de la dirección de búsqueda pasada (momentum) inyectada en ambas fases | Media (tomado de la lógica del paper MSE-ABC, aplicado a WOA+ABC) | Media |
| **C** | Dispersión real de la población (diversidad) en cada iteración | Alta (adaptativo al estado real del problema, no a un cronograma fijo) | Media-Alta |
| **B+C** | Combina memoria histórica (dirección) + diversidad (cuándo cambiar) | Alta (contribución más defendible) | Alta |

**Recomendación de progresión:** implementa A → B → C en ese orden. Cada una se construye sobre la anterior y te permite comparar experimentalmente si el mecanismo adicional realmente mejora el desempeño (esto es evidencia empírica valiosa para tu trabajo).

---

## 2. Fundamentos matemáticos de WOA

### 2.1 Parámetros base

```
a(t) = 2 − t · (2 / MaxIt)        # decrece linealmente de 2 a 0
A = 2·a·r₁ − a                     # r₁ ∈ U(0,1), vector aleatorio
C = 2·r₂                           # r₂ ∈ U(0,1), vector aleatorio
p = rand(0,1)                      # decide entre encircling/spiral
l ∈ U(−1,1)                        # aleatorio, usado en la espiral
b = 1                              # constante de forma de la espiral logarítmica (fija)
```

### 2.2 Los tres comportamientos

**(1) Encircling prey — rodeo de la presa** (si `p < 0.5` y `|A| < 1`):
```
D = |C · X* − X(t)|
X(t+1) = X* − A · D
```
Aquí `X*` es la mejor solución conocida hasta el momento. El agente se mueve hacia ella con un factor de atracción `A`.

**(2) Bubble-net attack — ataque espiral** (si `p ≥ 0.5`):
```
D' = |X* − X(t)|
X(t+1) = D' · e^(b·l) · cos(2π·l) + X*
```
Simula el movimiento en espiral logarítmica que usan las ballenas jorobadas para atrapar presas. Es un movimiento de **explotación local alrededor del mejor**, pero con trayectoria no lineal (a diferencia de PSO, por ejemplo).

**(3) Search for prey — búsqueda aleatoria** (si `p < 0.5` y `|A| ≥ 1`):
```
D = |C · X_rand − X(t)|
X(t+1) = X_rand − A · D
```
En vez de moverse hacia la mejor solución, se mueve hacia una solución aleatoria de la población (`X_rand`). Esto es lo que le da a WOA su capacidad de **exploración global**.

### 2.3 Lógica de decisión completa

```
Para cada individuo i:
    calcular A, C, p, l
    Si p < 0.5:
        Si |A| < 1:
            usar ecuación (1) — encircling (explotación)
        Si |A| ≥ 1:
            usar ecuación (3) — búsqueda aleatoria (exploración)
    Si p ≥ 0.5:
        usar ecuación (2) — espiral (explotación local)
```

**Observación importante:** en WOA puro, la transición exploración↔explotación depende exclusivamente de `a(t)`, que decrece de forma lineal y ciega al progreso real del algoritmo. Esta es precisamente la debilidad que las variantes B y C buscan corregir.

---

## 3. Fundamentos matemáticos de ABC

### 3.1 Inicialización

```
x_i,j = lb_j + rand(0,1) · (ub_j − lb_j)      para cada dimensión j
Fitness_i = f(X_i)
trial_i = 0    # contador de estancamiento por solución
```

### 3.2 Fase Employed Bee (abejas trabajadoras — explotación local)

```
Para cada fuente de comida i (i = 1..FoodNumber):
    k ← índice aleatorio ≠ i
    j ← dimensión aleatoria
    φ ∈ U(−1,1)
    x_new[i,j] = x[i,j] + (x[i,j] − x[k,j]) · φ
    Selección greedy: si f(x_new) < f(x_i) → reemplazar, trial_i = 0
                       si no → trial_i += 1
```

Nota clave: **solo se modifica una dimensión `j` a la vez** — esta es la debilidad estructural que motiva la hibridación.

### 3.3 Fase Onlooker Bee (abejas observadoras — selección por fitness)

```
P_i = 0.9 × [1/(1+Fitness_i)] / max[1/(1+Fitness)] + 0.1

Selección tipo ruleta usando P_i:
    Si rand(0,1) < P_i:
        aplicar la misma ecuación de employed bee sobre la solución i
```

Las fuentes con mejor fitness tienen más probabilidad de ser "visitadas" repetidamente y refinadas.

### 3.4 Fase Scout Bee (abejas exploradoras — escape de óptimos locales)

```
idx ← argmax(trial)   # la solución más estancada
Si trial[idx] ≥ limit:
    x[idx] = lb + rand(0,1) · (ub − lb)   # reinicio aleatorio total
    trial[idx] = 0
```

---

## 4. Variante A: Switching por parámetro `a` decreciente

### 4.1 Idea

Usar el propio parámetro `a(t)` de WOA (que ya existe de forma nativa en el algoritmo) como **interruptor global** que decide, en cada iteración, si toda la población se mueve con las reglas de WOA o con las reglas de ABC.

### 4.2 Ecuaciones

```
a(t) = 2 − t · (2 / MaxIt)

Si |A| ≥ 1  (early stage, a grande):
    → ejecutar FASE WOA completa (exploración)
Si |A| < 1  (late stage, a pequeño):
    → ejecutar FASE ABC completa (employed + onlooker + scout)
```

### 4.3 Pseudocódigo completo

```
INPUT: f(·), lb, ub, nD, Npop, MaxIt, limit

INICIALIZACIÓN
1.  Generar población X_i aleatoria, i = 1..Npop
2.  Evaluar Fitness_i
3.  BestPos, BestFit ← mejor solución inicial
4.  trial_i ← 0 para todo i

BUCLE PRINCIPAL (t = 1 a MaxIt)
5.  a(t) ← 2 − t·(2/MaxIt)
6.  Para cada individuo i:
7.      A ← 2·a·r₁ − a  ;  C ← 2·r₂  ;  p ← rand(0,1)
8.      Si |A| ≥ 1:
9.          ═══ RAMA WOA ═══
10.         Si p < 0.5:
11.             D ← |C·X_rand − X_i|
12.             X_new ← X_rand − A·D                    # búsqueda aleatoria
13.         Sino:
14.             D' ← |BestPos − X_i|
15.             X_new ← D'·e^(bl)·cos(2πl) + BestPos     # espiral
16.     Sino:
17.         ═══ RAMA ABC (employed bee) ═══
18.         k ← índice aleatorio ≠ i ; j ← dimensión aleatoria ; φ ∈ U(−1,1)
19.         X_new ← X_i ;  X_new[j] = X_i[j] + (X_i[j] − X_k[j])·φ
20.     Control de límites [lb, ub]
21.     Selección greedy: actualizar X_i si f(X_new) < f(X_i), sino trial_i += 1

22.  Si estamos en rama ABC, ejecutar también:
23.      FASE ONLOOKER BEE (ruleta + misma ecuación employed)
24.      FASE SCOUT BEE (reiniciar soluciones con trial ≥ limit)

25.  Actualizar BestPos, BestFit
26.  Guardar curva de convergencia

RETURN BestPos, BestFit, Curve
```

### 4.4 Ventajas y limitaciones

- ✅ Muy fácil de implementar y depurar (buen punto de partida)
- ✅ No introduce parámetros nuevos (usa el `a` nativo de WOA)
- ⚠️ Hereda la rigidez de WOA: la transición depende solo del tiempo, no del progreso real de la búsqueda
- ⚠️ Puede generar transiciones abruptas de comportamiento población-completa (todos exploran o todos explotan al mismo tiempo, sin gradualidad)

---

## 5. Variante B: Momentum-Guided (memoria histórica)

### 5.1 Idea

Introducir un término de **momentum**, análogo al usado en optimización de redes neuronales (SGD con momentum), que registra la dirección en la que se movió la mejor solución global entre iteraciones consecutivas, y usa esa dirección para "empujar" las actualizaciones de posición tanto en la fase WOA como en la fase ABC.

### 5.2 Ecuaciones del momentum

```
Momentum(t) = BestPos(t−1) − BestPos(t−2)
M(t) = β · Momentum(t)
```

Donde `β ∈ (0,1)` es el factor de momentum (más alto = más persistencia direccional).

### 5.3 Step size adaptativo (decaimiento exponencial)

```
step(t) = step_init · (step_final / step_init)^(t / MaxIt)
```

Esto genera pasos grandes al inicio (favorece exploración) y pasos pequeños al final (favorece explotación fina), sin necesitar ajuste manual iteración por iteración.

### 5.4 Ecuaciones modificadas

**Fase WOA + momentum:**
```
# Encircling
X_new = X* − A·D + step(t)·M(t)

# Spiral
X_new = D'·e^(bl)·cos(2πl) + X* + step(t)·M(t)

# Búsqueda aleatoria
X_new = X_rand − A·D + step(t)·M(t)
```

**Fase ABC (employed bee) + momentum:**
```
x_new[j] = x[j] + (x[j] − x_k[j])·φ + step(t)·β·Momentum[j](t)
```

### 5.5 Pseudocódigo completo

```
INPUT: f(·), lb, ub, nD, Npop, MaxIt, β, step_init, step_final, limit, GP

INICIALIZACIÓN
1.  Generar población X_i aleatoria
2.  Evaluar Fitness_i
3.  BestPos(0), BestFit(0) ← mejor solución inicial
4.  BestPos_prev ← BestPos(0)
5.  trial_i ← 0

BUCLE PRINCIPAL (t = 1 a MaxIt)
6.  step(t) ← step_init · (step_final/step_init)^(t/MaxIt)
7.  Momentum(t) ← BestPos(t−1) − BestPos_prev
8.  M(t) ← β · Momentum(t)
9.  BestPos_prev ← BestPos(t−1)
10. a(t) ← 2 − t·(2/MaxIt)                    # se conserva para las ecuaciones WOA

    ═══ FASE WOA CON MOMENTUM ═══
11. Para cada individuo i:
12.     r_i ~ U(0,1)
13.     Si r_i < GP:                           # GP = probabilidad de exploración
14.         A, C, p, l ← calcular parámetros WOA
15.         Si p < 0.5 y |A| ≥ 1:
16.             X_new ← X_rand − A·D + step(t)·M(t)
17.         Sino si p < 0.5 y |A| < 1:
18.             X_new ← BestPos − A·D + step(t)·M(t)
19.         Sino:
20.             X_new ← D'·e^(bl)cos(2πl) + BestPos + step(t)·M(t)
21.         Control de límites, selección greedy

    ═══ FASE ABC CON MOMENTUM (employed bee) ═══
22. Para cada fuente de comida i:
23.     k ← aleatorio ≠ i ; j ← dimensión aleatoria ; φ ∈ U(−1,1)
24.     x_new[j] = x[j] + (x[j] − x_k[j])·φ + step(t)·β·Momentum[j](t)
25.     Control de límites, selección greedy (trial_i actualizado)

    ═══ FASE ONLOOKER BEE ═══
26. Calcular P_i según fitness
27. Selección tipo ruleta + misma ecuación de employed bee (paso 24)

    ═══ FASE SCOUT BEE ═══
28. Reiniciar soluciones con trial_i ≥ limit

29. Actualizar BestPos(t), BestFit(t)
30. Guardar curva de convergencia

RETURN BestPos, BestFit, Curve
```

### 5.6 Ventajas y limitaciones

- ✅ Estabiliza la trayectoria de búsqueda, evita oscilaciones erráticas
- ✅ Adaptativo en el "cuánto" (step size), aunque no en el "cuándo" (sigue usando GP fijo o `a(t)`)
- ✅ Validado conceptualmente por el paper MSE-ABC (aplicado ahí a SFOA+ABC)
- ⚠️ Introduce 3 parámetros nuevos: β, step_init, step_final (requiere análisis de sensibilidad)
- ⚠️ El "cuándo" cambiar de fase sigue siendo semi-fijo (GP constante o basado en `a(t)`)

---

## 6. Variante C: Switching por diversidad poblacional

### 6.1 Idea

En lugar de decidir "explorar vs explotar" según el tiempo transcurrido (que es ciego al estado real de la búsqueda), medir qué tan dispersa está la población en cada iteración y usar eso como criterio de decisión:

- Población **muy dispersa** → ya hay buena exploración → conviene **explotar** (ABC)
- Población **muy agrupada** → riesgo de estancamiento en óptimo local → conviene **explorar** (WOA)

### 6.2 Ecuación de diversidad

```
X_mean(t) = (1/Npop) · Σ X_i(t)                        # centroide de la población

Diversidad(t) = (1/Npop) · Σ ||X_i(t) − X_mean(t)||     # dispersión promedio (norma euclidiana)
```

Para hacerla comparable entre problemas de distinta escala, se normaliza:

```
Diversidad_norm(t) = Diversidad(t) / Diversidad(0)      # relativa a la dispersión inicial
```

Donde `Diversidad(0)` es la diversidad de la población inicial (siempre alta, por ser aleatoria uniforme).

### 6.3 Umbral de decisión

```
Si Diversidad_norm(t) > umbral:
    → población dispersa → ejecutar FASE ABC (explotación, converger)
Si Diversidad_norm(t) ≤ umbral:
    → población agrupada → ejecutar FASE WOA (exploración, escapar)
```

**Sugerencia de umbral:** empezar con `umbral = 0.3` (30% de la diversidad inicial) y ajustar mediante el análisis de sensibilidad. También se puede usar un umbral **decreciente** con el tiempo para forzar mayor explotación en etapas tardías incluso si la diversidad se mantiene relativamente alta:

```
umbral(t) = umbral_init · (1 − t/MaxIt) + umbral_final · (t/MaxIt)
```//interpolación lineal entre un umbral alto al inicio y uno bajo al final

### 6.4 Pseudocódigo completo

```
INPUT: f(·), lb, ub, nD, Npop, MaxIt, umbral_init, umbral_final, limit

INICIALIZACIÓN
1.  Generar población X_i aleatoria
2.  Evaluar Fitness_i
3.  BestPos, BestFit ← mejor solución inicial
4.  Diversidad(0) ← calcular dispersión inicial (referencia)
5.  trial_i ← 0

BUCLE PRINCIPAL (t = 1 a MaxIt)
6.  umbral(t) ← umbral_init·(1 − t/MaxIt) + umbral_final·(t/MaxIt)
7.  X_mean ← promedio de la población actual
8.  Diversidad(t) ← (1/Npop)·Σ||X_i − X_mean||
9.  Diversidad_norm(t) ← Diversidad(t) / Diversidad(0)

10. Si Diversidad_norm(t) ≤ umbral(t):
11.     ═══ POBLACIÓN AGRUPADA → FASE WOA (exploración) ═══
12.     Para cada individuo i:
13.         calcular A, C, p, l (según ecuaciones de la Sección 2)
14.         aplicar ecuación WOA correspondiente
15.         Control de límites, selección greedy
16. Sino:
17.     ═══ POBLACIÓN DISPERSA → FASE ABC (explotación) ═══
18.     Ejecutar employed bee + onlooker bee + scout bee (Sección 3)

19. Actualizar BestPos, BestFit
20. Guardar curva de convergencia
21. Guardar Diversidad_norm(t) para análisis posterior (opcional, útil para graficar)

RETURN BestPos, BestFit, Curve
```

### 6.5 Ventajas y limitaciones

- ✅ Genuinamente **adaptativo al estado real del problema**, no a un cronograma predefinido
- ✅ Resuelve directamente la debilidad compartida de WOA y ABC (transición rígida)
- ✅ Es la variante más defendible como contribución académica original
- ⚠️ Requiere calcular la diversidad en cada iteración (costo computacional extra, aunque marginal)
- ⚠️ El umbral requiere ajuste/calibración (mitigado con el análisis de sensibilidad)
- ⚠️ Puede generar oscilaciones si el umbral está mal calibrado (población alterna fase constantemente) — mitigar con un filtro de histéresis si es necesario (ver nota abajo)

**Nota técnica (opcional, avanzado):** si notas que el algoritmo cambia de fase constantemente en iteraciones consecutivas (comportamiento errático), agrega histéresis: usa dos umbrales distintos, uno para pasar a WOA (`umbral_bajo`) y otro para pasar a ABC (`umbral_alto`), de forma que el sistema no oscile cerca del punto de corte.

---

## 7. Variante Combinada: MDG-WABC (B + C)

### 7.1 Idea

Esta es la versión más completa y la que recomendamos como aportación final:

- **De la variante C** tomamos el criterio de "cuándo" cambiar de fase (diversidad poblacional)
- **De la variante B** tomamos el "cómo" se mueven las soluciones dentro de cada fase (momentum + step adaptativo)

**Nombre sugerido:** MDG-WABC = **M**omentum **D**iversity-**G**uided **W**OA-**A**BC

### 7.2 Pseudocódigo completo (versión final)

```
INPUT: f(·), lb, ub, nD, Npop, MaxIt, β, step_init, step_final, 
       umbral_init, umbral_final, limit

INICIALIZACIÓN
1.  Generar población X_i aleatoria, i = 1..Npop
2.  Evaluar Fitness_i
3.  BestPos(0), BestFit(0) ← mejor solución inicial
4.  BestPos_prev ← BestPos(0)
5.  Diversidad(0) ← calcular dispersión inicial de referencia
6.  trial_i ← 0 para todo i

BUCLE PRINCIPAL (t = 1 a MaxIt)

    ─── Cálculo de parámetros dinámicos ───
7.  step(t) ← step_init · (step_final/step_init)^(t/MaxIt)
8.  Momentum(t) ← BestPos(t−1) − BestPos_prev
9.  M(t) ← β · Momentum(t)
10. BestPos_prev ← BestPos(t−1)
11. umbral(t) ← umbral_init·(1−t/MaxIt) + umbral_final·(t/MaxIt)
12. X_mean ← promedio de la población actual
13. Diversidad(t) ← (1/Npop)·Σ||X_i − X_mean||
14. Diversidad_norm(t) ← Diversidad(t) / Diversidad(0)
15. a(t) ← 2 − t·(2/MaxIt)          # necesario para las ecuaciones internas de WOA

    ─── Decisión de fase (criterio de diversidad) ───
16. Si Diversidad_norm(t) ≤ umbral(t):

        ═══ FASE WOA + MOMENTUM (exploración guiada) ═══
17.     Para cada individuo i:
18.         A, C, p, l ← calcular parámetros WOA
19.         Si p < 0.5 y |A| ≥ 1:
20.             X_new ← X_rand − A·D + step(t)·M(t)
21.         Sino si p < 0.5 y |A| < 1:
22.             X_new ← BestPos − A·D + step(t)·M(t)
23.         Sino:
24.             X_new ← D'·e^(bl)cos(2πl) + BestPos + step(t)·M(t)
25.         Control de límites [lb, ub]
26.         Selección greedy (actualizar si mejora, sino trial_i += 1)

27. Sino:
        ═══ FASE ABC + MOMENTUM (explotación guiada) ═══
28.     Para cada fuente de comida i (employed bee):
29.         k ← aleatorio ≠ i ; j ← dimensión aleatoria ; φ ∈ U(−1,1)
30.         x_new[j] = x[j] + (x[j] − x_k[j])·φ + step(t)·β·Momentum[j](t)
31.         Control de límites, selección greedy

32.     Calcular P_i según fitness (onlooker bee)
33.     Selección tipo ruleta + misma ecuación del paso 30

34.     Reiniciar soluciones con trial_i ≥ limit (scout bee)

    ─── Actualización global ───
35. Actualizar BestPos(t), BestFit(t)
36. Guardar curva de convergencia
37. Guardar Diversidad_norm(t) (para análisis y gráficos posteriores)

RETURN BestPos, BestFit, Curve, DiversityHistory
```

### 7.3 Por qué esta combinación es defendible como contribución

| Componente | Qué problema resuelve |
|---|---|
| Momentum (β, Momentum(t)) | Evita oscilaciones erráticas, da consistencia direccional a la búsqueda |
| Step size adaptativo | Pasa de exploración amplia a explotación fina sin necesitar tuning manual iteración a iteración |
| Switching por diversidad | Corrige la debilidad compartida de WOA y ABC (transición rígida basada solo en tiempo) — hace que el algoritmo "sienta" el estado real de la búsqueda |
| Umbral decreciente | Refuerza la explotación en etapas finales incluso si la diversidad se mantiene relativamente alta, evitando desperdiciar iteraciones finales en exploración innecesaria |

---

## 8. Estructura de código recomendada

### 8.1 Organización de archivos (sugerida, Python)

```
proyecto_hibrido/
│
├── functions/
│   ├── cec2022.py           # las 12 funciones CEC2022 (shift, rotate, básicas, híbridas, composición)
│   └── engineering.py        # Welded Beam, Pressure Vessel, Speed Reducer, Three-Bar Truss
│
├── algorithms/
│   ├── woa.py                 # WOA puro (baseline)
│   ├── abc.py                 # ABC puro (baseline)
│   ├── wabc_variant_a.py      # Variante A
│   ├── wabc_variant_b.py      # Variante B (momentum)
│   ├── wabc_variant_c.py      # Variante C (diversidad)
│   └── mdg_wabc.py            # Variante combinada B+C
│
├── utils/
│   ├── diversity.py           # cálculo de dispersión poblacional
│   ├── boundary_control.py    # control de límites [lb, ub]
│   └── stats.py               # Wilcoxon, Friedman, mean/std/best
│
├── experiments/
│   ├── run_cec2022.py         # corre las 4 variantes + baselines en CEC2022
│   ├── run_engineering.py     # corre en problemas de ingeniería
│   ├── sensitivity_analysis.py # análisis de sensibilidad de parámetros
│   └── convergence_plots.py   # generación de curvas de convergencia
│
└── results/
    ├── tables/                # resultados en CSV/Excel
    └── figures/                # curvas de convergencia, boxplots, etc.
```

### 8.2 Función base común (interfaz sugerida)

Todas tus implementaciones (WOA, ABC, variantes A/B/C/combinada) deberían compartir la misma firma de función para poder intercambiarlas fácilmente en los experimentos:

```python
def optimize(func, lb, ub, nD, Npop=30, MaxIt=1000, seed=None, **kwargs):
    """
    Parámetros
    ----------
    func : callable
        Función objetivo a minimizar, recibe un vector x y devuelve un escalar.
    lb, ub : array-like o escalar
        Límites inferior y superior del espacio de búsqueda.
    nD : int
        Dimensión del problema.
    Npop : int
        Tamaño de la población.
    MaxIt : int
        Número máximo de iteraciones.
    seed : int, opcional
        Semilla para reproducibilidad.
    **kwargs : 
        Parámetros específicos de cada variante (beta, step_init, umbral_init, etc.)

    Retorna
    -------
    best_pos : array
        Mejor solución encontrada.
    best_fit : float
        Mejor valor de fitness encontrado.
    curve : array
        Historial de BestFit por iteración (para graficar convergencia).
    """
    ...
    return best_pos, best_fit, curve
```

Esta interfaz uniforme te permite correr los experimentos con un bucle genérico:

```python
algoritmos = {
    "WOA": woa.optimize,
    "ABC": abc.optimize,
    "WABC-A": wabc_variant_a.optimize,
    "WABC-B": wabc_variant_b.optimize,
    "WABC-C": wabc_variant_c.optimize,
    "MDG-WABC": mdg_wabc.optimize,
}

for nombre, algo in algoritmos.items():
    for run in range(30):
        best_pos, best_fit, curve = algo(func, lb, ub, nD, seed=run)
        # guardar resultados...
```

---

## 9. Parámetros y valores por defecto

| Parámetro | Variante(s) | Significado | Valor sugerido inicial | Rango para análisis de sensibilidad |
|---|---|---|---|---|
| `Npop` | Todas | Tamaño de población | 30 | — (fijo, estándar en la literatura) |
| `MaxIt` | Todas | Iteraciones máximas | 1000 | — (fijo, estándar en la literatura) |
| `a` (decae 2→0) | A, B, C, B+C | Parámetro nativo de WOA | automático | — (no se ajusta, es determinístico) |
| `b` | A, B, C, B+C | Constante de forma espiral WOA | 1 | 0.5 – 2 |
| `GP` | B (si se usa switching probabilístico en vez de `a(t)`) | Probabilidad de exploración | 0.5 | 0.2 – 0.8 |
| `β` (beta) | B, B+C | Fuerza del momentum | 0.5 | 0.1 – 0.9 |
| `step_init` | B, B+C | Paso inicial | 1.0 | 0.5 – 2 |
| `step_final` | B, B+C | Paso final | 0.01 | 0.001 – 0.1 |
| `umbral_init` | C, B+C | Umbral de diversidad al inicio | 0.5 | 0.3 – 0.7 |
| `umbral_final` | C, B+C | Umbral de diversidad al final | 0.1 | 0.05 – 0.3 |
| `limit` | Todas (fase ABC) | Intentos antes de scout bee | 100 | 50 – 150 |

**Recomendación metodológica:** sigue el mismo esquema del paper de referencia (Tabla 3): fija todos los parámetros en su valor por defecto y varía uno a la vez en 3 niveles (bajo/medio/alto), evaluando en 4 funciones representativas (una unimodal, una multimodal, una híbrida, una de composición del set CEC2022). Esto te permite reportar "estabilidad de rango" (rank stability) igual que en el paper.

---

## 10. Plan de validación experimental

### 10.1 Fase 1 — Validación de piezas base
Antes de comparar las 4 variantes entre sí, verifica que tus implementaciones de **WOA puro** y **ABC puro** reproduzcan resultados razonables en funciones simples (esfera, Rastrigin clásica) — esto confirma que la base está bien implementada antes de hibridizar.

### 10.2 Fase 2 — Comparación interna de variantes
Corre las 4 variantes (A, B, C, B+C) + WOA puro + ABC puro sobre las 12 funciones CEC2022:

- 30 corridas independientes por función/algoritmo
- Reportar: Best, Mean, SD, tiempo de ejecución
- Test de Wilcoxon signed-rank (cada variante vs. las demás)
- Test de Friedman (ranking global)
- Curvas de convergencia

**Objetivo de esta fase:** determinar cuál variante es mejor, y sobre todo, **si el mecanismo adicional realmente aporta** (por ejemplo: ¿B+C es significativamente mejor que solo B o solo C? ¿o el costo extra de complejidad no se justifica?). Esto es evidencia empírica muy valiosa para justificar tu elección final.

### 10.3 Fase 3 — Comparación contra el estado del arte
Una vez que elijas tu mejor variante (probablemente B+C), compárala contra:
- Los algoritmos base sin hibridar (WOA, ABC)
- Otros algoritmos de tu pool (GWO, PSO, GA — como referencias adicionales)
- Opcionalmente: SHADE, CMA-ES (referentes SOTA reconocidos en la literatura)

### 10.4 Fase 4 — Problemas de ingeniería
Aplicar la mejor variante a los 4-5 problemas clásicos con restricciones (Welded Beam, Pressure Vessel, Speed Reducer, Three-Bar Truss, Tension Spring) usando función de penalización para las restricciones.

### 10.5 Fase 5 — Análisis de sensibilidad
Tabla de rank stability para cada parámetro nuevo introducido (β, step_init, step_final, umbral_init, umbral_final).

### 10.6 Fase 6 (opcional pero valorado) — Aplicación práctica
Un caso de uso real: optimización de hiperparámetros de un modelo de ML/DL, un problema de tu dominio específico, etc. — esto le da peso adicional a tu trabajo, igual que hizo el paper con CIFAR-10.

---

## 11. Checklist de implementación

### Etapa 1: Piezas base
- [ ] Implementar WOA puro y validar en funciones simples
- [ ] Implementar ABC puro y validar en funciones simples
- [ ] Implementar las 12 funciones CEC2022 con shift/rotate
- [ ] Función de cálculo de diversidad poblacional
- [ ] Función de control de límites [lb, ub]

### Etapa 2: Variante A
- [ ] Implementar switching por `a(t)`
- [ ] Validar en CEC2022 (comparar contra WOA y ABC puros)

### Etapa 3: Variante B
- [ ] Implementar cálculo de momentum
- [ ] Implementar step size adaptativo
- [ ] Inyectar momentum en ecuaciones WOA y ABC
- [ ] Validar en CEC2022

### Etapa 4: Variante C
- [ ] Implementar cálculo de diversidad normalizada
- [ ] Implementar umbral decreciente
- [ ] Implementar switching basado en diversidad
- [ ] Validar en CEC2022, revisar si hay oscilación de fase (agregar histéresis si es necesario)

### Etapa 5: Variante combinada B+C
- [ ] Integrar momentum + step adaptativo + switching por diversidad
- [ ] Validar en CEC2022

### Etapa 6: Comparación y selección
- [ ] Correr las 4 variantes + baselines, 30 corridas, 12 funciones
- [ ] Aplicar Wilcoxon + Friedman
- [ ] Generar curvas de convergencia
- [ ] Determinar variante ganadora con evidencia estadística

### Etapa 7: Validación extendida
- [ ] Problemas de ingeniería (5 problemas clásicos)
- [ ] Análisis de sensibilidad de parámetros
- [ ] Comparación contra algoritmos SOTA adicionales (opcional)
- [ ] Caso de aplicación práctica (opcional)

### Etapa 8: Documentación final
- [ ] Redactar pseudocódigo final limpio (formato Algoritmo 1, como en el paper de referencia)
- [ ] Tablas de resultados (formato similar a Tabla 2 y 5 del paper de referencia)
- [ ] Figuras de convergencia (formato similar a Figura 2)
- [ ] Conclusiones y limitaciones

---

## Notas finales

1. **Reproducibilidad:** fija siempre una semilla (`seed`) por corrida y documenta el hardware/software usado, igual que hace el paper de referencia en la sección de "Materials and methods".

2. **Nombrado:** si publicas o documentas formalmente este trabajo, elige un nombre distintivo para tu variante final (ej. `MDG-WABC`) y evita coincidir textualmente con nombres ya usados en la literatura (MSE-ABC, HABCSMO, etc.) para diferenciar claramente tu contribución.

3. **Próximo paso sugerido:** una vez que tengas el código de la Etapa 1 (WOA y ABC puros) funcionando, avísame y seguimos con el código real de cada variante en el lenguaje que prefieras.
