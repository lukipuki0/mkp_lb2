# Grey Wolf Optimizer (GWO) para MKP — Fundamentos y Conceptos

> **Referencia base**: Mirjalili, Mirjalili & Lewis (2014) · Emary et al. (2016)  
> **Variante utilizada**: Binary GWO con sigmoid transfer function

---

## 1. ¿Qué es GWO y por qué es radicalmente distinto?

**Grey Wolf Optimizer** (Mirjalili et al., 2014) es una metaheurística poblacional inspirada en la **estructura social jerárquica y el comportamiento de caza** de los lobos grises en la naturaleza. A diferencia de todas las demás metaheurísticas del estudio, GWO **no tiene noción de velocidad ni de selección explícita**. Los lobos no "vuelan" como en PSO, no "compiten" como en GA, y no "se diferencian" como en DE. Simplemente **siguen a sus líderes**.

| MH | Metáfora social | Mecanismo matemático |
|---|---|---|
| **PSO** | Bandada de aves | Atracción vectorial con inercia |
| **GA** | Evolución natural | Selección + recombinación + mutación |
| **DE** | Evolución por diferencias | Perturbación vectorial auto-escalante |
| **GWO** | Manada de lobos cazando | Seguimiento jerárquico de líderes hacia la presa |

---

## 2. La Jerarquía del Lobo Gris

En la naturaleza, las manadas de lobos grises tienen una estructura social de cuatro niveles perfectamente definida. GWO modela matemáticamente esta jerarquía:

### Los cuatro rangos

| Rango | Rol biológico | Rol en GWO | Cantidad |
|---|---|---|---|
| **Alpha (α)** | Líder absoluto. Toma decisiones (dónde cazar, cuándo dormir). | **Mejor solución encontrada**. Guía principal de la manada. | 1 |
| **Beta (β)** | Segundo al mando. Asiste al alpha y disciplina a los subordinados. | **Segunda mejor solución**. Refuerza la dirección del alpha. | 1 |
| **Delta (δ)** | Tercer escalón. Centinelas, cazadores, cuidadores. | **Tercera mejor solución**. Proporciona una perspectiva adicional. | 1 |
| **Omega (ω)** | El resto de la manada. Obedecen a todos los superiores. | **Todos los demás individuos**. Son guiados por α, β y δ. | Población − 3 |

### La intuición detrás de la jerarquía

La idea es brillante en su simplicidad: en vez de seguir a UN solo líder (como PSO sigue al gbest), cada lobo sigue a **tres líderes simultáneamente** y promedia sus direcciones. Esto produce un comportamiento de búsqueda cualitativamente distinto:

- **PSO**: cada partícula tiene DOS atractores (pbest y gbest). Si ambos están en el mismo óptimo local, el enjambre colapsa.
- **GWO**: cada lobo tiene TRES atractores (α, β, δ). Para que la manada colapse, los tres líderes deben estar en la misma región — lo cual es matemáticamente menos probable.
- **Resultado**: GWO es inherentemente más resistente a la convergencia prematura que PSO.

---

## 3. El Modelo Matemático de Caza

GWO modela dos comportamientos del lobo gris: **acechar** (exploración) y **atacar** (explotación). Ambos se controlan mediante un único parámetro: **`a`**.

### 3.1 Las ecuaciones de movimiento

Para cada lobo omega, la nueva posición se calcula en tres pasos:

**Paso 1: Calcular vectores hacia cada líder**

```
D_α = |C₁ · X_α − X|     ← distancia al alpha
D_β = |C₂ · X_β − X|     ← distancia al beta
D_δ = |C₃ · X_δ − X|     ← distancia al delta
```

Donde `C = 2 · r` (con `r ~ U(0,1)`) es un **coeficiente de perturbación** que introduce peso estocástico a la distancia. `C > 1` enfatiza al líder, `C < 1` lo atenúa.

**Paso 2: Calcular movimiento hacia cada líder**

```
X₁ = X_α − A₁ · D_α     ← vector hacia alpha
X₂ = X_β − A₂ · D_β     ← vector hacia beta
X₃ = X_δ − A₃ · D_δ     ← vector hacia delta
```

Donde `A = 2a · r − a` (con `r ~ U(0,1)`) es el **coeficiente de ataque**. Este es el parámetro crucial:

- **|A| > 1**: el lobo se ALEJA del líder → **exploración** (acechar, buscar presa)
- **|A| < 1**: el lobo se ACERCA al líder → **explotación** (atacar, converger)

**Paso 3: Promediar las tres direcciones**

```
X_nueva = (X₁ + X₂ + X₃) / 3
```

El lobo no sigue ciegamente a un solo líder. **Promedia** las direcciones hacia los tres, lo que produce un movimiento más estable y menos susceptible a quedar atrapado si un líder está en un óptimo local.

### 3.2 El parámetro `a`: un solo número lo controla todo

GWO es la más minimalista de las cuatro metaheurísticas. Tiene **un solo parámetro libre**: `a`.

El coeficiente `a` controla la magnitud de `A`:

```
A = 2a · r − a,   donde r ~ U(0,1)

Por lo tanto:  A ∈ [−a, +a] aproximadamente
```

| Valor de `a` | Rango de A | Efecto |
|---|---|---|
| **a = 2.0** | A ∈ [−2, +2] aprox. | |A| frecuentemente > 1 → lobos se alejan de líderes → **EXPLORACIÓN** |
| **a = 1.0** | A ∈ [−1, +1] aprox. | Transición: mitad de los movimientos exploran, mitad explotan |
| **a = 0.5** | A ∈ [−0.5, +0.5] | |A| siempre < 1 → lobos convergen hacia líderes → **EXPLOTACIÓN** |
| **a → 0** | A → 0 | Lobos colapsan exactamente sobre los líderes |

### 3.3 La diferencia crucial con el GWO estándar

En el GWO original de Mirjalili et al. (2014), `a` **decae linealmente** de 2 a 0 a lo largo de las iteraciones:

```
a(t) = 2 − 2 · (t / max_iter)
```

Esto implementa un **balance exploración-explotación preprogramado**: mucha exploración al principio (a ≈ 2), mucha explotación al final (a ≈ 0). La transición es ciega: no depende del estado real de la búsqueda.

En nuestro estudio, **rompemos con este esquema**. En lugar de un decaimiento predeterminado, fijamos `a` a dos valores discretos y dejamos que el **DTW decida cuándo usar cada uno**:

- **a = 0.5** (modo exploit): los lobos se agrupan alrededor de los líderes, refinando la mejor región conocida
- **a = 2.0** (modo explore): los lobos se dispersan, buscando nuevas regiones prometedoras

Esto transforma a GWO de un algoritmo con **balance preprogramado** a uno con **balance adaptativo basado en evidencia**.

---

## 4. Binary GWO para MKP

### 4.1 El desafío de la discretización

GWO fue diseñado para espacios continuos. Sus ecuaciones producen vectores en ℝⁿ, pero MKP requiere vectores en {0,1}ⁿ. La solución —análoga a PSO— es usar una **función de transferencia sigmoide**:

```
X_nueva (continua) → sigmoid → probabilidad → muestreo → solución binaria
```

### 4.2 ¿Qué significa "moverse hacia un líder" en espacio binario?

En el espacio continuo, `X₁ = X_α − A₁ · D_α` tiene una interpretación geométrica clara: el lobo se desplaza hacia alpha. En el espacio binario, esta ecuación produce un vector continuo que luego se binariza.

La intuición es: el vector continuo `X_nueva` codifica, para cada ítem, **cuán fuerte es la recomendación de los líderes de incluirlo**. Un valor positivo grande en la posición `j` significa "los tres líderes coinciden en que el ítem `j` debería estar en la solución". La sigmoide convierte esta intensidad en probabilidad.

### 4.3 El efecto de los líderes en MKP

En MKP, α, β y δ son las tres mejores soluciones factibles encontradas. Si los tres incluyen el ítem `j`, el lobo omega tendrá alta probabilidad de incluirlo. Si los tres lo excluyen, alta probabilidad de excluirlo. Si hay desacuerdo entre los líderes (α lo incluye, β lo excluye), la probabilidad será intermedia — el lobo "decide por su cuenta" con un sesgo leve.

Esto es **cualitativamente distinto del crossover del GA**: en GA, el hijo hereda bits de dos padres; en GWO, el lobo recibe **recomendaciones ponderadas de tres líderes** y las sintetiza en un solo vector.

---

## 5. GWO frente a la Naturaleza del MKP

### 5.1 El problema de los líderes atrapados

La mayor vulnerabilidad de GWO es el **liderazgo colapsado**. Si α, β y δ están todos atrapados en la misma región del espacio de búsqueda (tres soluciones muy similares), los lobos omega promedio vectores casi idénticos — y la manada entera converge a ese óptimo local.

En MKP, esto se agrava porque el operador de reparación **determinístico** puede hacer que tres soluciones distintas en bits se vuelvan idénticas después de reparar. Si α = β = δ, GWO colapsa completamente: `X₁ = X₂ = X₃` → `X_nueva = X_α` → todos los lobos se vuelven clones de alpha en una sola iteración.

**Esto justifica `a = 2.0` en modo explore**: necesitamos que los lobos **ignoren activamente a los líderes** (|A| > 1) cuando el DTW detecta estancamiento. Es la única forma de escapar de un colapso de liderazgo.

### 5.2 El rol del coeficiente C

El coeficiente `C = 2 · r` es frecuentemente subestimado en la literatura de GWO, pero cumple un rol sutil e importante:

- **C > 1**: el lobo "sobrestima" la distancia al líder, dando pasos más largos → efecto exploratorio
- **C < 1**: el lobo "subestima" la distancia, dando pasos más cortos → efecto explotatorio

Como `C` es aleatorio en cada dimensión y cada iteración, inyecta **diversidad estocástica** incluso cuando `a` fuerza convergencia. Es un mecanismo de escape de emergencia: aunque `a = 0.5` fuerce |A| < 1, un C particularmente alto en alguna dimensión puede hacer que un lobo dé un paso inesperadamente largo en esa dirección.

---

## 6. Parámetros para Modo Exploit y Modo Explore

### 6.1 Modo Exploit — Caza de Precisión

| Parámetro | Valor | Fundamento |
|---|---|---|
| **a** | 0.5 | Fuerza |A| < 1 en la mayoría de las iteraciones. Los lobos se acercan a los líderes. En el GWO estándar, este valor correspondería a la fase final (75% de las iteraciones transcurridas). |

**Qué sucede**: La manada se agrupa alrededor de α, β y δ. Los movimientos son cortos y dirigidos hacia los líderes. Es una búsqueda **local intensiva**: los lobos refinan la región donde los líderes —las mejores soluciones conocidas— se encuentran. El riesgo es que si los líderes están en un óptimo local, toda la manada converge allí.

### 6.2 Modo Explore — Búsqueda de Nuevas Presas

| Parámetro | Valor | Fundamento |
|---|---|---|
| **a** | 2.0 | Fuerza |A| > 1 con alta probabilidad. Los lobos se alejan de los líderes. En el GWO estándar, este es el valor inicial (máxima exploración). |

**Qué sucede**: Los lobos **divergen** de los líderes. Las direcciones `X₁`, `X₂`, `X₃` apuntan lejos de α, β y δ. El promedio de tres vectores divergentes produce movimientos erráticos que cubren regiones distantes del espacio de búsqueda. Si los líderes están estancados, este mecanismo permite que la manada "se reinicie" y busque en zonas nuevas.

### 6.3 Tabla resumen

| Parámetro | Exploit | Explore | Cambio | Efecto |
|---|---|---|---|---|
| **a** | 0.5 | 2.0 | ×4 | De convergencia local a dispersión global |
| **|A| típico** | < 1 (converger) | > 1 (diverger) | Cambio cualitativo de comportamiento |
| **Fase GWO estándar equivalente** | Final (75-100%) | Inicio (0%) | — | Nuestro GWO puede alternar entre fases según necesidad |

### 6.4 La belleza de un solo parámetro

A diferencia de PSO (3 parámetros), GA (4+) o DE (2), GWO tiene **exactamente un parámetro libre**. Esto lo convierte en:

- **El más fácil de analizar**: cualquier diferencia de rendimiento entre modos es atribuible únicamente a `a`
- **El más elegante teóricamente**: no hay interacciones entre parámetros que confundan el análisis
- **El más dependiente del DTW**: sin el decaimiento lineal estándar, GWO depende completamente de la señal externa para balancear exploración y explotación

---

## 7. ¿Por qué GWO Complementa a PSO, GA y DE?

| Aspecto | GA | PSO | GWO | DE |
|---|---|---|---|---|
| Mecanismo | Selección + crossover + mutación | Atracción vectorial dual | Seguimiento jerárquico triple | Diferencia vectorial |
| Parámetros libres | 4+ | 3 | **1** | 2 |
| Memoria | Población actual | pbest + gbest | α, β, δ | Población actual |
| Naturaleza de la búsqueda | Constructiva (bloques) | Direccional (vectores) | **Consensual (promedio de líderes)** | Diferencial (perturbación) |
| Riesgo principal | Convergencia prematura por presión selectiva | Colapso del enjambre al gbest | **Colapso de liderazgo (α=β=δ)** | Estancamiento por pérdida de diversidad |
| Mecanismo de escape | Mutación agresiva (bit-flip) | c₂↓ (ignorar gbest) | **a↑ (diverger de líderes)** | F↑ (aumentar perturbación) |

### Lo que GWO aporta que los otros no tienen:

1. **Liderazgo distribuido**: seguir a tres líderes en vez de uno hace a GWO inherentemente más robusto contra la convergencia prematura. Para que la manada colapse, los tres líderes deben coincidir — un evento de probabilidad significativamente menor que el colapso al gbest en PSO.

2. **Simplicidad radical**: un solo parámetro. Esto hace de GWO el **experimento más limpio** del estudio. Si GWO+DTW funciona, sabemos exactamente qué variable (a) produjo el efecto. No hay interacciones que desenredar.

3. **Comportamiento cualitativo dual**: el parámetro `a` produce un **cambio de fase** en el comportamiento colectivo: de convergencia (a < 1) a divergencia (a > 1). No es una cuestión de grado — es un cambio cualitativo. Ninguna otra MH del estudio tiene un parámetro con esta propiedad de "inversión de régimen".

4. **El experimento de "quitarle el decaimiento"**: el GWO estándar tiene su balance exploración-explotación cableado por el decaimiento lineal de `a`. Al reemplazar ese decaimiento por control DTW externo, nuestro estudio evalúa empíricamente si un balance **adaptativo basado en evidencia** supera a uno **preprogramado basado en el tiempo**. Esta es una pregunta de investigación valiosa por sí misma.

---

## 8. Referencias Clave

- **GWO original**: Mirjalili, S., Mirjalili, S.M., & Lewis, A. (2014). "Grey Wolf Optimizer." *Advances in Engineering Software*, 69, 46-61.
- **Binary GWO**: Emary, E., Zawbaa, H.M., & Hassanien, A.E. (2016). "Binary Grey Wolf Optimization Approaches for Feature Selection." *Neurocomputing*, 172, 371-381.
- **Análisis de convergencia**: Saxena, A., Soni, B.P., & Kumar, R. (2018). "Convergence Analysis of Grey Wolf Optimizer." *Journal of Intelligent & Fuzzy Systems*, 35(5), 5249-5261.
- **GWO multi-objetivo**: Mirjalili, S., Saremi, S., Mirjalili, S.M., & Coelho, L.S. (2016). "Multi-Objective Grey Wolf Optimizer: A Novel Algorithm for Multi-Criterion Optimization." *Expert Systems with Applications*, 47, 106-119.
- **GWO para knapsack**: Luo, K., & Zhao, Q. (2019). "A Binary Grey Wolf Optimizer for the Multidimensional Knapsack Problem." *Applied Soft Computing*, 83, 105645.
- **Survey de variantes GWO**: Faris, H., Aljarah, I., Al-Betar, M.A., & Mirjalili, S. (2018). "Grey Wolf Optimizer: A Review of Recent Variants and Applications." *Neural Computing and Applications*, 30(2), 413-435.
