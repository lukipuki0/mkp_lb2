# Genetic Algorithm (GA) para MKP — Fundamentos y Conceptos

> **Referencia base**: Holland (1975) · Goldberg (1989) · Chu & Beasley (1998)  
> **Variante utilizada**: GA binario generacional con elitismo

---

## 1. ¿Qué es un Algoritmo Genético?

Un Algoritmo Genético es una metaheurística poblacional inspirada en la **evolución natural por selección**. La idea central es tratar las soluciones candidatas como **individuos de una población** que compiten, se reproducen y mutan a lo largo de generaciones. Los más aptos (mayor fitness) tienen más probabilidad de transmitir sus "genes" a la siguiente generación.

A diferencia de PSO —donde las partículas se mueven por atracción vectorial— o de GWO —donde los lobos siguen líderes—, el GA opera mediante **operadores genéticos explícitos** que emulan los mecanismos de la evolución biológica:

| MH | Metáfora | Mecanismo |
|---|---|---|
| **PSO** | Bandada de aves | Atracción por vectores hacia pbest y gbest |
| **GWO** | Manada de lobos | Seguimiento jerárquico de líderes (α, β, δ) |
| **DE** | Evolución por diferencias | Perturbación vectorial auto-escalante |
| **GA** | Evolución natural | Selección, recombinación sexual y mutación aleatoria |

---

## 2. Los Tres Operadores Fundamentales

Todo GA se construye sobre tres pilares. Cambiar cualquiera de ellos altera profundamente el comportamiento del algoritmo.

### 2.1 Selección — ¿Quién se reproduce?

La selección es el mecanismo que decide **qué individuos tienen oportunidad de pasar sus genes** a la siguiente generación. Es el análogo a la "supervivencia del más apto".

El operador más común —y el utilizado en este estudio— es la **selección por torneo**:

- Se eligen `k` individuos al azar de la población (en nuestro caso, `k = 3`)
- El de mayor fitness entre ellos es seleccionado como progenitor
- Se repite para obtener el segundo progenitor

**¿Por qué torneo y no ruleta?** La selección por ruleta (proporcional al fitness) sufre de **dominancia prematura**: un individuo excepcionalmente bueno acapara todas las oportunidades de reproducción, colapsando la diversidad en pocas generaciones. El torneo, al elegir los competidores al azar, le da oportunidad de reproducirse incluso a individuos mediocres — preservando diversidad.

**Efecto en exploración/explotación**:

- **Torneo pequeño (k = 2)**: baja presión selectiva. Incluso individuos malos se reproducen. Favorece **exploración**.
- **Torneo grande (k = 5+)**: alta presión selectiva. Solo los muy buenos se reproducen. Favorece **explotación**.
- **k = 3** es un balance estándar (Miller & Goldberg, 1995).

### 2.2 Crossover (Recombinación) — ¿Cómo se mezclan los genes?

El crossover emula la **reproducción sexual**: dos progenitores combinan su material genético para producir descendencia. La hipótesis subyacente —conocida como el **building block hypothesis** (Goldberg, 1989)— es que los buenos "bloques" de genes de ambos padres pueden combinarse en hijos aún mejores.

El operador utilizado es el **crossover uniforme**:

- Para cada posición del cromosoma (cada ítem en MKP), se elige al azar de cuál padre heredar el bit
- Cada posición se decide independientemente (no hay puntos de corte como en el crossover de uno o dos puntos)

**¿Por qué uniforme y no de un punto?** El crossover de un punto preserva bloques contiguos de genes. En MKP, donde los ítems en el vector binario no tienen relación espacial (el ítem 5 y el ítem 6 no están necesariamente relacionados), no hay razón para preservar contigüidad. El crossover uniforme trata cada ítem de forma independiente, lo cual es más apropiado para problemas donde el orden no importa.

**El parámetro `crossover_rate`** controla la probabilidad de que una pareja de padres efectivamente produzca hijos por crossover (si no, los hijos son copias exactas de los padres):

- **Crossover rate alto (0.9)**: casi todas las parejas se recombinan. Los hijos son mezclas de ambos padres → **explotación** (combina bloques buenos existentes)
- **Crossover rate bajo (0.6)**: muchas parejas producen clones. Solo algunas se recombinan → **exploración** (mantiene individuos intactos, la diversidad viene de la mutación)

### 2.3 Mutación — ¿Qué introduce novedad?

La mutación es el operador que **inyecta diversidad**. Sin mutación, el GA solo podría explorar combinaciones de genes ya presentes en la población inicial. La mutación permite "descubrir" nuevos genes que no existían.

El operador es **bit-flip**: cada bit del cromosoma tiene una probabilidad independiente de invertirse (0→1, 1→0).

**El parámetro `mutation_rate`** es el más crítico para el balance exploración-explotación:

- **Mutación baja (0.01 = 1/n para n=100)**: solo ~1 bit cambia por individuo. Búsqueda local fina → **explotación**
- **Mutación alta (0.15)**: ~15 bits cambian por individuo (de 100). Saltos grandes en el espacio de búsqueda → **exploración**

La literatura clásica (Bäck, 1993) recomienda `mutation_rate = 1/n` (donde n es la longitud del cromosoma) como óptimo teórico para problemas sin conocimiento a priori. Nuestro modo exploit usa 0.01 ≈ 1/100, consistente con esta recomendación.

---

## 3. Elitismo — No Perder lo Mejor

El elitismo es una estrategia complementaria: los `k` mejores individuos de cada generación se **copian directamente** a la siguiente, sin pasar por selección, crossover ni mutación. Esto garantiza que el mejor fitness de la población **nunca empeore** (es monótonamente no-decreciente).

Con `elitism = 2`, los dos mejores individuos sobreviven intactos. Esto:

- Acelera la convergencia (el mejor no se pierde por mala suerte en el crossover)
- Reduce artificialmente la diversidad (los élites ocupan espacios que podrían ser para hijos diversos)
- Es **esencial en MKP** porque el operador de reparación puede degradar buenas soluciones

---

## 4. La Paradoja del GA: Selección vs. Diversidad

El GA vive una tensión permanente entre dos fuerzas opuestas:

```
SELECCIÓN                              MUTACIÓN
   ↓                                      ↓
"Que sobrevivan              "Que aparezcan cosas
 los más aptos"               que nunca existieron"
   ↓                                      ↓
Reduce diversidad                     Aumenta diversidad
   ↓                                      ↓
EXPLOTACIÓN                          EXPLORACIÓN
```

El crossover ocupa un lugar intermedio: puede crear combinaciones novedosas (exploración) pero solo a partir de genes ya existentes (explotación).

**El arte de calibrar un GA** está en encontrar la presión selectiva y la tasa de mutación que mantengan este balance. Demasiada selección + poca mutación = convergencia prematura. Demasiada mutación + poca selección = búsqueda aleatoria.

---

## 5. Parámetros para Modo Exploit y Modo Explore

### 5.1 Modo Exploit — Refinamiento Local

| Parámetro | Valor | Fundamento |
|---|---|---|
| **Crossover rate** | 0.9 | Alta probabilidad de recombinación. Los hijos heredan de ambos padres, combinando bloques buenos. Consistente con Goldberg (1989): crossover rates de 0.8-0.95 son estándar. |
| **Mutation rate** | 0.01 | ≈ 1/n para n=100. Bäck (1993) demostró que 1/n es óptimo para convergencia en problemas binarios. Solo ~1 bit cambia por individuo: perturbación mínima, búsqueda local. |

**Qué sucede en la práctica**: La población converge rápidamente. El crossover recombina soluciones buenas produciendo mejores. La mutación baja actúa como "fine-tuning": ajustes mínimos que refinan sin descarrilar. El elitismo preserva al mejor. El riesgo es la **convergencia prematura**: si todos los individuos se vuelven similares, el crossover se vuelve inútil (cruzar dos idénticos no produce nada nuevo) y la mutación baja no alcanza a diversificar.

### 5.2 Modo Explore — Diversificación Agresiva

| Parámetro | Valor | Fundamento |
|---|---|---|
| **Crossover rate** | 0.6 | Se reduce un 33%. Menos parejas se recombinan → más individuos pasan intactos a la siguiente generación. La diversidad se preserva porque no todos los individuos son "mezclados". |
| **Mutation rate** | 0.15 | Se multiplica por 15. Ahora ~15 bits cambian por individuo. Esto es una **exploración agresiva**: los individuos dan saltos grandes en el espacio de búsqueda, potencialmente escapando de óptimos locales. |

**Qué sucede en la práctica**: La población se diversifica violentamente. La mutación alta actúa como "motor de exploración": cada individuo es perturbado significativamente, visitando regiones lejanas del espacio de búsqueda. El crossover reducido evita que las buenas soluciones se "diluyan" al recombinarse con soluciones muy diferentes. El riesgo es **perder buenas soluciones**: la mutación agresiva puede degradar individuos excelentes. El elitismo se vuelve crucial aquí: los 2 mejores se preservan intactos como "ancla".

### 5.3 Tabla resumen

| Parámetro | Exploit | Explore | Cambio | Efecto del cambio |
|---|---|---|---|---|
| **Crossover rate** | 0.9 | 0.6 | −33% | Menos recombinación = más individuos intactos = más diversidad |
| **Mutation rate** | 0.01 | 0.15 | ×15 | Bits cambiados por individuo: ~1 → ~15. De refinamiento local a exploración global |
| **Torneo** | k=3 | k=3 | — | Sin cambios. Presión selectiva constante en ambos modos |
| **Elitismo** | 2 | 2 | — | Sin cambios. Preservar élites es crítico en modo explore |

---

## 6. GA frente a la Naturaleza del MKP

El MKP impone desafíos específicos que afectan cómo se comporta el GA:

### 6.1 El problema de la factibilidad

En MKP, una solución aleatoria tiene altísima probabilidad de ser **infactible** (violar restricciones de capacidad). El operador `reparar()` —que primero remueve ítems de baja densidad y luego agrega greedy— garantiza factibilidad, pero lo hace de forma **determinística**.

Para el GA, esto significa que:

- Dos individuos genéticamente distintos pueden volverse **idénticos** después de reparar
- La diversidad genotípica (bits diferentes) no siempre se traduce en diversidad fenotípica (fitness diferente)
- El crossover puede producir hijos que, tras reparar, son idénticos a uno de los padres → **crossover ineficaz**

### 6.2 El efecto meseta

El MKP tiene muchas soluciones factibles con **idéntico fitness** pero diferente estructura (diferentes combinaciones de ítems que suman el mismo profit). Para el GA, esto es un problema porque:

- La selección por torneo no puede distinguir entre dos individuos con igual fitness
- La población puede llenarse de soluciones equivalentes pero diversas en bits
- El DTW detecta esto como "estancamiento" (fitness no mejora) aunque la población sea genéticamente diversa

Esto justifica el uso de **mutation_rate alto** en modo explore: incluso si la población está en una meseta de fitness, la mutación fuerte puede encontrar la combinación de ítems que "rompa" la meseta hacia un fitness superior.

### 6.3 La correlación espacial (o su ausencia)

En algunos problemas, los bits cercanos en el cromosoma están relacionados (ej: características consecutivas de un diseño). En MKP, **no hay correlación espacial**. El ítem 5 y el ítem 6 son tan independientes como el ítem 5 y el ítem 500 (salvo que compartan restricciones).

Esto valida la elección de **crossover uniforme** sobre crossover de uno o dos puntos: no hay "bloques contiguos" que preservar.

---

## 7. ¿Por qué GA Complementa a PSO, GWO y DE?

| Aspecto | GA | PSO | GWO | DE |
|---|---|---|---|---|
| Mecanismo | Selección + recombinación + mutación | Atracción vectorial | Seguimiento de líderes | Diferencia vectorial |
| Fuente de diversidad | Mutación explícita (bit-flip) | Estocasticidad en velocidades | Parámetro `a` + aleatoriedad | Factor F sobre diferencias |
| Memoria | Población completa (sin memoria individual) | pbest + gbest | Alpha/Beta/Delta | Sin memoria (población actual) |
| Presión selectiva | Explícita (torneo) | Implícita (atracción al gbest) | Implícita (líderes) | Local (1-a-1 greedy) |
| Parámetros | 4+ (cx, mut, torneo, elitism) | 3 (w, c1, c2) | 1 (a) | 2 (F, CR) |
| Naturaleza de la convergencia | Generacional: toda la población se reemplaza | Gradual: partículas se mueven | Por colapso hacia líderes | Por selección individual |

### Lo que el GA aporta que los otros no tienen:

1. **Diversidad por mutación explícita**: es la única MH del estudio donde la exploración se logra mediante un operador diseñado específicamente para introducir ruido (bit-flip). En PSO, la exploración es un efecto secundario de la configuración de pesos; en GA, es el propósito explícito de la mutación.

2. **Selección competitiva global**: el torneo compara individuos de toda la población. Esto crea una presión selectiva más intensa que DE (selección local 1-a-1) o PSO (sin selección explícita). Es un arma de doble filo: acelera la convergencia pero arriesga pérdida de diversidad.

3. **Independencia del paisaje de atracción**: GA no asume que moverse "hacia" buenas soluciones es beneficioso. Simplemente recombina y muta. Esto lo hace más robusto en paisajes engañosos (donde moverse hacia el mejor actual te aleja del óptimo global), aunque potencialmente más lento en paisajes suaves.

4. **El operador de crossover como hipótesis testeable**: la building block hypothesis es una de las ideas más debatidas en computación evolutiva. Tener GA en el estudio permite evaluar si la recombinación sexual es realmente útil para MKP, o si la mutación sola (con selección) sería suficiente.

---

## 8. Referencias Clave

- **Fundación**: Holland, J.H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press. (Segunda edición: MIT Press, 1992).
- **El libro canónico**: Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.
- **Crossover uniforme**: Syswerda, G. (1989). "Uniform Crossover in Genetic Algorithms." *Proceedings of the 3rd International Conference on Genetic Algorithms*, 2-9.
- **Tasa de mutación óptima**: Bäck, T. (1993). "Optimal Mutation Rates in Genetic Search." *Proceedings of the 5th International Conference on Genetic Algorithms*, 2-8.
- **Tamaño de torneo**: Miller, B.L., & Goldberg, D.E. (1995). "Genetic Algorithms, Tournament Selection, and the Effects of Noise." *Complex Systems*, 9(3), 193-212.
- **Building block hypothesis**: Goldberg, D.E., Korb, B., & Deb, K. (1989). "Messy Genetic Algorithms: Motivation, Analysis, and First Results." *Complex Systems*, 3(5), 493-530.
- **GA para MKP (referencia del problema)**: Chu, P.C., & Beasley, J.E. (1998). "A Genetic Algorithm for the Multidimensional Knapsack Problem." *Journal of Heuristics*, 4(1), 63-86.
- **Elitismo**: De Jong, K.A. (1975). *An Analysis of the Behavior of a Class of Genetic Adaptive Systems*. Doctoral dissertation, University of Michigan.
