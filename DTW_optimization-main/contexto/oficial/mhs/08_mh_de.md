# Differential Evolution (DE) para MKP — Fundamentos y Conceptos

> **Referencia base**: Storn & Price (1997) · Das & Suganthan (2011)  
> **Variante utilizada**: DE/rand/1/bin binarizado con sigmoid transfer function

---

## 1. ¿Qué es DE y por qué es radicalmente distinto?

**Differential Evolution** (Storn & Price, 1997) es un algoritmo evolutivo poblacional para optimización en espacios continuos. A primera vista parece "otro GA", pero su mecanismo de búsqueda es **fundamentalmente distinto**: en vez de cruzar padres o seguir líderes, **genera nuevas soluciones sumando diferencias entre vectores de la población**.

| MH | Cómo genera nuevas soluciones |
|---|---|
| **GA** | "Selecciono dos padres, mezclo sus genes (crossover), y muto un poco" |
| **PSO** | "Ajusto mi velocidad según mi mejor pasado (pbest) y el mejor del grupo (gbest)" |
| **GWO** | "Promedio las direcciones hacia mis tres líderes (α, β, δ)" |
| **DE** | "Tomo tres individuos al azar, calculo la DIFERENCIA entre dos, y se la SUMO al tercero" |

La clave está en esa **diferencia entre vectores**. Es un mecanismo auto-escalante: cuando la población es diversa, las diferencias son grandes → DE explora con pasos largos. Cuando la población converge, las diferencias se achican → DE explota con pasos cortos. La magnitud de la exploración **emerge naturalmente de la estructura de la población**, sin necesidad de un parámetro externo de escala.

---

## 2. Los Tres Operadores de DE

DE aplica tres operadores a **cada individuo** en cada iteración. El orden es fijo e innegociable: mutación → crossover → selección.

### 2.1 Mutación Diferencial — El Corazón de DE

Para cada individuo `x_i` (llamado *target vector*):

1. Se eligen **tres individuos distintos al azar**: `x_r1`, `x_r2`, `x_r3` (todos ≠ `x_i`)
2. Se calcula el **vector mutante**:

```
v_i = x_r1 + F · (x_r2 − x_r3)
```

Donde `F ∈ [0, 2]` es el **factor de escala**.

**¿Qué está ocurriendo matemáticamente?** Se toma un individuo base (`x_r1`) y se le suma una **perturbación vectorial** proporcional a la diferencia entre otros dos (`x_r2 − x_r3`). La dirección de la perturbación es aleatoria (porque `r2` y `r3` son aleatorios), pero su **magnitud** depende de cuán dispersa esté la población:

```
Población diversa (fase inicial):
  x_r2 = [0.8, 0.1, 0.9, 0.3, ...]
  x_r3 = [0.1, 0.9, 0.2, 0.7, ...]
  diferencia = [0.7, −0.8, 0.7, −0.4, ...]  ← GRANDE → exploración

Población convergida (fase final):
  x_r2 = [0.51, 0.49, 0.52, 0.48, ...]
  x_r3 = [0.49, 0.51, 0.50, 0.50, ...]
  diferencia = [0.02, −0.02, 0.02, −0.02, ...]  ← CHICA → explotación
```

Esta propiedad se llama **auto-escalado** (*self-scaling*) y es la contribución teórica más elegante de DE. A diferencia de PSO (donde se necesita `v_max` para evitar divergencia) o GA (donde la mutación es ruido externo), en DE la **intensidad de búsqueda se autorregula** con la diversidad poblacional.

### 2.2 Crossover Binomial — ¿Qué hereda del mutante?

El vector mutante `v_i` se cruza con el vector original `x_i` para crear un **vector de prueba** `u_i`:

```
Para cada dimensión j:
    si rand() < CR  o  j == j_rand:
        u_i[j] = v_i[j]    ← hereda del mutante
    si no:
        u_i[j] = x_i[j]    ← conserva del original
```

Donde:
- **`CR ∈ [0, 1]`** es la tasa de crossover: qué fracción de dimensiones vienen del mutante
- **`j_rand`** es una dimensión elegida al azar que **siempre** viene del mutante (garantiza que `u_i ≠ x_i`)

**¿Binomial?** El término "binomial" refiere a que el número de dimensiones heredadas del mutante sigue aproximadamente una distribución binomial con parámetro `CR`. No tiene nada que ver con representación binaria (0/1).

**Intuición del crossover**:

- **CR alto (0.9)**: casi todo viene del mutante → el vector de prueba es muy distinto al original → cambio grande → **explotación** (en el sentido de DE: probar activamente nuevas combinaciones)
- **CR bajo (0.3)**: pocas dimensiones cambian → el vector de prueba es muy similar al original → cambio pequeño → **exploración** (en el sentido de DE: preservar la estructura del individuo)

> **Cuidado con la inversión semántica**: en DE, CR alto produce cambios grandes (porque el mutante domina), lo cual puede parecer "exploración". Pero en la literatura de DE, CR alto se asocia con **explotación** porque el algoritmo confía en la mutación diferencial para generar candidatos radicalmente nuevos. La exploración en DE se logra con CR bajo + F alto, donde la mutación es agresiva pero solo se aplica a pocas dimensiones.

### 2.3 Selección Greedy 1-a-1 — Competencia Individual

La selección en DE es **determinística, elitista y local**: el vector de prueba `u_i` compite **exclusivamente contra su padre** `x_i`:

```
si fitness(u_i) ≥ fitness(x_i):
    x_i = u_i      ← el hijo reemplaza al padre
si no:
    x_i = x_i      ← el padre sobrevive
```

Esto es radicalmente distinto de:
- **GA**: selección por torneo entre toda la población (competencia global)
- **PSO**: no hay selección; las partículas simplemente se mueven
- **GWO**: no hay selección; los lobos se actualizan por promedio de líderes

La selección 1-a-1 tiene una consecuencia profunda: **no hay presión selectiva global**. Un individuo mediocre puede sobrevivir indefinidamente mientras sus propios hijos no lo superen. Esto preserva diversidad de forma natural: no hay un "mejor global" que acapare la población.

---

## 3. La Notación DE/x/y/z

DE tiene una nomenclatura estándar que describe su configuración:

```
DE / base / num_diferencias / crossover_type
```

| Componente | Significado | Opciones comunes |
|---|---|---|
| **base** | Qué vector se perturba | `rand` (aleatorio), `best` (mejor global), `current-to-best` |
| **num_diferencias** | Cuántos pares de diferencia se usan | `1` (un par), `2` (dos pares sumados) |
| **crossover_type** | Tipo de recombinación | `bin` (binomial), `exp` (exponencial) |

### Variantes principales

**DE/rand/1/bin** (la utilizada en este estudio):
```
v_i = x_r1 + F · (x_r2 − x_r3)
```
- Base aleatoria, un par de diferencias, crossover binomial
- **Muy exploratoria**: el vector base es aleatorio → no hay sesgo hacia el mejor

**DE/best/1/bin**:
```
v_i = x_best + F · (x_r1 − x_r2)
```
- Base = mejor global
- **Muy explotatoria**: converge rápido hacia el mejor, pero arriesga convergencia prematura

**DE/current-to-best/1/bin**:
```
v_i = x_i + F · (x_best − x_i) + F · (x_r1 − x_r2)
```
- El individuo se mueve hacia el mejor + perturbación aleatoria
- **Balanceada**: combina atracción al mejor con diversidad

### ¿Por qué DE/rand/1/bin para este estudio?

Porque queremos que el **DTW controle completamente** el balance exploración-explotación. DE/rand/1/bin, al usar base aleatoria, no tiene sesgo inherente hacia el mejor global. Esto pone toda la responsabilidad de la adaptación en el DTW: la señal externa decide cuándo explorar (F↑, CR↓) y cuándo explotar (F↓, CR↑), sin interferencia de un sesgo de "seguir al mejor" cableado en el algoritmo.

---

## 4. DE Binario para MKP

### 4.1 El problema: DE opera en ℝⁿ, MKP requiere {0,1}ⁿ

DE fue diseñado para optimización continua. La mutación diferencial (`x_r2 − x_r3`) y el crossover binomial asumen que las variables son números reales. Para aplicar DE a MKP, necesitamos un puente entre ambos espacios.

### 4.2 Representación dual

La solución: cada individuo mantiene **dos representaciones paralelas**:

| Representación | Espacio | Propósito |
|---|---|---|
| **Continua** | ℝⁿ | Donde opera la aritmética de DE (mutación, crossover) |
| **Binaria** | {0,1}ⁿ | Donde se evalúa el fitness en MKP |

El ciclo de vida de cada individuo es:

```
x_cont (ℝⁿ) → mutación + crossover → u_cont (ℝⁿ) → sigmoid → u_bin ({0,1}ⁿ) → reparar → evaluar fitness
```

La representación continua nunca se evalúa directamente; solo existe como "lienzo" para que los operadores de DE trabajen. La sigmoide es el traductor: convierte cada valor continuo en una probabilidad de que el ítem correspondiente esté en la solución.

### 4.3 La sigmoide como puente

```
prob[j] = 1 / (1 + e^(−u_cont[j]))
x_bin[j] = 1 si rand() < prob[j], sino 0
```

Un valor continuo muy positivo → alta probabilidad de incluir el ítem. Un valor muy negativo → alta probabilidad de excluirlo. Un valor cercano a 0 → 50% de probabilidad (máxima incertidumbre, el ítem "está en debate").

---

## 5. Los Parámetros F y CR

DE tiene solo **dos parámetros libres** (excluyendo tamaño de población). Su interpretación es directa pero su interacción es sutil.

### 5.1 F — Factor de Escala (el acelerador)

Controla cuánto se **amplifica** la diferencia entre vectores:

```
F bajo (0.5):
  v_i = x_r1 + 0.5 · (x_r2 − x_r3)
  → La perturbación es atenuada. Pasos cortos.
  → EXPLOTACIÓN: búsqueda local alrededor del vector base.

F alto (0.9):
  v_i = x_r1 + 0.9 · (x_r2 − x_r3)
  → La perturbación es casi completa. Pasos largos.
  → EXPLORACIÓN: saltos grandes hacia regiones distantes.
```

**El rango efectivo**: F ∈ [0.4, 1.0] es lo recomendado en la literatura. F < 0.4 produce muy poca perturbación (el algoritmo se vuelve casi aleatorio). F > 1.0 puede causar divergencia (aunque en algunos problemas es beneficioso).

### 5.2 CR — Tasa de Crossover (el mezclador)

Controla qué fracción de dimensiones del vector de prueba provienen del mutante:

```
CR alto (0.9):
  ~90% de las dimensiones vienen del mutante
  → El vector de prueba es casi un clon del mutante
  → Cambios masivos y coordinados
  → EXPLOTACIÓN (en el sentido de DE): confía en la mutación diferencial

CR bajo (0.3):
  ~30% de las dimensiones vienen del mutante
  → El vector de prueba conserva ~70% del original
  → Cambios quirúrgicos: pocas dimensiones, pero cada una puede cambiar mucho (por F alto)
  → EXPLORACIÓN (en el sentido de DE): preserva estructura, perturba quirúrgicamente
```

### 5.3 La interacción F × CR

| | CR bajo (0.3) | CR alto (0.9) |
|---|---|---|
| **F bajo (0.5)** | Explotación máxima: pocos cambios, pasos cortos. Búsqueda local fina. | Mezcla agresiva de pasos cortos. Muchas dimensiones cambian un poco. |
| **F alto (0.9)** | **Exploración máxima**: pocos cambios pero cada uno es un salto grande. Perturbación quirúrgica. | Cambios masivos y agresivos. Puede ser demasiado disruptivo (casi reinicio). |

### 5.4 El auto-escalado atenúa la necesidad de F preciso

Una propiedad notable de DE: incluso con F fijo, la **magnitud real** de la perturbación se adapta porque `x_r2 − x_r3` se achica cuando la población converge. Esto significa que:

- Con **F = 0.5**: cuando la población es diversa, las diferencias son grandes → perturbación moderada. Cuando converge, las diferencias son chicas → perturbación fina. El auto-escalado hace que F=0.5 sea "explotación" en la práctica, incluso sin cambiarlo.
- Con **F = 0.9**: la perturbación siempre es agresiva, pero aún se atenúa naturalmente cuando la población converge.

Esto hace que DE sea **intrínsecamente balanceado**: incluso sin DTW, tiende a explorar al inicio (población diversa → diferencias grandes) y explotar al final (población convergida → diferencias chicas). El DTW acelera y refina este proceso al forzar F↑ y CR↓ cuando detecta que el balance natural no es suficiente.

---

## 6. Parámetros para Modo Exploit y Modo Explore

### 6.1 Modo Exploit — Refinamiento por Consenso

| Parámetro | Valor | Fundamento |
|---|---|---|
| **F** | 0.5 | Perturbación moderada. El auto-escalado de DE la atenúa aún más cuando la población converge. Storn & Price (1997) recomiendan F ∈ [0.5, 1.0] como rango efectivo. |
| **CR** | 0.9 | Alta tasa de crossover. El vector de prueba hereda casi todo del mutante. Combinado con F=0.5, los cambios son frecuentes pero de baja magnitud → refinamiento. |

**Qué sucede**: La mutación diferencial produce perturbaciones suaves (F=0.5 atenúa las diferencias). El crossover alto (CR=0.9) propaga estos cambios a casi todas las dimensiones. El resultado es una **búsqueda local intensiva**: los individuos exploran minuciosamente la vecindad de las soluciones actuales, refinándolas progresivamente.

### 6.2 Modo Explore — Perturbación Quirúrgica

| Parámetro | Valor | Fundamento |
|---|---|---|
| **F** | 0.9 | Perturbación cercana al máximo recomendado. Las diferencias entre vectores se amplifican casi completamente, generando vectores mutantes muy distintos a la base. |
| **CR** | 0.3 | Baja tasa de crossover. Solo ~30% de las dimensiones cambian por iteración. Esto **concentra** la perturbación agresiva en pocas dimensiones, produciendo cambios quirúrgicos. |

**Qué sucede**: La mutación diferencial es agresiva (F=0.9), pero el crossover bajo (CR=0.3) la enfoca. En vez de cambiar muchas dimensiones un poco, DE cambia **pocas dimensiones pero mucho**. Esto es exploración en el sentido más puro: el individuo da un salto grande en una dirección específica del espacio de búsqueda, preservando el resto de su estructura. Es como decir: "de todo lo que sé, voy a cuestionar profundamente solo una pequeña parte".

### 6.3 Tabla resumen

| Parámetro | Exploit | Explore | Cambio | Efecto del cambio |
|---|---|---|---|---|
| **F** (mutación) | 0.5 | 0.9 | +80% | De perturbación suave a agresiva |
| **CR** (crossover) | 0.9 | 0.3 | −67% | De cambio generalizado a cambio focalizado |
| **Estrategia** | Muchos cambios pequeños | Pocos cambios grandes |

---

## 7. DE frente a la Naturaleza del MKP

### 7.1 El operador de reparación y la representación dual

La representación dual (continua + binaria) de DE interactúa de forma peculiar con el operador de reparación determinístico. Cuando un vector de prueba supera a su padre en fitness, **ambas** representaciones se actualizan: la continua (para futuras mutaciones) y la binaria (para evaluar fitness).

Pero la reparación puede hacer que dos vectores continuos muy diferentes produzcan la misma solución binaria después de reparar. Esto crea una situación donde:
- La población **continua** es diversa (bueno para DE)
- La población **binaria** está colapsada (malo para el fitness)
- El DTW detecta estancamiento por el fitness, pero DE "cree" que hay diversidad

Esto justifica F alto en modo explore: necesitamos forzar perturbaciones tan grandes que **atraviesen** la zona de colapso de la reparación.

### 7.2 La selección 1-a-1 y las mesetas de fitness

En MKP, muchas soluciones factibles diferentes pueden tener exactamente el mismo fitness. La selección greedy 1-a-1 de DE, ante un empate (`fitness(u_i) == fitness(x_i)`), reemplaza al padre. Esto significa que DE **sí cambia** en mesetas de fitness — a diferencia de lo que podría parecer.

Esta propiedad es sutil pero importante: incluso cuando el DTW detecta estancamiento (fitness no mejora), DE puede estar **reconfigurando** la población hacia regiones del espacio de búsqueda que, aunque aún no producen mejor fitness, son más prometedoras. Es como "preparar el terreno" silenciosamente.

---

## 8. ¿Por qué DE Complementa a GA, PSO y GWO?

| Aspecto | GA | PSO | GWO | DE |
|---|---|---|---|---|
| Mecanismo | Selección + crossover + mutación | Atracción vectorial | Seguimiento de líderes | Diferencia vectorial |
| Fuente de diversidad | Mutación externa (bit-flip) | Estocasticidad en velocidades | Aleatoriedad en C | **Auto-escalado poblacional** |
| Selección | Torneo (competitiva) | Implícita (pbest/gbest) | Implícita (líderes) | **Greedy local 1-a-1** |
| Parámetros | 4+ | 3 | 1 | **2** |
| Memoria | Población actual | pbest + gbest | α, β, δ | Población actual |
| Representación | Solo binaria | Solo binaria (con velocidades) | Solo binaria | **Dual (continua + binaria)** |
| Convergencia | Generacional | Gradual (vectorial) | Por colapso a líderes | **Individual (1-a-1)** |

### Lo que DE aporta que los otros no tienen:

1. **Auto-escalado natural**: es la **única MH del estudio cuya intensidad de búsqueda se adapta sin intervención externa**. Cuando la población converge, las diferencias se achican y DE automáticamente pasa a búsqueda local fina. El DTW no necesita forzar este comportamiento — simplemente lo acelera o lo contrarresta cuando es necesario.

2. **Representación dual**: DE es la única MH que mantiene un espacio de trabajo continuo separado del espacio de evaluación binario. Esto le da una "memoria de navegación" que las otras no tienen: puede moverse en direcciones que temporalmente producen peor fitness binario pero que geométricamente son prometedoras.

3. **Selección sin presión global**: a diferencia del torneo del GA o la atracción al gbest del PSO, la selección 1-a-1 de DE no fuerza la convergencia de toda la población hacia el mejor. Esto preserva diversidad de forma **estructural**, no por parámetros. Es la MH naturalmente más resistente a la convergencia prematura.

4. **Geometría diferencial como información**: DE es la única MH que explota la **estructura geométrica** de la población (distancias y direcciones entre individuos) como fuente de información para la búsqueda. GA trata a los individuos como bolsas de genes independientes; PSO y GWO solo miran hacia los líderes. DE mira **las relaciones entre todos los individuos**.

---

## 9. Referencias Clave

- **DE original**: Storn, R., & Price, K. (1997). "Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces." *Journal of Global Optimization*, 11(4), 341-359.
- **Survey definitivo**: Das, S., & Suganthan, P.N. (2011). "Differential Evolution: A Survey of the State-of-the-Art." *IEEE Transactions on Evolutionary Computation*, 15(1), 4-31.
- **DE para optimización binaria**: Pampara, G., Engelbrecht, A.P., & Franken, N. (2006). "Binary Differential Evolution." *IEEE Congress on Evolutionary Computation*, 1873-1879.
- **Selección de parámetros F y CR**: Gämperle, R., Müller, S.D., & Koumoutsakos, P. (2002). "A Parameter Study for Differential Evolution." *WSEAS International Conference on Advances in Intelligent Systems*, 293-298.
- **DE adaptativo (JADE)**: Zhang, J., & Sanderson, A.C. (2009). "JADE: Adaptive Differential Evolution with Optional External Archive." *IEEE Transactions on Evolutionary Computation*, 13(5), 945-958.
- **DE para knapsack**: Tasgetiren, M.F., Liang, Y.C., Sevkli, M., & Gencyilmaz, G. (2006). "Differential Evolution Algorithm for Permutation Flowshop Sequencing Problem with Makespan Criterion." *International Journal of Production Research*. (Adaptable a MKP).
- **Análisis del auto-escalado**: Zaharie, D. (2009). "Influence of Crossover on the Behavior of Differential Evolution Algorithms." *Applied Soft Computing*, 9(3), 1126-1138.
