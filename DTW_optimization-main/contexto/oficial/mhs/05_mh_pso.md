# Binary Particle Swarm Optimization (PSO) — De cero a MKP

> **Referencia base**: Kennedy & Eberhart (1995, 1997) · Clerc & Kennedy (2002)  
> **Implementación**: `mkp_common/mh/pso.py` — `BinaryPSO`

---

## 1. ¿Qué es PSO y por qué es diferente?

**Particle Swarm Optimization** (Kennedy & Eberhart, 1995) es una metaheurística poblacional inspirada en el comportamiento social de bandadas de aves y cardúmenes de peces. A diferencia de los algoritmos evolutivos (GA, DE), PSO **no usa selección, crossover ni mutación**. En su lugar, cada partícula ajusta su trayectoria combinando tres fuerzas:

| MH | Cómo busca nuevas soluciones |
|----|------------------------------|
| **GA** | "Tomo dos padres buenos, mezclo sus genes, y muto un poco" |
| **PSO** | "Recuerdo dónde estuve mejor (pbest), miro dónde está el mejor del grupo (gbest), y vuelo hacia allá con inercia" |
| **GWO** | "Sigo al líder alpha, al beta, al delta, y promedio sus posiciones" |
| **DE** | "Tomo 3 individuos al azar, calculo la DIFERENCIA entre 2, y se la SUMO al tercero" |

La metáfora es simple y potente: cada partícula tiene **memoria individual** (su mejor posición personal, *pbest*) y **memoria social** (la mejor posición global del enjambre, *gbest*). En cada iteración, ajusta su velocidad como un compromiso entre:

1. **Inercia**: seguir en la dirección que traía
2. **Nostalgia**: volver hacia donde ella misma estuvo mejor
3. **Conformismo**: acercarse a donde el grupo estuvo mejor

Esta simplicidad es su mayor fortaleza: solo 3 parámetros controlan todo el comportamiento.

---

## 2. PSO Continuo Estándar

### 2.1 Las ecuaciones canónicas

Para cada partícula `i` en cada dimensión `j`:

```
v_ij(t+1) = w · v_ij(t)                       ← inercia
           + c₁ · r₁ · (pbest_ij − x_ij(t))   ← componente cognitivo
           + c₂ · r₂ · (gbest_j  − x_ij(t))   ← componente social

x_ij(t+1) = x_ij(t) + v_ij(t+1)
```

Donde:
- **`w`** — peso de inercia: cuánto de la velocidad anterior se conserva
- **`c₁`** — coeficiente cognitivo: atracción hacia la mejor posición personal
- **`c₂`** — coeficiente social: atracción hacia la mejor posición global
- **`r₁, r₂`** — vectores aleatorios ~ U(0,1) que introducen estocasticidad
- **`pbest_i`** — mejor posición visitada por la partícula `i`
- **`gbest`** — mejor posición visitada por cualquier partícula del enjambre

### 2.2 La intuición geométrica

Imaginá el espacio de búsqueda como un paisaje con colinas (fitness alto) y valles (fitness bajo). Cada partícula es un explorador con un mapa mental:

```
        pbest_i  ← "Mi mejor campamento hasta ahora"
       /
      /   c₁ (nostalgia)
     /
    ✈ ——→ dirección actual (inercia w)
     \
      \   c₂ (conformismo)
       \
        gbest  ← "El mejor campamento de toda la expedición"
```

La velocidad resultante es una **suma vectorial ponderada** de estas tres fuerzas. Los pesos `c₁` y `c₂` determinan si la partícula es más "individualista" (explora por su cuenta) o más "gregaria" (sigue al grupo).

### 2.3 v_max: el límite de velocidad

Para evitar que las partículas "exploten" (velocidades que crecen sin control), se aplica un clamping:

```
v_ij = clip(v_ij, −v_max, +v_max)
```

Con `v_max = 6.0`, típico en la literatura. Un `v_max` muy bajo restringe la exploración; muy alto permite divergencia.

---

## 3. Binary PSO (BPSO) — Kennedy & Eberhart 1997

Para problemas de optimización combinatoria como MKP, donde las soluciones son vectores binarios `x ∈ {0,1}ⁿ`, el PSO continuo no es directamente aplicable. Kennedy & Eberhart (1997) propusieron la versión binaria:

### 3.1 El problema de la discretización

En PSO continuo, la posición `x` se actualiza sumando la velocidad. En el espacio binario, `x + v` no tiene sentido. La solución: tratar la velocidad como una **probabilidad de cambio**.

### 3.2 La función sigmoide (transfer function)

```
P(x_ij = 1) = S(v_ij) = 1 / (1 + e^(−v_ij))
```

- Si `v_ij = 0` → `S(0) = 0.5` → 50% probabilidad de ser 0 o 1 (máxima incertidumbre)
- Si `v_ij → +∞` → `S(v) → 1` → casi seguro que es 1
- Si `v_ij → −∞` → `S(v) → 0` → casi seguro que es 0

La sigmoide mapea la velocidad continua a una probabilidad. Luego se muestrea:

```
x_ij = 1 si rand() < S(v_ij), sino 0
```

### 3.3 ¿Qué significa la velocidad en BPSO?

En el espacio binario, la velocidad ya no es "hacia dónde me muevo", sino **"cuán seguro estoy de que este bit debería ser 1"**:

- `v ≈ 0`: el bit es incierto, puede cambiar fácilmente
- `|v| grande`: el bit está "comprometido", es poco probable que cambie
- `v > 0`: tendencia a ser 1; `v < 0`: tendencia a ser 0

Esto es conceptualmente distinto del PSO continuo, pero preserva la misma mecánica de actualización de velocidad.

---

## 4. Los 3 Parámetros y Qué Controlan

PSO tiene exactamente 3 parámetros (más `v_max`). Su interpretación es directa:

### 4.1 `w` — Peso de Inercia

Controla cuánto de la velocidad anterior se conserva. Es el parámetro que más impacta el balance exploración-explotación.

| w | Comportamiento | Fuente |
|---|---|---|
| **w > 1.0** | Las partículas aceleran, divergen. Exploración extrema, puede ser inestable. | Engelbrecht (2005) |
| **w = 0.9** | Alta inercia. Las partículas mantienen su dirección, cubren más distancia. **Exploración**. | Shi & Eberhart (1998) — valor inicial típico |
| **w = 0.729** | Inercia moderada con constriction factor. Las partículas convergen suavemente. **Explotación**. | Clerc & Kennedy (2002) |
| **w < 0.4** | Baja inercia. Las partículas "frenan" rápido, búsqueda local fina. | Shi & Eberhart (1998) — valor final típico |

```
w alto (0.9):
  ✈ → → → → →  "Mantengo el rumbo, exploro lejos"
  Bueno para: escapar de óptimos locales, cubrir el espacio de búsqueda

w bajo (0.729 con constriction):
  ✈ → → . "Ajusto el rumbo constantemente, refino la posición"
  Bueno para: convergencia fina alrededor de buenas soluciones
```

### 4.2 `c₁` — Coeficiente Cognitivo (nostalgia)

Peso de la atracción hacia el mejor personal (*pbest*).

```
c₁ alto (2.5):
  "Confío en MI experiencia. No me importa lo que diga el grupo."
  → Cada partícula explora su propia vecindad → DIVERSIDAD

c₁ moderado (1.49):
  "Equilibro mi experiencia con la del grupo."
  → Balance entre individualismo y gregarismo

c₁ bajo (< 1.0):
  "Mi experiencia personal no es tan importante."
  → Las partículas ignoran su historia → CONVERGENCIA al gbest
```

### 4.3 `c₂` — Coeficiente Social (conformismo)

Peso de la atracción hacia el mejor global (*gbest*).

```
c₂ alto (> 2.0):
  "Sigo ciegamente al líder."
  → Todas las partículas convergen al gbest → EXPLOTACIÓN
  → Riesgo: convergencia prematura si gbest es un óptimo local

c₂ moderado (1.49):
  "Escucho al grupo pero no dejo de pensar por mí mismo."
  → Balance

c₂ bajo (0.5):
  "Desconfío del líder. Prefiero explorar por mi cuenta."
  → Las partículas ignoran al gbest → EXPLORACIÓN
  → Clave cuando el gbest está estancado en un óptimo local
```

### 4.4 La interacción c₁ × c₂

| | c₂ bajo (social débil) | c₂ alto (social fuerte) |
|---|---|---|
| **c₁ alto** (cognitivo fuerte) | **Exploración máxima**: cada partícula sigue su propio camino, ignora al grupo | Búsqueda dispersa con atracción al grupo |
| **c₁ bajo** (cognitivo débil) | Búsqueda sin memoria ni guía — errática | **Convergencia al gbest**: todas las partículas colapsan al líder |

---

## 5. Parámetros para Modo Exploit y Modo Explore

### 5.1 Modo Exploit — Clerc & Kennedy (2002) Constriction Factor

| Parámetro | Valor | Justificación |
|---|---|---|
| **w** | 0.729 | Constriction factor χ. Demostrado matemáticamente que garantiza convergencia (Clerc & Kennedy, 2002, *IEEE TEC*) |
| **c₁** | 1.49445 | χ × φ₁ donde φ₁ = 2.05. Balance óptimo con constriction |
| **c₂** | 1.49445 | χ × φ₂ donde φ₂ = 2.05. Simétrico con c₁ |

**Intuición**: Con estos valores, el sistema es **convergente por construcción**. Las partículas se atraen mutuamente hacia la región del gbest, refinando la solución. Es el "gold standard" para explotación en PSO.

**Ecuación completa del constriction factor** (Clerc & Kennedy, 2002):
```
χ = 2 / |2 − φ − √(φ² − 4φ)|,  donde φ = φ₁ + φ₂ > 4

Con φ₁ = φ₂ = 2.05 → φ = 4.1 → χ ≈ 0.729
```

### 5.2 Modo Explore — Forzar Diversificación

| Parámetro | Valor | Justificación |
|---|---|---|
| **w** | 0.9 | Valor inicial del esquema de inercia decreciente de Shi & Eberhart (1998). Representa la fase de exploración del PSO clásico. Con w = 0.9, las partículas mantienen ∼90% de su velocidad anterior, cubriendo más distancia por iteración. |
| **c₁** | 2.5 | Significativamente mayor que el estándar (2.0). Cada partícula prioriza fuertemente su propia historia sobre la del grupo. Esto genera **diversidad**: cada partícula explora una región distinta alrededor de su pbest. |
| **c₂** | 0.5 | Significativamente menor que el estándar (2.0). Las partículas **desconfían del gbest**, que presumiblemente está estancado. Esto evita que todo el enjambre colapse a un óptimo local. |

**Intuición**: Cuando el DTW detecta estancamiento, el enjambre está atrapado: el gbest no mejora y todas las partículas convergieron hacia él. La respuesta es **romper el consenso**: cada partícula debe seguir su propio camino (c₁ alto), ignorar al líder estancado (c₂ bajo), y mantener el impulso para alejarse (w alto).

### 5.3 Tabla resumen

| Parámetro | Exploit | Explore | Δ | Efecto del cambio |
|---|---|---|---|---|
| **w** (inercia) | 0.729 | 0.9 | +23% | Más inercia = más exploración, partículas "vuelan" más lejos |
| **c₁** (cognitivo) | 1.49 | 2.5 | +68% | Más peso a pbest = cada partícula explora su propia zona |
| **c₂** (social) | 1.49 | 0.5 | −66% | Menos peso a gbest = el enjambre se desacopla del líder estancado |

---

## 6. Binarización y Operador de Reparación

### 6.1 Flujo completo de una iteración BPSO

```
PARA cada partícula i:
    1. Actualizar velocidad:
       v_i = w·v_i + c₁·r₁·(pbest_i − x_i) + c₂·r₂·(gbest − x_i)
       v_i = clip(v_i, −6, 6)

    2. Binarizar con sigmoid:
       prob = 1 / (1 + e^(−v_i))
       x_i[j] = 1 si rand() < prob[j], sino 0

    3. Reparar factibilidad MKP:
       x_i = reparar(x_i, instancia)

    4. Evaluar fitness:
       fit = Σ p_j · x_i[j]

    5. Actualizar memorias:
       si fit > fit_pbest_i → pbest_i = x_i
       si fit > fit_gbest   → gbest   = x_i
```

### 6.2 El operador `reparar()` y su efecto en PSO

El operador de reparación es **determinístico**: ante dos soluciones infactibles iguales, produce la misma salida. Para PSO esto tiene una consecuencia importante: si dos partículas convergen a posiciones similares, la reparación las hace idénticas, **reduciendo la diversidad real** del enjambre más allá de lo que las velocidades sugerirían.

Esto refuerza la necesidad de que el modo explore use c₂ bajo: si el gbest está en un óptimo local y todas las partículas son atraídas hacia él, la reparación colapsa la diversidad rápidamente.

---

## 7. Pseudocódigo Completo

```
ENTRADA: instancia MKP, num_particulas, max_iter, w, c1, c2, v_max
SALIDA: mejor solución encontrada

1.  INICIALIZAR:
      para cada partícula i:
        x_i = vector binario aleatorio
        reparar(x_i)
        v_i = aleatorio en [-v_max, v_max]
        pbest_i = x_i
      gbest = argmax(fitness(pbest))

2.  PARA iter = 1 hasta max_iter:
3.      PARA cada partícula i:
4.
5.          // Actualizar velocidad
6.          v_i = w·v_i + c1·rand()·(pbest_i − x_i) + c2·rand()·(gbest − x_i)
7.          v_i = clip(v_i, −v_max, +v_max)
8.
9.          // Binarizar
10.         prob = sigmoid(v_i)
11.         x_i = (rand() < prob)
12.
13.         // Reparar y evaluar
14.         x_i, fit = reparar(x_i, instancia)
15.
16.         // Actualizar pbest
17.         si fit > fitness(pbest_i):
18.             pbest_i = x_i
19.
20.         // Actualizar gbest
21.         si fit > fitness(gbest):
22.             gbest = x_i
23.
24. RETORNAR gbest
```

---

## 8. Mapeo a la Interfaz BaseMH

La implementación en `BinaryPSO` expone los parámetros como constantes de clase, permitiendo que el runner externo (DTW) controle el modo:

```
Parámetros estáticos por modo:
  EXPLOIT: w=0.729,  c1=1.49445, c2=1.49445
  EXPLORE: w=0.9,    c1=2.5,     c2=0.5

Interfaz:
  adapt(fire: bool):
    fire=True  → cambiar a EXPLORE (w↑, c1↑, c2↓)
    fire=False → volver a EXPLOIT

  adapt_continuous(intensity ∈ [0,1]):
    w  = 0.729 + intensity × (0.9   − 0.729)
    c1 = 1.49  + intensity × (2.5   − 1.49)
    c2 = 1.49  + intensity × (0.5   − 1.49)
    (interpolación lineal entre los extremos)
```

---

## 9. ¿Por qué PSO complementa bien a GA, GWO y DE?

| Aspecto | GA | PSO | GWO | DE |
|---|---|---|---|---|
| Mecanismo de búsqueda | Recombinación genética | Velocidad + atracción social | Jerarquía de líderes | Diferencia de vectores |
| Memoria | No (solo población actual) | Sí (pbest individual + gbest global) | Solo líderes (alpha, beta, delta) | No (población actual) |
| Presión selectiva | Torneo (explícita, competitiva) | Implícita (pbest/gbest funcionan como élites) | Implícita (líderes son élites) | Greedy local 1-a-1 |
| Parámetros libres | 4+ (cx, mut, torneo, elitism) | 3 (w, c1, c2) | 1 (a) | 2 (F, CR) |
| Naturaleza de la convergencia | Por remplazo generacional | Por atracción vectorial (suave) | Por seguimiento de líderes | Por diferencias que se achican |
| Exploit/Explore | mut_rate ↑, cx_rate ↓ | c1↑ y c2↓ para explorar; inverso para explotar | a: 2→0 naturalmente | F y CR |

### Lo que PSO aporta al estudio que los otros no tienen:

1. **Convergencia por atracción, no por selección**: En PSO, las soluciones no "mueren". Todas las partículas sobreviven y se adaptan. Esto produce curvas de convergencia más suaves que GA (donde hay reemplazo generacional) o DE (donde hay selección greedy).

2. **Memoria dual**: Cada partícula recuerda su propio mejor (pbest) y el mejor global (gbest). Esto crea un **paisaje de atracción** más rico que el de GWO (solo 3 líderes) o GA (sin memoria).

3. **El constriction factor como baseline teórico**: Los valores de Clerc & Kennedy (2002) no son heurísticos — tienen una **demostración matemática de convergencia**. Esto le da solidez teórica a nuestro modo exploit que ningún otro algoritmo del estudio puede reclamar.

4. **Sensibilidad a c₂**: El coeficiente social es el parámetro más "interruptor" del estudio. Pasarlo de 1.49 a 0.5 esencialmente le dice al enjambre "ignorá al líder". Esto es una palanca de exploración extremadamente efectiva y fácil de interpretar.

---

## 10. Referencias Clave

- **PSO original**: Kennedy, J., & Eberhart, R.C. (1995). "Particle Swarm Optimization." *Proceedings of IEEE International Conference on Neural Networks*, 1942-1948.
- **Binary PSO**: Kennedy, J., & Eberhart, R.C. (1997). "A Discrete Binary Version of the Particle Swarm Algorithm." *IEEE International Conference on Systems, Man, and Cybernetics*, 4104-4108.
- **Inertia weight**: Shi, Y., & Eberhart, R.C. (1998). "A Modified Particle Swarm Optimizer." *IEEE International Conference on Evolutionary Computation*, 69-73.
- **Constriction factor**: Clerc, M., & Kennedy, J. (2002). "The Particle Swarm — Explosion, Stability, and Convergence in a Multidimensional Complex Space." *IEEE Transactions on Evolutionary Computation*, 6(1), 58-73.
- **PSO convergence analysis**: van den Bergh, F., & Engelbrecht, A.P. (2006). "A Study of Particle Swarm Optimization Particle Trajectories." *Information Sciences*, 176(8), 937-971.
- **Self-adaptive PSO**: Ratnaweera, A., Halgamuge, S.K., & Watson, H.C. (2004). "Self-Organizing Hierarchical Particle Swarm Optimizer with Time-Varying Acceleration Coefficients." *IEEE Transactions on Evolutionary Computation*, 8(3), 240-255.
- **BPSO for knapsack**: Bansal, J.C., & Deep, K. (2012). "A Modified Binary Particle Swarm Optimization for Knapsack Problems." *Applied Mathematics and Computation*, 218(22), 11042-11061.
