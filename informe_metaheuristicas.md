# Informe Técnico: Metaheurísticas para el Problema de la Mochila Multidimensional (MKP)

Este documento detalla todas las implementaciones algorítmicas realizadas en el proyecto para resolver el Problema de la Mochila Multidimensional (MKP). El enfoque central se basa en la aplicación de tres metaheurísticas (Recocido Simulado, Algoritmo Genético y Búsqueda Tabú) apoyadas por un monitor avanzado de estancamiento basado en *Dynamic Time Warping (DTW)* y un catálogo unificado de variantes de escape.

---

## 1. Núcleo Compartido (`mkp_core`)

Independientemente del algoritmo que se utilice, todos interactúan de forma consistente con las instancias del problema.

### 1.1 Modelo del Problema (`problem.py`)
La clase `MKPInstance` encapsula los datos del problema (ganancias `p`, pesos `r` y capacidades `b`). Durante la inicialización, calcula la **densidad mínima** de cada ítem ($min_i(p_j / r_{i,j})$). Esto determina un ordenamiento (índices ascendentes y descendentes) que guía la heurística constructiva y destructiva a lo largo de todos los algoritmos.

### 1.2 Función de Reparación (`repair.py`)
Cualquier movimiento exploratorio o perturbación (ej. mutaciones o cruces) puede generar soluciones infactibles que violen la capacidad. La función `reparar_solucion()` asegura la factibilidad en dos fases (es un operador *puramente determinista*):
1. **Fase de Expulsión:** Si la mochila excede alguna capacidad, extrae ítems uno por uno, comenzando por los de **menor densidad**, hasta que se cumplan todas las restricciones.
2. **Fase de Inserción:** Luego, intenta agregar ítems en orden descendente, comenzando por los de **mayor densidad**, maximizando la ganancia restante sin romper la factibilidad.

---

## 2. Monitor de Estancamiento (`dtw_stagnation.py`)

Para evitar quedar atrapado en óptimos locales, todas las metaheurísticas incorporan un `StagnationMonitor` que funciona como un observador externo del historial de la búsqueda.

### ¿Cómo funciona?
En lugar de depender únicamente de una "cantidad máxima de iteraciones sin mejora", el monitor aplica **DTW (Dynamic Time Warping)** —o su derivada, DDTW— para comparar el comportamiento reciente de la curva de mejores objetivos contra dos "señales base":
- **Una rampa ascendente:** Representa un algoritmo explorando y encontrando mejoras.
- **Una línea constante:** Representa un claro estancamiento (meseta).

El sistema calcula una "distancia DTW" contra la línea constante ($D_2$) y contra la rampa ($D_1$). Si la curva reciente de la metaheurística se parece mucho a la constante (distancia pequeña) y muy poco a la rampa (distancia grande), y además han pasado $N$ épocas sin una mejoría clara, el monitor "dispara" (*fires*). Ese disparo gatilla dinámicamente un **rescate**.

---

## 3. Implementaciones de Metaheurísticas (MH)

### 3.1. Recocido Simulado / Simulated Annealing (`sa_mkp`)
**Estrategia:** Un algoritmo de búsqueda local probabilística que simula el enfriamiento de un material. A altas temperaturas, acepta peores soluciones con frecuencia, reduciendo esa probabilidad a medida que el sistema "se enfría".

- **Vecindario Dinámico:** Soporta múltiples operadores de perturbación:
  - `flip_bits` (por defecto): Invierte un número pequeño de bits al azar.
  - `swap_bits`: Intercambia un objeto que está en la mochila por uno que está fuera, manteniendo la cantidad de objetos antes de reparar.
  - `block_flip`: Invierte un bloque contiguo de ítems, explorando cambios estructurales.
  Todos los movimientos pasan invariablemente por la función determinista de reparación. **Nota sobre DTW:** Cuando el monitor de estancamiento se activa, sobrescribe dinámicamente la magnitud de estos operadores (ej. cambiando el tamaño del bloque o la cantidad de intercambios de 3 a 1 para explotar, o a 10 para explorar masivamente).
- **Enfriamiento:** Utiliza un decaimiento geométrico clásico: $T_{nueva} = T_{actual} \times \alpha$.
- **Comportamiento Base de Rescate:** Si el monitor DTW detecta estancamiento y estamos en el comportamiento normal (estrategia *reheat*), se reinicia bruscamente la temperatura a un gran porcentaje de la temperatura inicial, para forzar el rechazo e "hervir" el espacio de búsqueda nuevamente.

### 3.2. Algoritmo Genético / Genetic Algorithm (`ga_mkp`)
**Estrategia:** Algoritmo poblacional inspirado en la evolución biológica. Mantiene múltiples soluciones ("individuos") que compiten e intercambian información genética.

- **Selección:** Torneo; se escogen individuos al azar y gana el de mayor ganancia (*fitness*).
- **Elitismo:** Los mejores $N$ individuos pasan inmutables a la siguiente generación para no perder terreno ganado.
- **Crossover (Cruce):** Modular. Soporta Uniforme, de 1 punto y de 2 puntos. 
- **Mutación:** Probabilidad muy pequeña de cambiar algún bit en los individuos nuevos. Soporta mutación por *Bitflip* o mutación *Swap* (intercambio entre un bit 0 y un 1).
- **Comportamiento Base de Rescate:** Si el monitor detecta estancamiento (estrategia *hypermutation*), la tasa de mutación se multiplica dramáticamente durante algunas generaciones, forzando la diversidad genética que se había agotado.

### 3.3. Búsqueda Tabú / Tabu Search (`ts_mkp`)
**Estrategia:** Algoritmo de búsqueda local agresivo que explora siempre todo el vecindario inmediato y prohíbe explícitamente volver a los pasos recientes.

- **Vecindario Activo:** En cada iteración, se examinan múltiples vecinos realizando *flips* y se elige *el mejor vecino factible*, **incluso si es peor** que la solución actual.
- **Lista Tabú:** Cuando se voltea un ítem (cambia un bit), ese índice entra en la lista tabú por un tiempo (*tenure*, ej. 10 iteraciones). Durante ese tiempo, no se puede volver a voltear ese ítem a menos que la ganancia rompa un récord global (criterio de aspiración).
- **Comportamiento Base de Rescate:** Ante estancamiento (estrategia *random_restart*), la lista tabú se vacía y se revierten drásticamente bits de la solución de forma aleatoria, enviando a la búsqueda a otro valle lejano.

---

## 4. Las 8 Variantes de Rescate (Escapes de Estancamiento)

Para comparar de forma exhaustiva, todas las MH fueron programadas para cambiar *su propio comportamiento en caliente* cuando el `StagnationMonitor` detecta que ya no avanzan. Se evaluaron las siguientes estrategias (*V1* a *V8*):

**Nota Importante:** Cada rescate se dispara de forma diferente dependiendo de la naturaleza de la MH, para respetar la lógica del algoritmo subyacente.

#### **V1: Exploit (Intensificación)**
- **SA:** Baja la temperatura al 10% y reduce los cambios a 1 bit. Busca un óptimo local muy profundo inmediatamente.
- **GA:** Quita la mutación, usa cruce local de 1 punto y aplica *Hill Climbing* a la población élite.
- **TS:** Desactiva la memoria tabú temporalmente y amplía los saltos a 2 flips inmediatos para escarbar a fondo.

#### **V2: Cycle (Exploración / Explotación Cíclica)**
Alterna comportamientos en cada disparo.
- **SA:** El disparo $i$ sube mucho la temperatura y hace saltos gigantes (5 flips); el disparo $i+1$ la baja al mínimo con 1 flip.
- **GA:** Alterna entre (Mutación alta, cruce de 2 puntos) y (Mutación baja, cruce uniforme).
- **TS:** Alterna un *Tenure* altísimo (mucha prohibición, explora) y un *Tenure* muy bajo (explotación libre).

#### **V3: Explore (Diversificación)**
- **SA:** Sube mucho la temperatura y permite voltear hasta 10 ítems o 1/4 de toda la mochila.
- **GA:** Trae "inmigrantes masivos": descarta al 50% peor de la población y los reemplaza por aleatorios totales.
- **TS:** Vacía la lista tabú y genera una solución de cero completamente nueva.

#### **V4: Nonlinear (Decaimiento Exponencial de Estrés)**
- **SA:** Ajusta la temperatura dinámicamente: la reinicia fuerte la primera vez, pero menos intensa en disparos sucesivos (decae exponencialmente).
- **GA:** La mutación crece matemáticamente con el exponente del estancamiento: si te estancas repetidas veces, muta muchísimo y cambia a operador *Swap*.
- **TS:** El castigo tabú (*tenure*) se vuelve más estricto y largo según el nivel de desesperación.

#### **V5: Heuristic (Rescate Basado en Reglas)**
Descompone la solución basada en la densidad.
- **En todas las MH:** Vacía la mochila a la mitad y vuelve a construir todo priorizando exclusivamente a los objetos más rentables.

#### **V6: LP (Programación Lineal Básica)**
- **En todas las MH:** Usa heurística basada en los pesos reales de los valores óptimos (relajados, variables continuas). Siembra los resultados de la matemática determinista dentro del estado actual (la temperatura de SA, la población élite de GA o el estado de TS).

#### **V7: Tabu LP (Lineal con Memoria Tabú)**
- **En todas las MH:** Igual a V6 (relajación matemática), pero se protegen/bloquean forzosamente ciertos ítems en una memoria reciente ("Estos ítems no se tocan"). En GA lo hace prohibiendo la mutación de los genes inyectados por LP.

#### **V8: Ruin & Recreate (Destrucción y Recreación)**
- **En todas las MH:** Una técnica sacada del *Vehicule Routing*. Destruye aleatoriamente porciones gigantes de la solución construida y obliga al algoritmo a repararla desde abajo. Más agresivo que la versión heurística porque destruye partes sin importar si el ítem era rentable o no, rompiendo falsos patrones óptimos.
