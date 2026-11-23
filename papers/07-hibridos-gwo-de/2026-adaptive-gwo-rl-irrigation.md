# Adaptive Multi-Strategy Grey Wolf Optimizer with Reinforcement Learning for Multi-Objective Precision Irrigation Optimization

**Clave de cita:** `Yin_2026`  
**PDF:** [PDF](./2026-adaptive-gwo-rl-irrigation.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Guangluan Yin, Wuke Li, Qi Xiong |
| Año | 2026 (recibido 30-01-2026, publicado 24-02-2026) |
| Venue | Algorithms, 19, 168 (MDPI) |
| DOI | 10.3390/a19030168 |
| Sección primaria | `07-hibridos-gwo-de` |
| Etiquetas | AMSGWO-RL-MO; Q-learning sobre cuatro estrategias; estado HV, diversidad y progreso; archivo de Pareto; riego multiobjetivo |

## Aporte para el estado del arte

Es la fuente del grupo con control realimentado más explícito: un Q-learning elige, en cada generación, una de cuatro estrategias de búsqueda (GWO, Lévy, DE y oposición élite) a partir de un estado discreto que mezcla mejora del hipervolumen, diversidad y progreso. Sirve para ubicar el GWO-DE actual entre los esquemas que adaptan la elección de operador, no los parámetros internos.

## Problema y método

Optimizan un calendario de riego de 40 días (`x ∈ [0, 50]` mm por día) con tres objetivos: rendimiento, costo e impacto ambiental (ec. 1, p. 5). Cada candidato se evalúa simulando un balance hídrico del suelo (ecs. 16-17, p. 8). La restricción de agua total entra como penalización (p. 5).

El marco usa una población de 40 y 120 generaciones con un archivo de Pareto y orden no dominado (p. 8). Las estrategias son: GWO estándar con `a` lineal de 2 a 0 (ecs. 18-20, p. 9), vuelo de Lévy con paso 0.01 (ecs. 21-23, pp. 9-10), mutación DE `X_base + F·(X_r1 − X_r2)` con `F ~ U(0.5, 0.8)` y `CR = 0.9` (ec. 24, p. 10) y una perturbación gaussiana alrededor de un miembro del archivo (ec. 25, p. 10).

El MDP (pp. 10-11) tiene 27 estados: tasa de mejora del HV (umbrales 0.01 y 0.05), diversidad PD como media de desviaciones por dimensión (umbrales 3 y 8) y progreso `t/T` en tercios. Las acciones son las cuatro estrategias. La recompensa es `ΔHV·100 + 5` si crece el archivo (ec. 26). Q-learning usa `α = 0.3`, `γ = 0.9` y ε-greedy desde 0.35 con decaimiento 0.95 hasta 0.05 (ecs. 27-28). Un TOPSIS modificado elige el punto rodilla (pp. 11-13).

La granularidad queda abierta en el texto: el módulo B "selects the most appropriate search strategy" en cada iteración (p. 8), sin decir si la acción se aplica a toda la población o por individuo. Tampoco se describe la regla de aceptación de las nuevas posiciones ni cómo se aplica el `CR = 0.9` de la estrategia DE. No hay reinicio; la preservación élite es el archivo de Pareto.

## Qué sostiene el texto

Con 30 corridas y semillas explícitas (p. 14), comparan contra GWO-WS y MS-GWO-WS, ambos con suma ponderada fija (0.45, 0.40, 0.15); MS-GWO-WS usa las mismas cuatro estrategias con probabilidades en tres etapas fijas (p. 15). El punto rodilla logra rendimiento medio 96.96 % frente a 93.44 % y 93.26 %, con costo e impacto ambiental mayores (Tabla 3, p. 16). La prueba de suma de rangos de Wilcoxon da `p < 10⁻⁹` en rendimiento y `p = 1.00` en costo e impacto (Tabla 4, p. 17). Frente a NSGA-II, NSGA-III y MOPSO (Tabla 6, p. 19), el rendimiento no difiere de MOPSO (`p = 0.72`) y el texto admite que MOPSO alcanza mayor HV medio (p. 20). El HV propio medio es 0.3933 (Tabla 5, p. 17).

## Límites

**Hallazgos de los autores.** La validación se limita a simulación; dejan para el futuro pruebas de campo y sensores en tiempo real (p. 21).

**Lectura crítica.** La ablación no aísla el Q-learning: MS-GWO-WS cambia a la vez la elección de estrategia y el tratamiento multiobjetivo (suma ponderada contra archivo de Pareto con rodilla). La comparación de rendimiento enfrenta un punto rodilla, elegido con preferencias propias, contra la única solución de una suma ponderada. Los `p = 1.00` en métricas donde la propuesta es peor sugieren una prueba unilateral en la dirección favorable; el texto no lo declara. No informan presupuestos de evaluación de NSGA-II, NSGA-III y MOPSO, frecuencias de uso de cada acción ni sensibilidad a los umbrales del estado. La lista de contribuciones (p. 4) menciona "dimension learning" y "random walk", que no coinciden con las cuatro estrategias de la sección 2.4.2.

## Relación con el GWO-DE actual

Ambos usan una población compartida y una señal de la búsqueda para decidir. Aquí la señal es multiobjetivo (ΔHV, diversidad, progreso) y la decisión es qué operador aplicar; DE es una de cuatro acciones y sus `F` y `CR` no se adaptan. En el GWO-DE actual, cada individuo recibe un candidato GWO y uno DE que compiten por fitness, y la señal (la trayectoria del mejor fitness) elige perfiles conjuntos de `a`, `F` y `CR`. Esta fuente aporta un precedente de selección de operador por aprendizaje por refuerzo, no evidencia sobre perfiles de parámetros ni sobre competencia por individuo.

## Evidencia

- p. 4: brecha 2 (probabilidades estáticas de estrategia) y contribuciones.
- p. 5: formulación (ecs. 1-2) y penalización de agua.
- p. 8: balance hídrico, población 40, 120 generaciones, módulos A-D.
- pp. 9-10: estrategias (ecs. 18-25).
- pp. 10-11: estado, acciones, recompensa y Q-learning (ecs. 26-28).
- p. 14: 30 corridas, semillas y Tabla 1.
- p. 15: líneas base GWO-WS y MS-GWO-WS.
- pp. 16-17: Tablas 3, 4 y 5.
- pp. 19-20: Tablas 6 y 7, Fig. 6 y comentario sobre HV de MOPSO.
- p. 21: discusión y conclusiones.
