# P028 — Híbridos global–local en funciones no convexas

**Título:** Hybrid Optimization: Combining Global and Local Search Algorithms for Efficient Non-Convex Optimization

**Cita:** `El_Hashash_2025` — Mahmoud El-Hashash (2025). *International Mathematical Forum*, 20(3), 75–86. DOI: 10.12988/imf.2025.914512.

**PDF:** [PDF](./Hybrid%20Optimization%EF%BC%9A%20Combining%20Global%20and%20Local%20Search%20Algorithms%20for%20Efficient%20Non-Convex%20Optimization.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

## Qué aporta

Cuatro híbridos que exploran con un algoritmo genético o con PSO y refinan con Newton o gradiente conjugado. Es hibridación de buscador global y método local clásico, no un modelo de aprendizaje.

## Problema y método

Los problemas no convexos tienen muchos mínimos. GA (el texto cita el artículo de 1992 de Holland, no el libro de 1975) y PSO exploran pero convergen lento; Newton y gradiente conjugado convergen rápido y caen en mínimos locales. Se proponen GA+Newton, GA+CG, PSO+Newton y PSO+CG, medidos por valor final, iteraciones, tiempo de CPU y orden de convergencia, sobre Rosenbrock, Rastrigin, Ackley y Himmelblau. El autor lo presenta como continuación de un híbrido símplex–murciélago (HSBA) publicado aparte.

## Hallazgos del texto

El relato de resultados dice que los cuatro híbridos superan a sus componentes en todas las funciones de prueba. GA+Newton sería el de convergencia más rápida y menor valor final, sobre todo en Rosenbrock y Ackley. Los híbridos con gradiente conjugado se presentan como más robustos si falta Hessiano o gradiente fiable. PSO+Newton se reserva a funciones suaves y diferenciables. La extracción de texto no recupera las celdas numéricas de las tablas: el PDF las anuncia (“comparison table”, “performance comparison”) pero los valores no quedan en el texto plano. Por eso esta ficha no repite cifras de error o de tiempo.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El trabajo futuro es pasar a problemas con restricciones, más dimensiones y un criterio adaptativo para cambiar de fase global a local. |
| Lectura crítica | Sin cifras recuperables no se puede auditar la magnitud de la mejora ni el presupuesto de evaluaciones. Cuatro funciones clásicas no representan un paisaje de dimensionamiento con variables mixtas. No hay hidrógeno, control ni DTW. |

## Uso en el estado del arte

Contrasta la hibridación por fases de enjambres (Lale) con el esquema exploración poblacional más pulido newtoniano. Útil solo como patrón algorítmico, no como resultado cuantitativo citado.

Etiquetas secundarias: Newton, gradiente conjugado, benchmarks clásicos.

## Evidencia

- PDF p. 1: cuatro híbridos y funciones de prueba.
- PDF p. 8–10: métricas y afirmaciones cualitativas; tablas no legibles en el texto extraído.
- PDF p. 11: conclusión y trabajo futuro.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
