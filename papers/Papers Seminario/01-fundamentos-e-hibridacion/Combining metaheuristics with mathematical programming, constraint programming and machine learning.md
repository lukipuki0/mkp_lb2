# Combining metaheuristics with mathematical programming, constraint programming and machine learning

**Clave de cita:** `Talbi_2013`  
**PDF:** [PDF](./Combining%20metaheuristics%20with%20mathematical%20programming%2C%20constraint%20programming%20and%20machine%20learning.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | El-Ghazali Talbi |
| Año | 2013 |
| Venue | 4OR, 11(2), 101-150 |
| DOI | 10.1007/s10288-013-0242-3 |
| Sección primaria | `01-fundamentos-e-hibridacion` |
| Etiquetas | survey; matheuristics; programación con restricciones; minería de datos |

## Aporte para el estado del arte

Extiende la taxonomía de 2002 a cuatro combinaciones: metaheurística con metaheurística, con programación matemática, con programación con restricciones y con aprendizaje o minería de datos. Es el puente entre el vocabulario de híbridos y los papers que meten regresión dentro de la búsqueda.

## Problema y método

Survey invitado, sin experimento nuevo. Reutiliza la clasificación de diseño e implementación y la idea de gramática, ahora con frases que mezclan metaheurísticas, programación matemática, CP y ML. Cada clase se ilustra con ejemplos publicados y con un análisis crítico. La sección 2 recuerda que los métodos de una solución intensifican y los poblacionales diversifican. La sección 3 trata matheuristics. La 4, restricciones. La 5, minería y ML, incluido extraer patrones de soluciones élite para guiar la búsqueda. El autor dice que la misma clasificación vale para multiobjetivo, pero no desarrolla un caso multiobjetivo propio.

## Qué sostiene el texto

La tesis de cierre es cualitativa: los problemas difíciles se resuelven mezclando paradigmas, y las metaheurísticas puras no ajustan bien paisajes de alta dimensión. De ahí concluye que los híbridos son, en la práctica de entonces, los algoritmos más eficientes para muchos problemas clásicos y reales, por el volumen de híbridos exitosos ya publicados. También prevé que combinar metaheurísticas seguirá siendo estrategia habitual. No aporta una tabla de gaps porcentuales.

## Límites

**Hallazgos de los autores.** La eficiencia se infiere de la literatura ilustrada, no de un protocolo único de reimplementación. El artículo advierte que un nombre como GA cubre algoritmos muy distintos; la taxonomía no sustituye el detalle interno.

**Lectura crítica.** No demuestra que todo híbrido supere a CPLEX o a un GA puro. No nombra HRES, hidrógeno ni DTW. Un estado del arte puede usarlo para clasificar LB2 como combinación con ML, sin afirmar que Talbi haya evaluado esa combinación.

## Relación con hibridación y aprendizaje

La sección de ML es el encaje de regresión, clustering o minería de élites dentro de la búsqueda. Debe distinguirse de usar un optimizador solo para entrenar una red de pronóstico, caso que este survey no trata como contribución central.

## Evidencia

- pp. 1-2: cuatro tipos de combinación y extensión explícita de Talbi 2002.
- p. 43: conclusión sobre paisajes de alta dimensión y papel de los híbridos.
