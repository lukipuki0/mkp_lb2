# Design and Analysis of Schemes for Adapting Migration Intervals in Parallel Evolutionary Algorithms

- PDF: [PDF](./Design%20and%20Analysis%20of%20Schemes%20for%20Adapting%20Migration%20Intervals%20in%20Parallel%20Evolutionary%20Algorithms.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Mambrini_2015`

## Lectura útil

Adapta el intervalo de migración de un modelo de islas según aparezcan o no mejoras de fitness. Es teoría de algoritmos evolutivos paralelos, no un selector aprendido.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Andrea Mambrini; Dirk Sudholt |
| Soporte | Evolutionary Computation, 23(4), 559-582 |
| DOI | 10.1162/EVCO_a_00153 |
| Tipo | Análisis teórico de migración adaptativa |
| Páginas del PDF | 24 |

## Método

El resumen (p. 1) aumenta el intervalo mientras no hay mejora, para comunicar menos, y lo reduce cuando el mejor fitness mejora, para difundir la solución. Dan un método de cotas para tiempo esperado y esfuerzo de comunicación.

## Qué sostiene el texto

El resumen (p. 1) sostiene que, en funciones ejemplo, los esquemas adaptativos compiten o superan al mejor intervalo fijo, en tiempo y en comunicación. Esta ficha no releyó las pruebas.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Las cotas se ilustran en funciones ejemplo, no en un portafolio heterogéneo de metaheurísticas. |
| Lectura crítica | No decide qué algoritmo corre en cada isla. No usa diversidad como señal, ni DTW, ni aprendizaje. El DOI sí está impreso en la p. 1. |

## Para el estado del arte

Muestra que el calendario de cooperación puede depender del progreso de la búsqueda. No sustituye a Araujo (2023), que migra por diversidad, ni a los papers de Żychowski.

Etiquetas secundarias: islas; intervalo de migración; cotas; sin política entrenada.

## Evidencia

- p. 1: DOI, regla de aumento y reducción del intervalo, y la afirmación sobre el intervalo fijo.

## Procedencia del archivo

El PDF llegó suelto en `papers/mambrini2015.pdf` y se movió a esta sección. La ficha usa la apertura del PDF, no una relectura de las tablas de resultados.
