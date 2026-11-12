# Search trajectory networks: A tool for analysing and visualising the behaviour of metaheuristics

- PDF: [PDF](./Search%20trajectory%20networks%EF%BC%9A%20A%20tool%20for%20analysing%20and%20visualising%20the%20behaviour%20of%20metaheuristics.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Ochoa_2021`

## Lectura útil

Propone las search trajectory networks para ver y contrastar cómo se mueven las metaheurísticas. Es análisis, no un controlador.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Gabriela Ochoa; Katherine M. Malan; Christian Blum |
| Soporte | Applied Soft Computing 109 (2021) 107492 |
| DOI | 10.1016/j.asoc.2021.107492 |
| Tipo | Herramienta de análisis de trayectorias |
| Páginas del PDF | 15 |

## Método

El resumen (p. 1) construye el grafo con datos recogidos mientras el algoritmo resuelve el problema, sin un muestreo aparte. Los casos cubren optimización continua y combinatoria, y paradigmas evolutivo, de enjambre y búsqueda local.

## Qué sostiene el texto

La p. 1 plantea que, quitada la metáfora, faltan herramientas para saber si los algoritmos se comportan distinto. STN es esa herramienta. No se copian métricas de los casos.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El resumen lo presenta como modelo de análisis y visualización. |
| Lectura crítica | No decide cuándo cambiar de solver. Reijnen reutiliza STN dentro de un agente; no es este artículo. Tampoco es DTW de la serie de fitness. |

## Para el estado del arte

Separa trayectoria como grafo de soluciones visitadas de trayectoria como serie temporal del mejor fitness. El controlador de ASOC está en el segundo sentido.

Etiquetas secundarias: STN; grafos; visualización; continuo y combinatorio.

## Evidencia

- p. 1: DOI, construcción sin muestreo extra y los dos casos.

## Procedencia del archivo

El PDF llegó suelto en `papers/1-s2.0-S1568494621004154-main.pdf` y se movió a esta sección. La ficha usa la apertura del PDF, no una relectura de las tablas de resultados.
