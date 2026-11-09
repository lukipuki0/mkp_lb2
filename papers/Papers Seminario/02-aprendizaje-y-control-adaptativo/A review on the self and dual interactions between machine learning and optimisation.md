# A review on the self and dual interactions between machine learning and optimisation

**Clave de cita:** `Song_2019`  
**PDF:** [PDF](./A%20review%20on%20the%20self%20and%20dual%20interactions%20between%20machine%20learning%20and%20optimisation.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Heda Song; Isaac Triguero; Ender Özcan |
| Año | 2019 |
| Venue | Progress in Artificial Intelligence, 8(2), 143-165 |
| DOI | 10.1007/s13748-019-00185-z |
| Sección primaria | `02-aprendizaje-y-control-adaptativo` |
| Etiquetas | revisión; meta-aprendizaje; hiperheurísticas; selección de algoritmos; terminología |

## Aporte para el estado del arte

Ordena cuatro familias de interacción: aprendizaje que mejora aprendizaje, optimización que mejora aprendizaje, aprendizaje que mejora optimización y optimización que mejora optimización. Es el marco conceptual para no llamar híbrido a cualquier red entrenada con un optimizador.

## Problema y método

El artículo es un tutorial-revisión. Parte de que ML y optimización se hibridan entre sí y consigo mismos, pero usan términos solapados con significados distintos, en particular meta-aprendizaje, metaheurística e hiperheurística. Para cada interacción propone una taxonomía, compara técnicas que atacan la misma tarea y discute ventajas y desventajas. No entrena un modelo ni resuelve un banco de funciones. Entre los mecanismos de optimización que se optimizan a sí misma están la mezcla de heurísticas, la búsqueda cooperativa y la configuración de parámetros formulada como otro problema de optimización. En el lado ML hacia optimización revisa extracción de conocimiento desde instancias para transferirlo a instancias no vistas.

## Qué sostiene el texto

El cierre no cuantifica un ganador. Sostiene que hace falta una visión conjunta porque las revisiones previas miraban una sola interacción. Como direcciones abiertas señala combinar varias interacciones, la falta de una API que conecte herramientas de ML y de optimización, y que el uso de ML para mejorar optimización sigue apoyado en algoritmos convencionales, redes, SVM y k-means, más que en métodos entonces emergentes. También observa que el flujo habitual es extraer conocimiento y después optimizar, menos eficiente que un esquema extremo a extremo, difícil porque las variables combinatorias son discretas y muchos parámetros de ML son continuos.

## Límites

**Hallazgos de los autores.** Las taxonomías son cualitativas. Los ejemplos, como tabu search para permutar heurísticas de coloreado, ilustran la clase; no son resultados nuevos.

**Lectura crítica.** No mide si una regresión lineal en línea, como LB2, cae en una casilla mejor que una hiperheurística de selección. Publicar en 2019 deja fuera refuerzo profundo posterior. No habla de DTW ni de HRES; esa omisión no es un hueco empírico.

## Relación con control adaptativo

La sección de optimización para optimización es el antecedente directo del control de parámetros y de la selección de operadores. Distingue ese control de usar un metaheurístico solo para ajustar una red de pronóstico.

## Evidencia

- p. 1: cuatro objetivos del estudio, incluido aclarar terminología.
- pp. 16-18: transferencia de conocimiento, optimización de parámetros y remarks sobre APIs y métodos extremo a extremo.
