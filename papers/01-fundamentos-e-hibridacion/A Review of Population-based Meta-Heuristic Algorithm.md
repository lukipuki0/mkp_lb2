# A Review of Population-based Meta-Heuristic Algorithm

**Clave de cita:** `Beheshti_2013`  
**PDF:** [PDF](./A%20Review%20of%20Population-based%20Meta-Heuristic%20Algorithm.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Zahra Beheshti; Siti Mariyam Hj. Shamsuddin |
| Año | 2013 |
| Venue | Int. J. Advance. Soft Comput. Appl., 5(1), marzo 2013 |
| DOI | no DOI verificado |
| Sección primaria | `01-fundamentos-e-hibridacion` |
| Etiquetas | revisión; metaheurísticas poblacionales; continuo y binario; sin experimento propio |

## Aporte para el estado del arte

Es un panorama pedagógico de metaheurísticas poblacionales anterior a la ola de control por aprendizaje. Sirve para fijar vocabulario de exploración y explotación y para recordar que ningún algoritmo domina todos los problemas, no para justificar un diseño híbrido concreto.

## Problema y método

El texto argumenta que los métodos exactos no escalan cuando el espacio crece con el tamaño del problema y que las heurísticas clásicas, como las voraces, dependen de supuestos difíciles de validar. A partir de ahí reseña algoritmos poblacionales en espacios continuos y binarios: AIS, GA, ACO, PSO, SDS, ABC, IWD, RFD, GSA y CSS, entre otros. Describe diseño, algoritmo principal, ventajas y desventajas. No propone un algoritmo nuevo ni reporta un banco de pruebas propio. La clasificación de fondo separa heurísticas de trayectoria y poblacionales, y métodos con memoria frente a métodos sin memoria.

## Qué sostiene el texto

La discusión enumera ventajas genéricas: robustez ante condiciones cambiantes, aplicabilidad a problemas multimodales, mecanismos para salir de óptimos locales, independencia relativa del problema, localización de regiones prometedoras en tiempo razonable y facilidad de paralelizar. La conclusión es explícita: ninguno de los algoritmos revisados rinde mejor que los demás en todos los problemas. Como defectos recurrentes citan convergencia prematura e impredecible en GA, estancamiento y lentitud en PSO, coste alto en GSA e ICA, muchos parámetros que ajustar y diseños pensados solo para representación real o solo binaria. Proponer algoritmos que reduzcan esas desventajas queda como problema abierto.

## Límites

**Hallazgos de los autores.** No hay garantía de óptimo. El artículo no mide tasas de éxito ni tiempos: las comparaciones son cualitativas.

**Lectura crítica.** La lista está fechada en 2013 y no cubre control en línea, regresión sobre trazas de búsqueda, hiperheurísticas ni DTW. No debe usarse como evidencia de que esos temas estaban resueltos o vacíos. No hay DOI en el PDF; una consulta Crossref por título y autor no devolvió este artículo.

## Relación con hibridación

Habla de esquemas mejorados de algoritmos existentes, pero no construye una taxonomía de híbridos. Su valor es el sustrato poblacional sobre el que papers posteriores injertan aprendizaje o combinan operadores.

## Evidencia

- p. 1: alcance, espacios continuo y discreto, y lista de algoritmos.
- pp. 26-28: ventajas, desventajas y conclusión de que no hay dominancia universal.

## Procedencia del archivo

- Integración: una consulta Crossref por título y autores (Beheshti y Shamsuddin, 2013) no devolvió este artículo. DOI permanece nulo; no se inventó identificador.
