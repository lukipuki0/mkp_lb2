# Three stage based reinforcement learning for combining multiple metaheuristic algorithms

- PDF: [PDF](./Three%20stage%20based%20reinforcement%20learning%20for%20combining%20multiple%20metaheuristic%20algorithms.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Liu_2025`

## Lectura útil

Usa aprendizaje por refuerzo para elegir, en tres etapas de la búsqueda, qué metaheurística completa genera soluciones. La política sí se entrena.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Xiaotong Liu; Tianlei Wang; Zhiqiang Zeng; Ye Tian; Jun Tong |
| Soporte | Swarm and Evolutionary Computation 95 (2025) 101935 |
| DOI | 10.1016/j.swevo.2025.101935 |
| Tipo | RL para combinar metaheurísticas |
| Páginas del PDF | 42 |

## Método

El resumen (p. 1) define exploración, exploración-explotación y explotación. La acción elige el algoritmo según el estado de la población, y la recompensa empuja a la población hacia el estado esperado.

## Qué sostiene el texto

La p. 1 presenta TSRL-CMM como respuesta a la dificultad de hacer colaborar varios algoritmos. Esta ficha no releyó las tablas de las 42 páginas.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El resumen no cuantifica una mejora universal; el detalle experimental queda fuera de esta ficha. |
| Lectura crítica | Elige algoritmos, no la forma de una serie de fitness. No es DAC general al estilo de Adriaensen ni STN. |

## Para el estado del arte

Ejemplo de política aprendida sobre un portafolio. Contrastar con un controlador por reglas, que no entrena estado, acción ni recompensa.

Etiquetas secundarias: refuerzo; tres etapas; selección de algoritmo; población.

## Evidencia

- p. 1: DOI, tres etapas, selección por estado de la población y recompensa.

## Procedencia del archivo

El PDF llegó suelto en `papers/1-s2.0-S2210650225000938-main.pdf` y se movió a esta sección. La ficha usa la apertura del PDF, no una relectura de las tablas de resultados.
