# P022 — ε-constraint aumentado (AUGMECON) para el frente de Pareto

**Título:** Effective implementation of the ε-constraint method in Multi-Objective Mathematical Programming problems

**Cita:** `Mavrotas_2009` — George Mavrotas (2009). *Applied Mathematics and Computation*, 213(2), 455–465. DOI: 10.1016/j.amc.2009.03.037.

**PDF:** [PDF](./Effective%20implementation%20of%20the%20%CE%B5-constraint%20method%20in%20Multi-Objective%20Mathematical%20Programming%20problems.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

## Qué aporta

Implementación operativa del método ε-constraint para programación matemática multiobjetivo, llamada AUGMECON, con código en GAMS. No es un algoritmo evolutivo ni un caso energético: es un generador de soluciones Pareto para problemas continuos, enteros o mixtos.

## Problema y método

AUGMECON parte de la tabla de pagos y de una grilla sobre objetivos pasados a restricciones. Suma holguras ponderadas para evitar óptimos Pareto débiles y corta bucles si la grilla queda infactible. En el uso interactivo el decisor elige soluciones y el espacio se contrae en criterios, no en pesos. En el ejemplo, *a* = 0,7 y se muestran cuatro soluciones.

## Hallazgos del texto

El artículo no reporta un banco de instancias con tiempos o cardinalidades de frente. Lo que demuestra es el mecanismo: lexicografía de la tabla de pagos, grilla adaptable (gruesa al inicio, densa al afinar), cotas inferiores guardadas para poder retroceder, y aplicabilidad cuando el espacio de decisión es discontinuo. La versión interactiva, en 2009, se operaba a mano; una interfaz más amigable se declara en desarrollo.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El diálogo con el decisor no está automatizado. La densidad del frente y los valores de reserva quedan a criterio del usuario. |
| Lectura crítica | No hay evidencia numérica, en este PDF, de superioridad frente a NSGA-II, MOPSO u otras metaheurísticas. Tampoco trata hidrógeno, control ni DTW. Sirve como método exacto de generación, no como prueba de que un híbrido metaheurístico sea necesario. |

## Uso en el estado del arte

Base clásica del ε-constraint que Maneengam (P037) luego usa en forma adaptativa de Kirlik y Sayın. Útil si el estado del arte distingue optimización matemática multiobjetivo de metaheurísticas poblacionales.

Etiquetas secundarias: multiobjetivo, Pareto débil, GAMS, decisión interactiva.

## Evidencia

- PDF p. 1: clasificación a priori / interactiva / a posteriori y definición de AUGMECON.
- PDF p. 9: ecuaciones (9)–(10) de contracción y diagrama del proceso interactivo.
- PDF p. 10–11: ejemplo con *a* = 0,7 y conclusiones sobre variables continuas y discretas.
