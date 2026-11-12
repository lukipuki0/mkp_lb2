# Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient

- PDF: [PDF](./Learning%20Adaptive%20Differential%20Evolution%20Algorithm%20From%20Optimization%20Experiences%20by%20Policy%20Gradient.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Sun_2021`

## Lectura útil

Aprende, con policy gradient y experiencias de varios problemas, un controlador que entrega F y CR de DE durante la búsqueda.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Jianyong Sun; Xin Liu; Thomas Bäck; Zongben Xu |
| Soporte | IEEE Transactions on Evolutionary Computation, 25(4), 666-680 |
| DOI | 10.1109/TEVC.2021.3060811 |
| Tipo | Control de parámetros de DE aprendido offline |
| Páginas del PDF | 15 |

## Método

El resumen (p. 1) modela el control como un MDP de horizonte finito. El agente se entrena sobre un conjunto de problemas y después se compara con nueve algoritmos evolutivos. El PDF es la versión aceptada, previa a la edición final.

## Qué sostiene el texto

La p. 1 propone el controlador aprendido como alternativa a fijar parámetros a mano. No se copia el ranking contra los nueve algoritmos.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La comparación declarada está fuera de esta ficha. El archivo advierte que el contenido puede cambiar antes de la publicación final. |
| Lectura crítica | Controla parámetros de un DE, no elige otro solver. No es Ding_2025, que usa PPO distribuido para mutación y F/CR a la vez. El DOI impreso coincide con la clave `Sun_2021`. |

## Para el estado del arte

Antecedente de aprender el control en un conjunto de problemas y aplicarlo después. Distinto de una regla que solo mira la ventana reciente de una corrida.

Etiquetas secundarias: DE; policy gradient; parámetros; experiencia offline.

## Evidencia

- p. 1: DOI, policy gradient, MDP de horizonte finito y la comparación anunciada con nueve algoritmos.

## Procedencia del archivo

El PDF llegó suelto en `papers/sun2021.pdf` y se movió a esta sección. La clave ya existía en `referencias.bib`. La ficha usa la apertura del PDF, no una relectura de las tablas.
