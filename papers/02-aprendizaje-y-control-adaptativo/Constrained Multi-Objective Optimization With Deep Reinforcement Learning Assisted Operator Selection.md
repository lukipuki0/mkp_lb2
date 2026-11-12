# Constrained Multi-Objective Optimization With Deep Reinforcement Learning Assisted Operator Selection

- PDF: [PDF](./Constrained%20Multi-Objective%20Optimization%20With%20Deep%20Reinforcement%20Learning%20Assisted%20Operator%20Selection.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Ming_2024`

## Lectura útil

Una Q-network elige el operador de un CMOEA según convergencia, diversidad y factibilidad de la población.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Fei Ming; Wenyin Gong; Ling Wang; Yaochu Jin |
| Soporte | IEEE/CAA Journal of Automatica Sinica, 11(4), 919-931 |
| DOI | 10.1109/JAS.2023.123687 |
| Tipo | Selección de operadores por DRL en CMOEA |
| Páginas del PDF | 13 |

## Método

El resumen (p. 1) toma el estado de la población, los operadores como acciones y la mejora del estado como recompensa. El marco se inserta en cuatro CMOEA y se prueba en 42 problemas, frente a nueve CMOEA del estado del arte.

## Qué sostiene el texto

La p. 1 dice que la selección asistida por DRL mejora esos cuatro algoritmos y da más versatilidad. Esta ficha no releyó las tablas.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El resultado declarado es sobre CMOEA con restricciones, no sobre un objetivo único sin restricciones. |
| Lectura crítica | Elige operadores, no solvers completos. No es el parameter control de Zhou_2024 ni DTW. |

## Para el estado del arte

Muestra el otro grano de la decisión: operador dentro de un algoritmo, con estado de factibilidad. No se compara con un controlador de portafolios.

Etiquetas secundarias: multiobjetivo; restricciones; Q-network; operadores.

## Evidencia

- p. 1: DOI, estado de convergencia/diversidad/factibilidad, cuatro CMOEA y 42 problemas.

## Procedencia del archivo

El PDF llegó suelto en `papers/Constrained-Multi-Objective-Optimization-With-Deep-Reinforcement-Learning-Assisted-Operator.pdf` y se movió a esta sección. La clave ya existía en `referencias.bib`. La ficha usa la apertura del PDF, no una relectura de las tablas.
