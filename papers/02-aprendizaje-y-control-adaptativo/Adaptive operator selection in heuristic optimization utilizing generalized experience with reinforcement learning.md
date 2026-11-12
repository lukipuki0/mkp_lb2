# Adaptive operator selection in heuristic optimization utilizing generalized experience with reinforcement learning

- PDF: [PDF](./Adaptive%20operator%20selection%20in%20heuristic%20optimization%20utilizing%20generalized%20experience%20with%20reinforcement%20learning.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Aydin_2025`

## Lectura útil

Propone un marco de RL para guardar experiencia de selección de operadores y reutilizarla, no solo dentro de una corrida. La clave ya estaba en la bibliografía, sin PDF.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Mehmet Emin Aydin; Rafet Durgut; Abdur Rakib; Khoa Phung |
| Soporte | International Journal of Parallel, Emergent and Distributed Systems; en línea 20 dic. 2025 |
| DOI | 10.1080/17445760.2025.2605297 |
| Tipo | Selección de operadores con experiencia generalizada |
| Páginas del PDF | 32 |

## Método

El resumen (p. 1) parte de varios operadores y de la dificultad de escalar un selector por RL. El marco adquiere, procesa y reutiliza experiencia inmediata y de largo plazo. Palabras clave impresas: operator selection, reinforcement learning, transfer learning.

## Qué sostiene el texto

El resumen (p. 1) dice que los experimentos dan un cierto nivel de éxito al validar el marco. No se reproduce una tabla.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La apertura no cuantifica una mejora universal ni nombra el banco. |
| Lectura crítica | No es Durgut_2022, que transfiere una Q-table de ABC entre corridas del mismo caso. Aquí la experiencia se presenta como generalizada. No hay DTW ni HRES. |

## Para el estado del arte

Cierra el hueco de `Aydin_2025` en el catálogo. Sirve como selector de operadores con transferencia, frente a Pei_2024, que mezcla experiencia offline y online en otro diseño.

Etiquetas secundarias: operadores; refuerzo; transferencia; experiencia generalizada.

## Evidencia

- p. 1: DOI, fecha 20 dic. 2025, resumen de experiencia generalizada y la afirmación de validación parcial.

## Procedencia del archivo

El PDF llegó suelto en `papers/Adaptive-operator-selection-in-heuristic-optimization-utilizing-generalized-experience-with.pdf` y se movió a esta sección. La clave ya existía en `referencias.bib`. La ficha usa la apertura del PDF, no una relectura de las tablas.
