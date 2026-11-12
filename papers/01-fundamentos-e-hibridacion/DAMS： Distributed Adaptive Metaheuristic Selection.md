# DAMS: Distributed Adaptive Metaheuristic Selection

- PDF: [PDF](./DAMS%EF%BC%9A%20Distributed%20Adaptive%20Metaheuristic%20Selection.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Derbel_2011`

## Lectura útil

DAMS coordina, en nodos distribuidos, qué metaheurística corre y qué información se comunica mientras la búsqueda sigue. El algoritmo concreto del PDF es Select Best and Mutate. No entrena una política.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Bilel Derbel; Sébastien Verel |
| Soporte | GECCO 2011, pp. 1955-1962 |
| DOI | 10.1145/2001576.2001839 (no impreso en el PDF; ver límites) |
| Tipo | Selección distribuida de metaheurísticas |
| Páginas del PDF | 8 |

## Método

La apertura (p. 1) define una arquitectura de tres capas: cada nodo decide qué comunicar y qué metaheurística aplicar. SBM explota la de mejor desempeño local y muta para detectar otra mejor. No es un híbrido de bajo nivel ni un modelo de aprendizaje.

## Qué sostiene el texto

El resumen (p. 1) sostiene que SBM es simple y eficiente, y que los experimentos lo comparan con otras estrategias adaptativas secuenciales y distribuidas. Esta ficha no releyó las figuras de topología.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El propio resumen limita la demostración a las experimentaciones del artículo. |
| Lectura crítica | El DOI no está impreso en el PDF. Se registra 10.1145/2001576.2001839 porque el título y los autores coinciden con el proceedings de GECCO 2011. No es el paper de migración de Araujo (2023) ni un controlador por DTW. |

## Para el estado del arte

Antecedente de selección adaptativa entre metaheurísticas completas, no de fusión de operadores. Va con los portafolios de islas, no con las encuestas de aprendizaje.

Etiquetas secundarias: DAMS; islas; selección adaptativa; sin política entrenada.

## Evidencia

- p. 1: título, autores, arquitectura de tres capas y definición de SBM.

## Procedencia del archivo

El PDF llegó suelto en `papers/derbel2011.pdf` y se movió a esta sección. La ficha usa la apertura del PDF, no una relectura de las tablas de resultados.
