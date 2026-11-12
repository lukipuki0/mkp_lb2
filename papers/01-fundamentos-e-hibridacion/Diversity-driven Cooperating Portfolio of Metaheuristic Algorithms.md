# Diversity-driven Cooperating Portfolio of Metaheuristic Algorithms

- PDF: [PDF](./Diversity-driven%20Cooperating%20Portfolio%20of%20Metaheuristic%20Algorithms.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Zychowski_2025`

## Lectura útil

Portafolio de metaheurísticas distintas en islas, con migración adaptativa por diversidad. No es el artículo de Araujo (2023), aunque lo cita.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Adam Żychowski; Xin Yao; Jacek Mańdziuk |
| Soporte | GECCO 2025, 9 páginas de PDF |
| DOI | 10.1145/3712256.3726442 |
| Tipo | Portafolio de islas con migración por diversidad |
| Páginas del PDF | 9 |

## Método

El resumen (p. 1) separa qué migrar, cuándo y hacia dónde. La evaluación declarada usa TSP y funciones BBOB. La migración se ajusta durante la corrida.

## Qué sostiene el texto

El resumen (p. 1) dice que supera a los algoritmos tradicionales y a sus variantes de islas en velocidad, calidad y resistencia al estancamiento. Esta ficha no releyó las tablas.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La afirmación de superioridad es la del resumen, sobre los benchmarks declarados. |
| Lectura crítica | No es el paper de timing de ICAISC ni el de selección de KES, aunque comparten autores. No usa DTW ni una red. |

## Para el estado del arte

Línea de cooperación entre algoritmos completos. Distinguirlo de Araujo: aquí el portafolio es heterogéneo y la migración tiene tres decisiones, no solo una estrategia de diversidad en un evolutivo.

Etiquetas secundarias: islas; diversidad; migración; TSP; BBOB.

## Evidencia

- p. 1: DOI, tres decisiones de migración, TSP y BBOB, y la comparación declarada.

## Procedencia del archivo

El PDF llegó suelto en `papers/3712256.3726442.pdf` y se movió a esta sección. La ficha usa la apertura del PDF, no una relectura de las tablas de resultados.
