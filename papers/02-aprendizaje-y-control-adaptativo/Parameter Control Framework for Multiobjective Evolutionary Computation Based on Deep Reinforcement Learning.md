# Parameter Control Framework for Multiobjective Evolutionary Computation Based on Deep Reinforcement Learning

- PDF: [PDF](./Parameter%20Control%20Framework%20for%20Multiobjective%20Evolutionary%20Computation%20Based%20on%20Deep%20Reinforcement%20Learning.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Zhou_2024`

## Lectura útil

Marco de control de parámetros por RL profundo y transferencia, probado al envolver cuatro MOEA clásicos. La red elige parámetros; no sustituye al algoritmo multiobjetivo.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Tianwei Zhou, Wenwen Zhang, Ben Niu, Pengcheng He, Guanghui Yue |
| Soporte | *International Journal of Intelligent Systems*, Volume 2024, Article ID 6740701, 17 páginas; publicado 3 ene. 2024 |
| DOI | 10.1155/2024/6740701 |
| Tipo | Marco de control y experimento en DTLZ/WFG |

El editor académico impreso es Fabio Carafni (p. 1). La entrada BibTeX reutilizada escribe Caraffini. Se conserva la entrada existente y se advierte el desajuste.

## Método

El estado evita el frente exacto: rasgos del problema, posición relativa, distribución del fitness y grid-IGD (p. 1). Definen el MDP y eligen parámetros en espacio continuo con RL profundo (p. 14). Los envoltorios son R-MOEA/D, R-NSGA-II, R-MOPSO y R-MODE, frente a su versión original, en cuatro DTLZ y ocho WFG, 30 corridas, IGD e HV (pp. 1, 12 y 14). Transfieren experiencia de ajuste; no mezclan las cuatro poblaciones.

## Qué sostiene el texto

Las versiones reforzadas convergen más rápido y son más robustas que las originales; los diagramas de caja de IGD muestran menor dispersión y, en el texto, el mejor resultado en todos los problemas mostrados (pp. 1 y 12). La Tabla de rankings de la p. 14 da, entre otros, R-MOEA/D 3,43 y 3,39 frente a cifras peores de las versiones clásicas en ese resumen de rangos. Esos números son rangos agregados, no valores de IGD.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La mejora tiene límite en problemas con los que el algoritmo base ya lucha (p. 12). |
| Lectura crítica | DTLZ/WFG no son despacho energético ni hidrógeno. Envolver cuatro MOEA no demuestra que el mismo estado sirva para DE monoobjetivo o VNS. Grid-IGD sigue siendo un indicador de archivo, no una medición de campo. No hay DTW. |

## Para el estado del arte

Control por DRL de parámetros multiobjetivo, distinto de un selector por GD/MS sin red. El multiobjetivo es el banco, no un sistema energético.

Etiquetas secundarias: MOEA/D; NSGA-II; MOPSO; MODE; transferencia; IGD/HV.

## Evidencia

- p. 1: estado, cuatro r-MOEA y bancos.
- p. 12: 30 corridas, boxplots y límite cuando el base ya falla.
- p. 14: cuatro nombres reforzados y rankings 3,43 / 3,39 de R-MOEA/D.
