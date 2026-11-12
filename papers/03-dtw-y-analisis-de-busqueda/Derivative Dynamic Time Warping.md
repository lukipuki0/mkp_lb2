# Derivative Dynamic Time Warping

- PDF: [PDF](./Derivative%20Dynamic%20Time%20Warping.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Keogh_2001`

## Lectura útil

Define Derivative Dynamic Time Warping: alinear por la forma local, estimada con la derivada, cuando el valor de la serie no coincide en el eje temporal.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Eamonn J. Keogh; Michael J. Pazzani |
| Soporte | SIAM International Conference on Data Mining, 2001 |
| DOI | 10.1137/1.9781611972719.1 (no impreso en el PDF; ver límites) |
| Tipo | Método DDTW |
| Páginas del PDF | 11 |

## Método

La p. 1 recuerda que DTW deforma el eje para comparar formas parecidas desalineadas, y cita a Berndt y Clifford (1994) entre los usos de DTW. El PDF tiene 11 páginas, coherente con las pp. 1-11 del proceedings.

## Qué sostiene el texto

La apertura sostiene que la distancia punto a punto es pesimista si las formas no están alineadas, y que DTW calcula ese alineamiento. El detalle de la derivada no se reextrajo más allá de la p. 1.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Los ejemplos de la p. 1 son series, incluido lenguaje de señas, no trayectorias de fitness. |
| Lectura crítica | El DOI no está impreso. Se registra 10.1137/1.9781611972719.1 por el registro SIAM del mismo título y autores. No es el paper de Dau sobre el ancho de ventana. |

## Para el estado del arte

Es la cita de DDTW de ASOC. Junto con Berndt explica el par de medidas; no dice cómo umbralizar una meseta de optimización.

Etiquetas secundarias: DDTW; derivada; alineación; series.

## Evidencia

- p. 1: título, autores, motivo del warping y la cita a Berndt y Clifford.

## Procedencia del archivo

El PDF llegó suelto en `papers/keogh2001.pdf` y se movió a esta sección. La ficha usa la apertura del PDF, no una relectura de las tablas de resultados.
