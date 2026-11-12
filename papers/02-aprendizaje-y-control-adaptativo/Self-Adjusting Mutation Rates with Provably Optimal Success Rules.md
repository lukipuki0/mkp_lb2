# Self-Adjusting Mutation Rates with Provably Optimal Success Rules

- PDF: [PDF](./Self-Adjusting%20Mutation%20Rates%20with%20Provably%20Optimal%20Success%20Rules.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Doerr_2021`

## Lectura útil

Analiza la regla de un quinto como familia de reglas de éxito, con fuerza de actualización y tasa de éxito, no como una constante fija.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Benjamin Doerr; Carola Doerr; Johannes Lengler |
| Soporte | Algorithmica 83 (2021) 3108-3147 |
| DOI | 10.1007/s00453-021-00854-3 |
| Tipo | Análisis de reglas de éxito |
| Páginas del PDF | 40 |

## Método

El resumen (p. 1) estudia el (1+1) EA en LeadingOnes. El mejor régimen declarado es una fuerza de actualización pequeña, F = 1 + o(1), y tasa de éxito 1/e. Comparan con la mejor tasa dependiente del fitness y con una variante que re-muestrea si no voltea ningún bit.

## Qué sostiene el texto

La p. 1 sostiene que ese régimen iguala, salvo términos de orden menor, el tiempo de la mejor tasa dependiente del fitness. Es un teorema, no una tabla de HRES.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El resultado de la apertura es para LeadingOnes y el (1+1) EA. |
| Lectura crítica | No entrena una política ni elige entre algoritmos. Lehre_2023 también es autoadaptación teórica, pero está en análisis de búsqueda porque sigue un óptimo móvil. Este texto es control de la tasa. |

## Para el estado del arte

Fundamento de un control sin red: la regla de éxito ya tiene un óptimo demostrable en un paisaje simple. No autoriza a copiar 1/5 o 1/e a MKP o a una planta.

Etiquetas secundarias: one-fifth; LeadingOnes; tasa de mutación; teoría.

## Evidencia

- p. 1: DOI, F = 1 + o(1), éxito 1/e y la comparación con la tasa dependiente del fitness.

## Procedencia del archivo

El PDF llegó suelto en `papers/s00453-021-00854-3.pdf` y se movió a esta sección. La clave ya existía en `referencias.bib`. La ficha usa la apertura del PDF, no una relectura de las tablas.
