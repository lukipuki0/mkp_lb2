# A framework for designing of genetic operators automatically based on gene expression programming and differential evolution

- PDF: [PDF](./A%20framework%20for%20designing%20of%20genetic%20operators%20automatically%20based%20on%20gene%20expression%20programming%20and%20differential%20evolution.pdf)
- Bibliografía: [referencias.bib](../../referencias.bib)
- Clave de cita: `Jiang_2021`

## Lectura útil

DGOA busca la ecuación del operador con programación de expresión génica mientras DE optimiza la función; no es un selector entre operadores ya escritos ni un controlador por refuerzo.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Dazhi Jiang, Zhihang Tian, Zhihui He, Geng Tu, Ruixiang Huang |
| Soporte | *Natural Computing* (2021) 20:395–411; aceptado 27 nov. 2020, publicado 9 ene. 2021 |
| DOI | 10.1007/s11047-020-09830-2 |
| Tipo | Marco de diseño automático de operadores, con verificación en funciones clásicas |

## Método

El diseño manual de operadores exige resumir muchos datos (p. 1). El marco tiene dos módulos que corren a la vez (p. 5, Fig. 2): GEP genera expresiones de operadores mediante selección, mutación, transposición y recombinación; DE las aplica al problema y devuelve el valor alcanzado para que GEP siga buscando. Los autores insisten en que el objetivo no es batir el estado del arte, sino mostrar un esquema de diseño automático (p. 1). Comparan el operador evolucionado con el de SaDE en funciones de minimización. Es hibridación GEP+DE en dos espacios —operadores y soluciones—, distinta de un portafolio que elige JADE o PSO.

## Qué sostiene el texto

En el cierre, el operador híbrido generado por GEP converge más rápido que el operador clásico de SaDE en las funciones mostradas, y el conjunto se describe como estable aunque algún caso no cumple la expectativa (p. 13). No se copia aquí una tabla de errores medios porque el pasaje de conclusión no fija esos valores.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Los operadores generados son impredecibles y algunos deben descartarse; el resultado ocasional no alcanza lo esperado (pp. 5 y 13). |
| Lectura crítica | El espacio de expresiones es enorme —ilustran N del orden de 1,9×10^14 para un cromosoma corto (p. 5)— pero el experimento no demuestra exploración sistemática ni ventaja fuera de funciones continuas clásicas. Invocar NFL para sugerir que operadores autoajustados «rompen» el teorema es una hipótesis de los autores, no una prueba. No hay RL, DTW, HRES ni hidrógeno. |

## Para el estado del arte

Contrasta generación de operadores (hiperheurística generativa) con selección de operadores ya definidos. Útil para no clasificar todo «diseño automático» como aprendizaje por refuerzo.

Etiquetas secundarias: GEP; DE; SaDE como línea base; diseño automático.

## Evidencia

- p. 1: motivación, dos pasos y alcance declarado.
- p. 5: módulos, interacción y operadores a evitar.
- p. 13: comparación cualitativa con SaDE.
