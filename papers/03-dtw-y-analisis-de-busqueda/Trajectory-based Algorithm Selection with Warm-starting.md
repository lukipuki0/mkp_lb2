# Trajectory-based Algorithm Selection with Warm-starting

- PDF: [PDF](./Trajectory-based%20Algorithm%20Selection%20with%20Warm-starting.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Jankovic_2022`

## Lectura útil

Calcula rasgos de paisaje con puntos que el solver ya evaluó, y contempla un único cambio de algoritmo con warm-start. Evita un muestreo previo solo para extraer rasgos.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Anja Jankovic; Diederick Vermetten; Ana Kostovska; Jacob de Nobel; Tome Eftimov; Carola Doerr |
| Soporte | 2022 IEEE Congress on Evolutionary Computation |
| DOI | 10.1109/CEC55065.2022.9870222 |
| Tipo | Selección por rasgos de la trayectoria |
| Páginas del PDF | 8 |

## Método

El resumen (p. 1) entrena regresión de desempeño y un selector por corrida sobre un portafolio de cinco algoritmos, con un presupuesto fijo de evaluaciones. Parte de ese presupuesto alimenta los rasgos.

## Qué sostiene el texto

La p. 1 dice que el selector por trayectoria con warm-start muestra desempeño prometedor. No se reproduce una tabla.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El cambio de solver considerado es uno. El warm-start depende de poder inicializar el segundo algoritmo con lo ya muestreado. |
| Lectura crítica | Los rasgos no conservan por sí mismos el orden temporal que usa DTW. Kostovska (2022) no está en esta carpeta. No es STN. |

## Para el estado del arte

Muestra selección durante la corrida a partir de la trayectoria muestral. El hueco que ASOC marca es otro: la forma ordenada del fitness, no un vector de rasgos.

Etiquetas secundarias: selección por corrida; warm-start; ELA; portafolio.

## Evidencia

- p. 1: DOI, cinco algoritmos, un cambio con warm-start y rasgos tomados del propio presupuesto.

## Procedencia del archivo

El PDF llegó suelto en `papers/jankovic2022.pdf` y se movió a esta sección. La ficha usa la apertura del PDF, no una relectura de las tablas de resultados.
