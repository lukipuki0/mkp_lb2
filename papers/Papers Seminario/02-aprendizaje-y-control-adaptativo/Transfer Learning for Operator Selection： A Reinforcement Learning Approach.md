# Transfer Learning for Operator Selection: A Reinforcement Learning Approach

- PDF: [PDF](./Transfer%20Learning%20for%20Operator%20Selection%EF%BC%9A%20A%20Reinforcement%20Learning%20Approach.pdf)
- Bibliografía: [referencias.bib](../../referencias.bib)
- Clave de cita: `Durgut_2022`

## Lectura útil

Q-learning elige entre tres operadores binarios dentro de ABC; el aporte propio es transferir esa experiencia entre ejecuciones del mismo caso, no diseñar un metaheurístico nuevo.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Rafet Durgut, Mehmet Emin Aydin, Abdur Rakib |
| Soporte | *Algorithms* 2022, 15, 24; recibido 17 dic. 2021, publicado 17 ene. 2022 |
| DOI | 10.3390/a15010024 |
| Tipo | Experimento de selección adaptativa de operadores con transferencia |

## Método

El banco es set union knapsack, 30 instancias (p. 1). ABC sigue como enjambre; Q-learning elige entre binABC, ibinABC y disABC, con el estado agrupado por Hard-C-Means. Comparan RLABC sin transferencia, RLABC-T y RLABC-TC, que reutilizan la tabla y difieren en si el aprendizaje sigue activo (pp. 8 y 15). Es control tabular, no una red que optimice ni una fusión de las tres ecuaciones.

## Qué sostiene el texto

La transferencia entre corridas del mismo caso mejora poco la calidad y de forma clara el tiempo y la convergencia (pp. 1 y 15). En las instancias 2_7 (400 dimensiones) y 2_9 (500), RLABC-T converge antes y se estanca cerca de la iteración 200; RLABC-TC sale de óptimos locales en iteraciones posteriores; RLABC es más lento (p. 13, Fig. 5). Los rangos de tiempo de las Tablas 6–7 favorecen a las variantes con transferencia, con signos del análisis estadístico que no son uniformes en todas las instancias (p. 13).

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Solo probaron el primer nivel: de una corrida a las siguientes del mismo caso. Transferir entre instancias y entre tipos de problema queda explícitamente como trabajo futuro (p. 15). Aprender sí o no tiene ventajas distintas. |
| Lectura crítica | No hay evidencia de que la Q-table sobreviva a otro operador, otro metaheurístico o un paisaje continuo. SUKP no informa sobre HRES, hidrógeno ni DTW. «Significativamente más rápido» es el juicio de los autores sobre sus tablas, no un factor universal. |

## Para el estado del arte

Precedente de reutilizar la experiencia de selección entre corridas del mismo caso, frente a entrenar siempre desde cero y frente a clasificadores offline o políticas neuronales.

Etiquetas secundarias: ABC; Q-learning; problema de mochila de unión de conjuntos.

## Evidencia

- p. 1: objetivo, 30 instancias SUKP y efecto declarado en tiempo.
- p. 8: tres operadores y diseño experimental.
- p. 13: Fig. 5 y tablas de tiempo.
- p. 15: conclusión y dos niveles de transferencia no hechos.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
