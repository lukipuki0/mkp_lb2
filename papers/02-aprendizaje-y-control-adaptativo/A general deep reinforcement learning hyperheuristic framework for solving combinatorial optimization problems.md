# A general deep reinforcement learning hyperheuristic framework for solving combinatorial optimization problems

- PDF: [PDF](./A%20general%20deep%20reinforcement%20learning%20hyperheuristic%20framework%20for%20solving%20combinatorial%20optimization%20problems.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Kallestad_2023`

## Lectura útil

DRLH sustituye la capa adaptativa de ALNS por un agente de RL profundo que elige heurísticas de destrucción e inserción; ALNS sigue siendo el esqueleto de búsqueda.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Jakob Kallestad, Ramin Hasibi, Ahmad Hemmati, Kenneth Sörensen |
| Soporte | *European Journal of Operational Research* 309 (2023) 446–468; en línea 16 ene. 2023 |
| DOI | 10.1016/j.ejor.2023.01.017 |
| Tipo | Marco de hiperheurística de selección con experimento en cuatro problemas combinatorios |

## Método

ALNS pondera heurísticas por el segmento reciente. DRLH observa el estado —incluida la diferencia de objetivo— y entrena una política MLP por RL, sin imitar a un experto (pp. 1 y 3). La aceptación es recocido simulado. El pool mezcla remociones e inserciones (greedy, beam search, primera factible, varianza, vehículo). Los problemas son CVRP, PJSP, PDP y PDPTW (p. 5). La red solo elige el operador; no construye la ruta.

## Qué sostiene el texto

Frente a ALNS y a selección uniforme, DRLH elige mejor y no se degrada al ampliar el pool; ALNS sí (p. 1). En las tablas de CVRP, DRLH supera a ALNS y URS, a veces con más tiempo (p. 12). Una lectura que los autores marcan como necesitada de más prueba: se puede entregar un pool grande y dejar que el agente descarte heurísticas inútiles durante el entrenamiento, ahorrando la preselección manual que ALNS suele exigir (p. 11).

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La integración de la recompensa RtMC quedó incompleta por tiempo de experimento (p. 16). La conclusión sobre pools grandes pide más evidencia empírica (p. 11). |
| Lectura crítica | «General» significa cuatro familias combinatorias con operadores de inserción/remoción, no dominios continuos, multiobjetivo, DTW o energía. El costo de entrenar el agente no desaparece: se traslada de la puesta a punto manual al entrenamiento. No comparar sus gaps de CVRP con un dimensionamiento de HRES. |

## Para el estado del arte

Puente entre selección clásica y control por red. No es clasificación offline ni Q-learning sobre un enjambre.

Etiquetas secundarias: ALNS; CVRP; PDP; programación de trabajos; MLP como política, no como optimizador.

## Evidencia

- p. 1: contraste con la capa adaptativa de ALNS y efecto del tamaño del pool.
- p. 3: diferencia frente a aprendices supervisados.
- p. 5: cuatro problemas.
- pp. 11 y 16: conclusión sobre preselección y recompensa pendiente.
