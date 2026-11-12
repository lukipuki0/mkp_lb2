# P031 — Learnheuristics: el dato cambia con la solución parcial

**Título:** Learnheuristics: hybridizing metaheuristics with machine learning for optimization with dynamic inputs

**Cita:** `Calvet_2017` — Laura Calvet, Jésica de Armas, David Masip y Angel A. Juan (2017). *Open Mathematics*, 15(1), 261–280. DOI: 10.1515/math-2017-0029.

**PDF:** [PDF](./Learnheuristics%EF%BC%9A%20hybridizing%20metaheuristics%20with%20machine%20learning%20for%20optimization%20with%20dynamic%20inputs.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

## Qué aporta

Define las learnheuristics: metaheurísticas coordinadas con un modelo de aprendizaje que actualiza entradas dinámicas del problema, no los operadores. El ejemplo es una regresión lineal de juguete dentro de Clarke–Wright, no una red entrenada como optimizador.

## Problema y método

Tras revisar combinaciones previas de metaheurísticas y aprendizaje, el artículo separa los problemas combinatorios con entradas dinámicas (COPDI): la demanda, el tiempo de viaje o el retorno de un activo dependen de la solución que se va construyendo, de forma predecible y no aleatoria. En cada paso el modelo de aprendizaje reescribe las entradas que usa la metaheurística. Los campos ilustrados son transporte, logística, producción y finanzas. El experimento numérico usa el heurístico de Clarke y Wright y la demanda *d* = max{*k1 d0*, *d0* − *k2 d0* (*p* − 1)}, con *k1* = 0,20 y *k2* = 0,05, donde *p* es la posición en la ruta.

## Hallazgos del texto

Con demandas fijas el heurístico obtiene costo 896,86 y 11 rutas. Con demandas que caen según el orden de visita, el mismo heurístico obtiene costo 791,26 y 8 rutas. El ahorro ilustra que ignorar la dependencia entre solución y dato cambia el problema, no que el aprendizaje haya descubierto una política. Los autores presentan el caso como demostración del marco.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El modelo de demanda es lineal y fijado a mano. El trabajo futuro queda abierto a instancias reales y a otros aprendizajes. |
| Lectura crítica | No es selección adaptativa de operadores ni un hiperheurístico. No hay HRES, hidrógeno, control de planta ni DTW. Citar el 791,26 como mejora algorítmica sería incorrecto: cambia el modelo de demanda. |

## Uso en el estado del arte

Distingue “aprender el dato que depende de la solución” de “aprender qué operador lanzar”. Complementa las taxonomías de Talbi y de Karimi-Mamaghan sin sustituirlas.

Etiquetas secundarias: COPDI, Clarke–Wright, entradas dinámicas.

## Evidencia

- PDF p. 1: definición de learnheuristic y de COPDI.
- PDF p. 13: aplicaciones en transporte, logística, producción y finanzas.
- PDF p. 14: ecuación (4), costos 896,86 y 791,26, y conclusiones.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
