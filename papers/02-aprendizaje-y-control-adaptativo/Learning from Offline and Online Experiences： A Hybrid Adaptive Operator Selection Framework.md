# P032 — Selección de operadores con experiencia offline y online

**Título:** Learning from Offline and Online Experiences: A Hybrid Adaptive Operator Selection Framework

**Cita:** `Pei_2024` — Jiyuan Pei, Jialin Liu y Yi Mei (2024). *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO ’24)*, pp. 1017–1025. ACM. DOI: 10.1145/3638529.3654062.

**PDF:** [PDF](./Learning%20from%20Offline%20and%20Online%20Experiences%EF%BC%9A%20A%20Hybrid%20Adaptive%20Operator%20Selection%20Framework.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

## Qué aporta

Marco híbrido de selección adaptativa de operadores (AOS) que mezcla experiencia offline, de problemas ya resueltos, y experiencia online, de la búsqueda en curso. El aprendizaje elige componentes de la metaheurística; no sustituye al optimizador por una red que emite la solución.

## Problema y método

La experiencia offline puede engañar si el problema nuevo no se parece a los anteriores. El marco HF combina un módulo de AOS sin estado y otro que aprende en línea, y ajusta la probabilidad de usar cada uno. Se compara con variantes de probabilidad fija (HF-NA con α = 0,5, 0,3 y 0,1), con DQN, con aprendizaje supervisado y con selección aleatoria. Las pruebas son 138 problemas reales de CEC 2017 y BBOB 2023, y 33 instancias C y RC de CVRPTW. El contraste es Wilcoxon de rangos con signo, α = 0,05, contando mejor / igual / peor.

## Hallazgos del texto

En continuos, HF frente a la variante sin el módulo online (HF-NU) queda 23/113/2; frente a aleatorio, 79/58/1; frente a DE-DDQN, 36/95/7. HF-NA(0,5) supera a HF en 19 problemas continuos (5/114/19). En CVRPTW, HF frente a aleatorio es 23/3/7 y frente a HF-NU es 13/15/5; ahí la probabilidad fija 0,3 rinde mejor que 0,5 y 0,1 en el agregado que discuten los autores. Concluyen que ajustar α en línea generaliza más que fijarlo, aunque no gana en cada familia.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El diseño parte de conclusiones previas sobre problemas combinatorios; en continuos una probabilidad fija puede bastar. No reportan costo de entrenamiento frente a una metaheurística sin aprendizaje. |
| Lectura crítica | Es control del algoritmo, no control de una planta ni DTW de series energéticas. No hay HRES. Una victoria en CEC o en ruteo no valida un dimensionamiento de hidrógeno. |

## Uso en el estado del arte

Pieza central si el seminario trata hibridación por aprendizaje dentro de la búsqueda: experiencia offline transferida más corrección online. No debe mezclarse con learnheuristics ni con “PSO que entrena una red”.

Etiquetas secundarias: AOS, DQN, CEC 2017, BBOB, CVRPTW.

## Evidencia

- PDF p. 1: problema de la experiencia offline engañosa y DOI.
- PDF p. 7: Tablas 2 y 3, conteos Wilcoxon.
- PDF p. 8: lectura de α fijo frente a α dinámico y conclusiones.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
