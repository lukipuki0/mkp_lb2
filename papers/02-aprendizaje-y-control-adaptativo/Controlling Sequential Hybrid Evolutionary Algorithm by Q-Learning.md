# Controlling Sequential Hybrid Evolutionary Algorithm by Q-Learning

- PDF: [PDF](./Controlling%20Sequential%20Hybrid%20Evolutionary%20Algorithm%20by%20Q-Learning.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Zhang_2023`

## Lectura útil

El agente Q-learning o DQN no optimiza la función: decide cuándo un híbrido secuencial cambia de fase, en particular hacia CMA-ES. El caso demostrado es HSES y una variante de LSHADE.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Haotian Zhang, Jianyong Sun, Thomas Bäck, Qingfu Zhang, Zongben Xu |
| Soporte | *IEEE Computational Intelligence Magazine* 18(1), 84–103, febrero 2023; artículo de la sección Research Frontier |
| DOI | 10.1109/MCI.2022.3222057 |
| Tipo | Marco de control estructural, con experimentos CEC 2014 y 2018 |

La portada de repositorio de Leiden (p. 1 del PDF) antecede al artículo. El título de portada del artículo es «Controlling Sequential Hybrid Evolutionary Algorithm by Q-Learning»; «Research Frontier» es el kicker de sección, no parte del título. Crossref repite `[Research Frontier]` dos veces y no se usó como título. La entrada `Zhang_2023` queda corregida a ese título de portada; el resto de campos se conserva.

## Método

En un EA híbrido secuencial, el instante de cambio asigna el presupuesto entre exploración y explotación y es un parámetro estructural, no numérico como F o CR (pp. 2–3). Modelan la búsqueda como proceso de decisión de Markov y entrenan Q-learning y deep Q-learning. Instancian DQ-HSES y Q-LSHADE. La Fig. 5 visualiza votos en 10, 30, 50 y 100 dimensiones: no cambiar a CMA-ES, cambiar, o acción aleatoria (p. 20). La Tabla XII compara tiempos de CPU de HSES y DQ-HSES.

## Qué sostiene el texto

En CEC 2014 y 2018, DQ-HSES y Q-LSHADE superan de forma significativa a HSES y LSHADE, y DQ-HSES queda por delante del conjunto comparado en general (p. 3). No se copia un ranking función a función ausente en el pasaje de contribuciones.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El costo de CPU del agente aparece en la Tabla XII; el marco se demuestra en dos híbridos secuenciales, no en cualquier EA. |
| Lectura crítica | Controlar el cambio de fase no es seleccionar un operador de bajo nivel ni un brazo de bandit. CEC 2014/2018 no incluye dinámica tipo Lehre, ni CVRP, ni energía. La portada Leiden no añade autores. No hay DTW ni HRES. |

## Para el estado del arte

Referencia para parámetros estructurales de híbridos secuenciales. Distinguir este Q-learning del de Hao (topología PSO) y del de Seyyedabbasi (fase exploración/explotación dentro de GWO).

Etiquetas secundarias: DQN; HSES; LSHADE; CMA-ES; CEC 2014; CEC 2018.

## Evidencia

- p. 2: título de revista, DOI y resumen del cambio de fase.
- p. 3: contribuciones, DQ-HSES, Q-LSHADE y suites.
- p. 20: Fig. 5, Tabla XII y conclusión.

## Procedencia del archivo

- La clave propuesta `La portada de repositorio de Leiden (p. 1 del PDF) antecede al artículo. La entrada `Zhang_2023` duplica la etiqueta `[Research Frontier]` en el título; el título del artículo no la repite.` ya identificaba otro estudio. Esta ficha y su entrada BibTeX usan `Zhang_2023`.
