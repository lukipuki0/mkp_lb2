# A hybridizing-enhanced differential evolution for optimization

**Clave de cita:** `Ghasemi_2023`  
**PDF:** [PDF](./peerj-cs-1420.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Mojtaba Ghasemi, Mohsen Zare, Pavel Trojovský, Amir Zahedibialvaei, Eva Trojovská |
| Año | 2023 (enviado 15-12-2022, publicado 01-06-2023) |
| Venue | PeerJ Computer Science, 9, e1420 |
| DOI | 10.7717/peerj-cs.1420 |
| Sección primaria | `07-hibridos-gwo-de` |
| Etiquetas | HDE; vector de caza GWO como mutación DE; probabilidad fija `Hm`; jHDE; CEC-2014; CEC-2019 |

## Aporte para el estado del arte

Es el antecedente más citado del grupo para meter GWO dentro de DE: el vector de caza de GWO sustituye, con probabilidad `Hm`, a la mutación DE de cada individuo, y después actúan el cruce y la selección greedy de DE. El artículo de 2026 sobre SEIG lo presenta como precedente directo del acoplamiento GWO-DE.

## Problema y método

Es optimización continua de caja negra. HDE parte de DE con una de ocho estrategias de mutación (ecs. 2-9, p. 7). En cada iteración y para cada miembro, si `rand < Hm` el mutante es el vector de caza de GWO construido con alfa, beta y delta (ecs. 22-24, p. 10); si no, se usa la mutación DE original (pasos 6-7, p. 11). Luego vienen el cruce binomial (ec. 10, p. 8) y la selección greedy (ec. 11, p. 8), con criterio de parada `NFEmax = Itermax · Npop` (p. 10).

Hay una sola población y un solo candidato por individuo: la moneda `Hm` elige su origen, no hay competencia entre dos candidatos evaluados. `Hm` y `CR` son constantes por variante (Tabla 3, p. 13), `F = 0.1 + 0.8·rand` y `a` baja de 2 a 0 (pp. 8 y 13). No hay señal de estancamiento ni reinicio; la élite se conserva solo por la selección greedy. En jHDE, la mutación de jDE se reemplaza por la de caza y se heredan los `F` y `CR` autoadaptados de jDE (pp. 26-31). El texto no indica qué `Hm` usa jHDE.

## Qué sostiene el texto

En CEC-2014, con `D = 30`, 300 000 evaluaciones, población 30 y 30 corridas (p. 13), comparan ocho variantes HDE con sus DE de origen y con GWO (Tabla 4, pp. 14-18), con rangos y prueba de Friedman (Tabla 5, p. 19). Las tres mejores globales son HDE/current-to-best/1, HDE/rand-to-best/2 y HDE/rand/1 (p. 19). En CEC-2019 destacan HDE/current-to-rand/1 y HDE/rand-to-best/1 (Tabla 6, p. 24). HDE/current-to-best/1 se compara con siete algoritmos publicados (Tabla 7, pp. 24-26). La prueba de rangos con signo de Wilcoxon sobre las 30 funciones (Tabla 8, p. 27) cubre seis de esos siete algoritmos (LJA no aparece) y es significativa contra cinco; contra RW-GWO no lo es (`p = 0.274`). jHDE mejora a jDE en 17 funciones, empeora en 3 y empata en 10, con población 60 y 5000 iteraciones (Tabla 9, pp. 27 y 31).

## Límites

**Hallazgos de los autores.** Señalan como principal debilidad la falta de ajuste adaptativo de `CR` y `Hm`, que dejan como trabajo futuro (p. 31).

**Lectura crítica.** `Hm` toma 0.1, 0.5 o 0.9 según la variante (Tabla 3) sin protocolo de ajuste declarado, así que parte de la ventaja puede venir de esa elección. Los signos "+" y "−" de la Tabla 4 comparan cada HDE con su DE, pero el texto no dice si responden a una prueba por función o solo a medias (p. 18). Tampoco aclara si los resultados de la Tabla 7 fueron reejecutados o copiados de las fuentes. La p. 20 habla de "CEC-2017" en un análisis de CEC-2014. La Tabla 4 sale desalineada en la extracción de texto, así que esta ficha no transcribe cifras por función.

## Relación con el GWO-DE actual

Comparte población única, alfa, beta y delta como guía, y selección greedy de DE. Difiere en que cada individuo produce un solo candidato cuyo origen elige una probabilidad fija; en el GWO-DE actual se generan y evalúan ambos y gana el mejor. Ningún parámetro de HDE lee la trayectoria de la búsqueda, y los autores señalan la adaptación de `Hm` y `CR` como la carencia principal. El GWO-DE actual explora la adaptación de los parámetros por otra vía (perfiles conjuntos de `a`, `F` y `CR`), y esta fuente no permite comparar las dos.

## Evidencia

- p. 1: resumen y variantes destacadas.
- p. 5: idea del vector de caza como mutación auxiliar con probabilidad por miembro.
- p. 7: estrategias de mutación DE (ecs. 2-9).
- p. 8: cruce (ec. 10), selección greedy (ec. 11), coeficientes GWO (ecs. 14-15).
- p. 10: vector de caza (ecs. 22-24) y criterio `NFEmax`.
- p. 11: pseudocódigo, pasos 0-15.
- p. 13: protocolo CEC-2014, `F` y Tabla 3.
- pp. 14-18: Tabla 4.
- p. 19: Tabla 5 y ranking.
- p. 20: CEC-2019 y descripción de Wilcoxon.
- p. 24: Tabla 6.
- pp. 26-27: jHDE y Tablas 8-9.
- p. 31: resultado jHDE y conclusión.
