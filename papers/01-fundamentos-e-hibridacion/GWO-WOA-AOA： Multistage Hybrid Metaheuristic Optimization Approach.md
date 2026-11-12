# P025 — Híbrido secuencial GWO, WOA y AOA en CEC 2019

**Título:** GWO-WOA-AOA: Multistage Hybrid Metaheuristic Optimization Approach

**Cita:** `Lale_2025` — Timur Lale (2025). *2025 9th International Symposium on Innovative Approaches in Smart Technologies (ISAS)*, pp. 1–8. IEEE. DOI: 10.1109/ISAS66241.2025.11101906.

**PDF:** [PDF](./GWO-WOA-AOA%EF%BC%9A%20Multistage%20Hybrid%20Metaheuristic%20Optimization%20Approach.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

## Qué aporta

Híbrido de optimización en tres fases: lobo gris (GWO) explora, ballena (WOA) explota y el algoritmo aritmético (AOA) afina. Es hibridación de metaheurísticas, no un modelo neuronal entrenado con un optimizador de pesos.

## Problema y método

El texto argumenta que cada algoritmo puro equilibra mal exploración y explotación. La búsqueda se parte en etapas sucesivas, con actualización local o de ajuste fino según un MOP que decrece despacio (parámetro 0,1) y un umbral aleatorio 0,5. La prueba son 10 funciones CEC 2019. Población 20 y 50 corridas para todos. En el híbrido cada etapa usa 100 iteraciones (300 en total); las bases ALO, PSO, DE, GWO y WOA también corren 300 iteraciones.

## Hallazgos del texto

En el polinomio de Chebyshev el híbrido alcanza mejor valor 1,000e+00, media 1,000e+00 y desviación 0, frente a medias de GWO 1,016e+06 y WOA 4,082e+07. En Lennard-Jones el mejor del híbrido es 1,035e+00 (GWO 1,706e+00, WOA 4,285e+00). En Rastrigin el mejor es 8,778e+01 frente a 9,961e+01 de GWO. En Griewank la media del híbrido es 2,837e+03, menor que 9,287e+03 de GWO y 4,405e+05 de WOA. La conclusión afirma convergencia más rápida y mayor precisión que los componentes y que las bases elegidas.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La evaluación queda en funciones de prueba de dimensión moderada (hasta 18 en Lennard-Jones). No se reporta un problema con restricciones de ingeniería. |
| Lectura crítica | El presupuesto de 300 iteraciones iguala el total, pero no demuestra que la ganancia venga del acoplamiento y no del orden fijo de fases. No hay aplicación a HRES, hidrógeno, control ni DTW. Un buen CEC 2019 no autoriza, por el teorema de no free lunch, a declararlo mejor en dimensionamiento energético. |

## Uso en el estado del arte

Ejemplo reciente de hibridación secuencial de tres enjambres, útil para contrastar con híbridos global–local (El-Hashash) y con selección aprendida de operadores (Pei).

Etiquetas secundarias: CEC 2019, exploración–explotación, enjambre.

## Evidencia

- PDF p. 1: esquema de tres fases y banco CEC 2019.
- PDF p. 5: población 20, 50 corridas y 100 iteraciones por etapa.
- PDF p. 6–7: medias, mejores valores y desviaciones citadas; conclusión.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
