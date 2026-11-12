# P029 — Modelo probabilístico de un híbrido solar–eólico

**Título:** Hybrid solar/wind power system probabilistic modelling for long-term performance assessment

**Cita:** `Tina_2006` — G. Tina, S. Gagliano y S. Raiti (2006). *Solar Energy*, 80(5), 578–588. DOI: 10.1016/j.solener.2005.03.013.

**PDF:** [PDF](./Hybrid%20solar%EF%BC%8Fwind%20power%20system%20probabilistic%20modelling%20for%20long-term%20performance%20assessment.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

## Qué aporta

Modelo analítico, por convolución de densidades, del desempeño de largo plazo de un sistema híbrido solar–eólico, aislado o conectado a red. No es una metaheurística ni un simulador horario: busca una forma cerrada alternativa a la simulación en el tiempo.

## Problema y método

Una sola fuente renovable obliga a sobredimensionar. El híbrido se evalúa con el índice de confiabilidad energética (EIR), ligado a la energía esperada no suministrada (EENS). Se deducen expresiones de la potencia conjunta de aerogenerador y FV. El periodo de estudio puede ser el año o una hora-tipo, para incorporar el valor temporal de la energía. El ejemplo numérico compara el modelo con simulación en el dominio del tiempo. Una figura de referencia usa demanda pico 40 kW, velocidades características 4,5 / 16 / 60 m/s y 40 kWp fotovoltaicos.

## Hallazgos del texto

La diferencia porcentual entre el EIR analítico y el numérico, en las configuraciones comparadas, queda entre 0,5 % y 1,9 %. Los autores concluyen que esa aproximación basta para estimar el promedio de largo plazo y para apoyar el dimensionamiento de un híbrido conectado a red sin baterías, incluyendo demanda contratada, costo anual esperado y energía. El resultado no es un tamaño óptimo único: es la relación entre parámetros y EIR.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El ejemplo ilustra el modelo; no agota climas ni incluye almacenamiento en el caso conectado que destacan. |
| Lectura crítica | Es anterior a la ola de metaheurísticas de dimensionamiento y no compite con ellas en el mismo terreno: evita la simulación cronológica a costa de hipótesis de densidad (claridad solar, viento). No hay hidrógeno, control adaptativo ni DTW. Una diferencia del 1,9 % valida el modelo probabilístico frente a Monte Carlo, no un algoritmo de búsqueda. |

## Uso en el estado del arte

Antecedente de evaluación de confiabilidad de híbridos FV–eólico sin tratar el problema como caja negra optimizable. Útil para separar modelo de recurso y método de solución.

Etiquetas secundarias: EIR/EENS, convolución, sin batería en el caso de red.

## Evidencia

- PDF p. 1: convolución, EIR/EENS y comparación con simulación temporal.
- PDF p. 9: diferencia 0,5–1,9 % y conclusiones sobre dimensionamiento sin baterías.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (/) por equivalentes de ancho completo. El título de la ficha permanece completo.
