# P037 — ε-constraint adaptativo y TOPSIS para rutas multimodales

**Título:** Multi-Objective Optimization of the Multimodal Routing Problem Using the Adaptive ε-Constraint Method and Modified TOPSIS with the D-CRITIC Method

**Cita:** `Maneengam_2023` — Apichit Maneengam (2023). *Sustainability*, 15(15), 12066. DOI: 10.3390/su151512066.

**PDF:** [PDF](./Multi-Objective%20Optimization%20of%20the%20Multimodal%20Routing%20Problem%20Using%20the%20Adaptive%20%CE%B5-Constraint%20Method%20and%20Modified%20TOPSIS%20w%E2%80%A6.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

## Qué aporta

Método de decisión para el ruteo multimodal de graneles en Tailandia: genera el frente con el ε-constraint adaptativo de Kirlik y Sayın y lo ordena con TOPSIS modificado y pesos D-CRITIC. No es una metaheurística poblacional ni un problema energético.

## Problema y método

El operador debe minimizar a la vez costo, tiempo y emisiones CO2-equivalente en una red con origen en el fondeadero del golfo de Tailandia y destino en planta. Hoy el ε-constraint clásico no dice qué ruta ofrecer al cliente. El método tiene tres fases: frente completo, pesos por correlación de distancias (D-CRITIC) y ranking TOPSIS. Las cotas de ε salen de resolver cada objetivo con un solver lineal.

## Hallazgos del texto

La Tabla 3 fija el tiempo entre 64 y 150 y el CO2e entre 215 171,92 y 751 698,36. El frente usado en el ranking tiene siete soluciones. La de mayor puntaje es la número 7, con 0,5929: 472,73 baht/t, 67 días y 225 396,08 kgCO2eq. El orden de oferta al cliente sigue 7, 4, 5, 3, 6, 1 y 2. Los autores dicen que el enfoque adaptativo encuentra más soluciones Pareto y mejor relación tiempo de cómputo / número de soluciones que el ε-constraint que ya usaba el operador, y que además entrega un ranking que el método vigente no daba.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Es un caso real de negociación, no un banco público de ruteo. El ranking depende de los pesos D-CRITIC de ese frente. |
| Lectura crítica | Superioridad “en número de soluciones y en ratio de tiempo” no viene acompañada, en las páginas leídas, de la tabla de tiempos; no invento esa cifra. No hay HRES, hidrógeno, control ni DTW. Sirve como uso aplicado de ε-constraint, emparentado con Mavrotas, no como evidencia de hibridación metaheurística. |

## Uso en el estado del arte

Ejemplo de cómo pasar de un frente a una alternativa ordenada cuando el decisor negocia. Útil en la sección multiobjetivo, no en la de dimensionamiento renovable.

Etiquetas secundarias: Tailandia, CO2e, D-CRITIC, TOPSIS, Kirlik–Sayın.

## Evidencia

- PDF p. 1: tres fases y caso del golfo de Tailandia.
- PDF p. 14: Tabla 3, cotas de tiempo y CO2e.
- PDF p. 16: Tabla 6, solución 7 con puntaje 0,5929, y conclusiones.

## Procedencia del archivo

- El nombre de archivo se acortó de forma visible con `…` para caber en la ruta de Windows. El título completo está en el encabezado de esta ficha.
