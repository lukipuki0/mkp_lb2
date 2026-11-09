# P040 — Microgrid multienergía dimensionada con E-PSO

**Título:** Optimal sizing and techno-economic analysis of energy- and cost-efficient standalone multi-carrier microgrid

**Cita:** `Lorestani_2019` — Alireza Lorestani, G. B. Gharehpetian y Mohammad Hassan Nazari (2019). *Energy*, 178, 751–764. DOI: 10.1016/j.energy.2019.04.152.

**PDF:** [PDF](./Optimal%20sizing%20and%20techno-economic%20analysis%20of%20energy-%20and%20cost-efficient%20standalone%20multi-carrier%20microgrid.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

## Qué aporta

Dimensiona una microrred aislada multienergía (eléctrica y térmica) con FV-térmico, eólica, microturbina, batería, almacenamiento térmico y caldera de gas. El optimizador es E-PSO, un enjambre con componente evolutiva, más una gestión por reglas que prioriza renovables. No es un controlador neuronal.

## Problema y método

Minimiza el costo anual total, con degradación de batería y confiabilidad, sobre datos de Rafsanjan (Irán). Compara configuraciones y sensibiliza almacenamientos, precio del gas y tasa de interés.

## Hallazgos del texto

La configuración completa deja un vertido del 3 % eléctrico y del 6 % térmico, y es la que los autores juzgan más eficiente en energía y costo para ese sitio. Sin el FV-térmico, costo y vertido suben 35,68 % y 5 %. Sin eólica, el costo anual sube 17,06 %. Juntar dos cogeneraciones —panel FV-térmico y microturbina— baja costo anual y vertido más de un 25 %. El óptimo de almacenamiento citado es 302,4 kWh eléctricos y 867 kWh térmicos. Frente a planta de vapor y gas natural, la Tabla 7 cifra la reducción de CO2 en 404 910,59 kg/año (404 t/año). E-PSO encuentra mejor solución, converge antes y con menos varianza que otras metaheurísticas usadas como referencia; el texto no detalla aquí el protocolo completo de esas bases.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La gestión es por reglas, no una optimización horaria conjunta como el MILP de Malheiro. El caso es un clima y una tarifa de gas. |
| Lectura crítica | “Más robusto que otras metaheurísticas” no se puede reutilizar sin las tablas de varianza, que esta ficha no reproduce porque no se extrajeron las celdas. No hay hidrógeno ni DTW. E-PSO no es una red entrenada con PSO: el enjambre dimensiona equipos. |

## Uso en el estado del arte

Extiende el dimensionamiento híbrido al acoplamiento eléctrico–térmico y a la degradación de batería, frente a los casos solo eléctricos o con hidrógeno del mismo lote.

Etiquetas secundarias: PVT, cogeneración, Rafsanjan, degradación, CO2.

## Evidencia

- PDF p. 1: componentes, TAC, EMS por reglas, E-PSO y Rafsanjan.
- PDF p. 13: vertidos 3 % y 6 %, sensibilidades 35,68 %, 17,06 % y más del 25 %, almacenamientos 302,4 y 867 kWh.
- PDF p. 14: Tabla 7, reducción de CO2 de 404 910,59 kg/año.
