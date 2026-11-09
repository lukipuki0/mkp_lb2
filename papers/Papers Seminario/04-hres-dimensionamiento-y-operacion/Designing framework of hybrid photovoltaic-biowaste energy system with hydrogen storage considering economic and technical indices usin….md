# P021 — WOA para dimensionar FV, biorresiduo e hidrógeno

**Título:** Designing framework of hybrid photovoltaic-biowaste energy system with hydrogen storage considering economic and technical indices using whale optimization algorithm

**Cita:** `Sun_2022` — Hongyue Sun, Abdol Ghaffar Ebadi, Mohsen Toughani, Saber Arabi Nowdeh, Amirreza Naderipour y Aldrin Abdullah (2022). *Energy*, 238, 121555. DOI: 10.1016/j.energy.2021.121555.

**PDF:** [PDF](./Designing%20framework%20of%20hybrid%20photovoltaic-biowaste%20energy%20system%20with%20hydrogen%20storage%20considering%20economic%20and%20technical%20indices%20usin%E2%80%A6.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

## Qué aporta

Dimensiona un sistema aislado fotovoltaico–biorresiduo–pila de combustible con almacenamiento de hidrógeno mediante el algoritmo de ballenas (WOA). No entrena una red neuronal: WOA es una metaheurística de enjambre. El caso es Ardabil (Irán), con vida útil de 20 años.

## Problema y método

Se minimiza el costo presente neto total (TNPC: inversión, mantenimiento y reemplazo) sujeto a confiabilidad (LPSP; también se informan LOLE y LOEE). El “almacenamiento” es electrolizador, tanque de hidrógeno y pila, no una batería. Las variables cubren unidades de biorresiduo, potencia FV, electrolizadores, tanques y pilas. Se compara con PSO bajo escenarios de disponibilidad y de costo de capital de la pila.

## Hallazgos del texto

Con disponibilidad de inversor igual a 1, la Tabla 12 da TNPC 2,820 M$, COE 0,5238 $/kWh, LPSP 0,0029, LOLE 220 h/año y LOEE 0,974 MWh/año. Al bajar esa disponibilidad a 0,99, TNPC sube a 3,159 M$, COE a 0,5868 $/kWh y LPSP a 0,0041. Sumar FV reduce el TNPC; WOA mejora a PSO; un mayor costo de capital de la pila reduce su potencia y empeora costo y confiabilidad. La Tabla 15 sitúa 0,5238 $/kWh por debajo de otros híbridos citados (0,83–1,87 $/kWh) en otras ciudades.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La limitación declarada es el acceso a datos horarios de irradiancia y temperatura. El trabajo futuro propuesto es un diseño multicriterio con incertidumbre de demanda. |
| Lectura crítica | El propio texto advierte que el COE compara el potencial del sitio, no la calidad del algoritmo entre regiones. No hay EMS horario validado en campo, ni control adaptativo, ni DTW. No valida un proyecto distinto del caso de Ardabil. |

## Uso en el estado del arte

Precedente de metaheurística de enjambre para dimensionar un híbrido con hidrógeno y restricción de confiabilidad. Contrasta con el MILP de Malheiro y con el NSGA-II de Tezer, del mismo lote.

Etiquetas secundarias: hidrógeno, LPSP/LOLE, metaheurística de enjambre, Irán.

## Evidencia

- PDF p. 1: objetivo TNPC, restricción LPSP, componentes de hidrógeno y horizonte de 20 años.
- PDF p. 19: implementación del WOA sobre las variables de diseño.
- PDF p. 26: Tabla 12, sensibilidad a la disponibilidad del inversor.
- PDF p. 27–28: costo de la pila, limitación de datos y Tabla 15 (COE 0,5238 $/kWh, Ardabil).

## Procedencia del archivo

- El nombre de archivo se acortó de forma visible con `…` para caber en la ruta de Windows. El título completo está en el encabezado de esta ficha.
