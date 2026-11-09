# A MILP model to design hybrid wind–photovoltaic isolated rural electrification projects in developing countries

**Clave de cita:** `FerrerMarti_2013`  
**PDF:** [PDF](./A%20MILP%20model%20to%20design%20hybrid%20wind%E2%80%93photovoltaic%20isolated%20rural%20electrification%20projects%20in%20developing%20countries.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | L. Ferrer-Martí; B. Domenech; A. García-Villoria; R. Pastor |
| Año | 2013 |
| Venue | European Journal of Operational Research, 226(2), 293-300 |
| DOI | 10.1016/j.ejor.2012.11.018 |
| Sección primaria | `04-hres-dimensionamiento-y-operacion` |
| Etiquetas | MILP; microrredes rurales; eólico-fotovoltaico; método exacto |

## Aporte para el estado del arte

Muestra el polo exacto del diseño de HRES aislados: un MILP que decide a la vez dónde poner generación eólica o fotovoltaica y cómo agrupar puntos de consumo en microrredes o sistemas individuales. Sirve de contraste con las metaheurísticas de sizing, no como competidor de un balanceador de exploración.

## Problema y método

El problema es electrificar comunidades dispersas cuando extender la red nacional es caro. El modelo minimiza la inversión inicial que cubre la demanda, con generación eólica y fotovoltaica, controladores, baterías, inversores, medidores y tipos de conductor. Las variables binarias marcan si un punto tiene generación, si existe un tramo de cable de cierto tipo y si hay medidor. El texto distingue una formulación entera y una binaria equivalente, esta última con restricciones extra para romper simetrías. No hay población, operador de movimiento ni ajuste en línea de parámetros.

## Qué sostiene el texto

Los autores dicen que instancias de tamaño realista se resuelven de forma eficiente y aplican el modelo a proyectos reales en Perú, con apoyo de Practical Action. En El Alumbre el modelo obtiene 58 674 USD frente a 75 517 USD del proyecto real: un microgrid eólico de 11 usuarios, 22 usuarios eólicos individuales y 2 fotovoltaicos individuales, contra 35 usuarios eólicos individuales. En Alto Perú, 35 843 USD frente a 51 691 USD. Esas cifras son costes de inversión de la tabla 2, no LCOE ni fiabilidad horaria.

## Límites

**Hallazgos de los autores.** El criterio es solo inversión inicial para cubrir la demanda. No optimizan operación horaria, degradación, hidrógeno ni emisiones. La comparación es contra el diseño real de esos proyectos, no contra un conjunto amplio de metaheurísticas.

**Lectura crítica.** El ahorro no se transfiere a otro país o a un campus con hidrógeno. El modelo asume datos de demanda y potencial ya estimados. No hay evidencia aquí sobre DTW, control adaptativo de metaheurísticas ni almacenamiento de hidrógeno.

## Relación con híbridos energéticos

El híbrido es tecnológico, viento más fotovoltaica y microgrid más sistemas individuales, no un híbrido algorítmico. Para un estado del arte de dimensionamiento, documenta que en instancias rurales acotadas un MILP puede mejorar diseños reales sin recurrir a búsqueda poblacional.

## Evidencia

- p. 1: objetivo, criterio de inversión y casos de Perú.
- pp. 4-5: variables binarias y función objetivo.
- p. 7: tabla 2, costes y esquemas de El Alumbre y Alto Perú.
