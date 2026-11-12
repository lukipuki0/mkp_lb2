# A review of hydrogen production and supply chain modeling and optimization

**Clave de cita:** `Riera_2023`  
**PDF:** [PDF](./A%20review%20of%20hydrogen%20production%20and%20supply%20chain%20modeling%20and%20optimization.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Jefferson A. Riera; Ricardo M. Lima; Omar M. Knio |
| Año | 2023 |
| Venue | International Journal of Hydrogen Energy, 48(37), 13731-13755 |
| DOI | 10.1016/j.ijhydene.2022.12.242 |
| Sección primaria | `05-hidrogeno-y-contexto-energetico` |
| Etiquetas | revisión; cadena de suministro; HSCD; HPD; MILP; incertidumbre |

## Aporte para el estado del arte

Delimita qué se optimiza cuando el objeto es hidrógeno y no una microrred doméstica: producción, captura de carbono, transporte, almacenamiento e integración con otros sectores. Sirve para no confundir un sizing de electrolizador local con un modelo de cadena de suministro.

## Problema y método

Es una revisión, no un caso resuelto por los autores. Separan diseño de cadena (HSCD) y diseño de proceso de producción (HPD). Recorren materias primas, rutas de producción, resolución temporal y espacial, inversión en una etapa o en varias, y medidas de desempeño. En métodos de solución describen el patrón de la literatura: HSCD determinista de un objetivo suele formularse como MILP o reformulación lineal y resolverse con branch-and-bound comercial; algunos LP usan símplex. Los estocásticos de un objetivo revisados también son MILP. Los HPD suelen ser no lineales: citan, entre otros, un algoritmo genético en dos niveles para un sistema fotovoltaico-electrolizador alcalino y otro genético para fotovoltaico-PEM. Esos GA son ejemplos ajenos, no un algoritmo nuevo de este artículo.

## Qué sostiene el texto

El texto distingue snapshot de un año objetivo, que reduce tamaño pero pierde inversiones intermedias, y modelos multietapa, más fieles a la demanda creciente y más pesados. Resume que las proyecciones de demanda usadas en la literatura dependen sobre todo de vehículos de pila de combustible. Las cifras de IRENA, 6 % de la demanda energética mundial, y del Hydrogen Council, cerca del 18 %, se citan como estimaciones externas, no como cálculo de los autores. Las direcciones que proponen son proyecciones de demanda más holísticas, modelado sectorial integrado, mejor resolución espacio-temporal, clima y métodos para modelos más sofisticados.

## Límites

**Hallazgos de los autores.** No rankean un solver universal. Advierten que añadir estocasticidad puede exigir descomposición y que el método de pesos depende fuertemente de los coeficientes.

**Lectura crítica.** No evalúa metaheurísticas poblacionales de control ni DTW. Un GA citado en un HPD no convierte esta revisión en evidencia de que los híbridos superan a CPLEX. La ausencia de LB2 en el resumen no es un hueco demostrado.

## Relación con hidrógeno y optimización

Documenta que la optimización de hidrógeno a escala de cadena es mayoritariamente programación matemática, mientras la hibridación metaheurística aparece de forma puntual en diseños de proceso. Útil para separar contexto energético de un lazo de búsqueda adaptativa.

## Evidencia

- pp. 1-3: alcance HSCD/HPD y estimaciones citadas de demanda.
- pp. 17-19: inversión en una o varias etapas, solvers y lagunas declaradas.
