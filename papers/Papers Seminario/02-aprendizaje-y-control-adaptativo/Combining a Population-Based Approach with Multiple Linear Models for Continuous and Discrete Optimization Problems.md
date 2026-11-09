# Combining a Population-Based Approach with Multiple Linear Models for Continuous and Discrete Optimization Problems

**Clave de cita:** `Vega_2022`  
**PDF:** [PDF](./Combining%20a%20Population-Based%20Approach%20with%20Multiple%20Linear%20Models%20for%20Continuous%20and%20Discrete%20Optimization%20Problems.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Emanuel Vega; Ricardo Soto; Pablo Contreras; Broderick Crawford; Javier Peña; Carlos Castro |
| Año | 2022 |
| Venue | Mathematics, 10(16), 2920 |
| DOI | 10.3390/math10162920 |
| Sección primaria | `02-aprendizaje-y-control-adaptativo` |
| Etiquetas | LMPB; varios modelos lineales; población; MKP; comparación con LB2 |

## Aporte para el estado del arte

Define LMPB, el paso entre LB2 y el balance autónomo de 2024: el parámetro gobernado es el tamaño de población y el aprendizaje ya no es una sola recta, sino varios modelos lineales. Permite citar la línea sin colapsar tres artículos en uno.

## Problema y método

El problema es el mismo de los poblacionales: un tamaño fijo de población no acompaña el estado de la búsqueda y el ajuste offline se sobreajusta a instancias vistas. LMPB se organiza en tres módulos: gestión del algoritmo poblacional, aprendizaje que transforma datos dinámicos en conocimiento, y selección del tamaño que se usará a continuación. Los tamaños candidatos son esquemas con probabilidades. Los modelos son varios ajustes lineales, más complejos que la regresión simple de LB2 según la comparación que hace el propio texto. Se prueba en 15 funciones continuas y en el multidimensional knapsack problem. Los comparadores continuos incluyen WOA, DE, GSA, PSO, VLE e INMDA, y además LB2 y SHO afinado con IRace. En MKP comparan con métodos publicados y con SHO.

## Qué sostiene el texto

La conclusión dice que LMPB es competitivo en continuo y en discreto. En la comparación detallada, LMPB alcanza mejores mejores valores que LB2 y que SHO-IRace en parte del banco, y el Mann-Whitney marca superioridad estadística en un subconjunto de funciones, no en todas. El mismo pasaje señala un criterio, ligado a los valores alcanzados y a la desviación, en el que LB2 queda por delante en todas las funciones resueltas. No resumo eso como victoria global de uno u otro. En MKP el texto reporta desempeño igual o mejor que varios baselines en las tablas 14-16, siempre dentro de esas instancias.

## Límites

**Hallazgos de los autores.** Sigue siendo un marco híbrido de metaheurística y estadística, sin garantía de óptimo. El trabajo futuro que enuncian es seguir evolucionando la arquitectura, no cerrarla.

**Lectura crítica.** No es un experimento de HRES ni de DTW. La superioridad parcial frente a LB2 no autoriza a sustituir LB2 en otro dominio. Los p-valores hay que leerlos función por función en las tablas, no como un único veredicto.

## Relación con control adaptativo

Comparte con LB2 el lazo en línea datos-modelo-probabilidades, pero el objeto de control es la población. Es aprendizaje para optimizar, en la clasificación de Song, no un modelo de pronóstico energético.

## Evidencia

- p. 1: tres módulos y bancos continuo y MKP.
- pp. 20 y 24-26: comparación mixta con LB2 y conclusión.
