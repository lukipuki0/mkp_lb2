# Autonomous Parameter Balance in Population-Based Approaches: A Self-Adaptive Learning-Based Strategy

**Clave de cita:** `Vega_2024`  
**PDF:** [PDF](./Autonomous%20Parameter%20Balance%20in%20Population-Based%20Approaches%EF%BC%9A%20A%20Self-Adaptive%20Learning-Based%20Strategy.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Emanuel Vega; José Lemus-Romani; Ricardo Soto; Broderick Crawford; Christoffer Löffler; Javier Peña; El-Ghazali Talbi |
| Año | 2024 |
| Venue | Biomimetics, 9(2), 82 |
| DOI | 10.3390/biomimetics9020082 |
| Sección primaria | `02-aprendizaje-y-control-adaptativo` |
| Etiquetas | población adaptativa; LBLP; problemas discretos |

## Aporte para el estado del arte

Pasa del balance de esquemas de movimiento al balance del tamaño de población. Es el antecedente inmediato para argumentar que un parámetro estructural, no solo un operador, puede gobernarse con datos generados durante la búsqueda. La copia extra se eliminó: esta es la única ficha.

## Problema y método

El problema es elegir el tamaño de población cuando el problema discreto no se ha explorado y el ajuste manual o el tuning offline no acompañan el cambio de paisaje. La estrategia tiene tres piezas: un componente de optimización que ejecuta la metaheurística, un componente de aprendizaje que convierte datos dinámicos en conocimiento, y un selector probabilístico que cambia la población. En la instanciación revisada, LBLP define cuatro tamaños como esquemas, entre ellos 20 y 30, y un umbral β. El aprendizaje usa modelos estadísticos de complejidad moderada, no una red profunda. Los bancos son Manufacturing Cell Design Problem, Set Covering Problem y Multidimensional Knapsack Problem. Los comparadores incluyen versiones clásicas, búsqueda autónoma e IRace, y más adelante LMPB.

## Qué sostiene el texto

El resumen y la conclusión dicen que la estrategia compite con metaheurísticas clásicas, autónomas y afinadas con IRace en esos tres problemas, y que el ajuste dinámico del número de soluciones es una vía abierta. Las tablas de Wilcoxon-Mann-Whitney y de RPD comparan LBLP, LMPB, SHO clásico y SHO con IRace. El texto reconoce tramos en los que LBLP no aprovecha el tamaño de población frente a la diversificación y en los que LMPB obtiene mejores valores. No debe leerse como dominancia uniforme.

## Límites

**Hallazgos de los autores.** El dominio es combinatorio discreto. No hay funciones continuas en el protocolo principal ni plantas energéticas. La conclusión ata la mejora a la complejidad moderada de los modelos estadísticos usados.

**Lectura crítica.** Duplicar el PDF no duplica la evidencia. No muestra que el mismo selector sirva para series DTW o para capacidad de electrolizadores. Comparar con LMPB exige citar también el artículo de 2022, porque aquí LMPB es baseline, no el objeto definido.

## Relación con control adaptativo

Es control en línea de un parámetro poblacional mediante aprendizaje sobre la traza, distinto de un híbrido que solo mezcla dos bioinspirados. El vínculo con Talbi es de autoría y de encaje taxonómico, no una validación de la taxonomía de 2002.

## Evidencia

- p. 1: tres componentes, problemas discretos y comparadores.
- pp. 4-5: esquemas de tamaño y β en LBLP.
- p. 29: conclusión sobre el tamaño de población como parámetro general.

## Procedencia del archivo

- Única ficha de `Vega_2024`. La copia extra se eliminó.
- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
