# Optimal sizing of standalone rural microgrid for sustainable electrification with renewable energy resources

Ficha de lectura para el estado del arte. PDF: [PDF](./Optimal%20sizing%20of%20standalone%20rural%20microgrid%20for%20sustainable%20electrification%20with%20renewable%20energy%20resources.pdf). Entrada bibliográfica: [referencias.bib](../referencias.bib). Clave BibTeX: `Kamal_2023`.

Etiquetas secundarias: evolución diferencial como optimizador de dimensionamiento, sensibilidad económica, electrificación rural. Sección primaria: sistemas híbridos renovables, dimensionamiento y operación.

## Qué aporta

Kamal, Ashraf y Fernandez (2023) dimensionan una microrred aislada para una comunidad rural de Uttarakhand (India) con recursos locales: fotovoltaica, microhidráulica, biogás, biomasa, eólica y batería. El optimizador es evolución diferencial (DE). No hay hidrógeno ni un algoritmo híbrido de metaheurísticas: DE, PSO y GA se comparan como solucionadores del mismo problema de coste.

## Problema, método y contribución

El problema es cubrir una demanda rural con bajo coste de la energía y sin red fiable. El modelo usa la disponibilidad local y la intermitencia. Se minimizan coste de la energía y tamaño; se reportan también el TNPC y la sensibilidad de TNPC y COE. La fiabilidad del caso optimizado se plantea como déficit nulo. DE se contrasta con PSO y GA; no se combinan en un portafolio. HOMER aparece solo como antecedente.

## Resultados que el texto sostiene

El resumen y la conclusión (pp. 1 y 19) dan, para la configuración óptima, un TNPC de 7,12,532.00 dólares —agrupación india; 712 532 USD— y un COE de 0,14 USD/kWh. Los autores afirman que DE optimiza mejor que PSO y GA y que la probabilidad de cubrir la demanda modelada es del 100 %. La conclusión sitúa el interés en microrredes con menor déficit para lugares remotos y en comparar fusiones de recursos. El trabajo futuro que ellos mismos enuncian es capacidad ante fallos y variaciones de tensión y frecuencia si hubiera integración a red (p. 19). No hay, en el texto leído, una tabla de tamaños de cada componente que pueda citarse sin ambigüedad de extracción.

## Límites

El caso es un emplazamiento y un año de simulación. La superioridad de DE se afirma frente a PSO y GA en ese modelo; no es un resultado sobre hibridación de operadores ni sobre control en línea de parámetros.

Lectura crítica: conviene no leer «DE gana a GA y PSO» como evidencia de que un controlador de metaheurísticas mejoraría este dimensionamiento. El artículo tampoco estudia hidrógeno, desalación ni DTW. Es útil como ejemplo de sizing rural multi-recurso con un evolutivo clásico y con cifras de COE/TNPC ancladas a Uttarakhand, no como validación de un proyecto distinto ni como hueco porque el resumen no mencione aprendizaje por refuerzo.

## Evidencia

Título, autores, *Sustainable Cities and Society* 88 (2023) 104298 y DOI en la p. 1. Cifras y conclusión en las pp. 1 y 19.
