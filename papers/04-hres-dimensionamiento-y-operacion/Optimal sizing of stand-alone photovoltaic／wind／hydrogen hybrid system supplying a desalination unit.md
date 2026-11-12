# Optimal sizing of stand-alone photovoltaic/wind/hydrogen hybrid system supplying a desalination unit

Ficha de lectura para el estado del arte. PDF: [PDF](./Optimal%20sizing%20of%20stand-alone%20photovoltaic%EF%BC%8Fwind%EF%BC%8Fhydrogen%20hybrid%20system%20supplying%20a%20desalination%20unit.pdf). Entrada bibliográfica: [referencias.bib](../referencias.bib). Clave BibTeX: `Smaoui_2015`.

Etiquetas secundarias: hidrógeno como almacenamiento, desalación, dimensionamiento determinista. Sección primaria: sistemas híbridos renovables, dimensionamiento y operación.

## Qué aporta

Smaoui, Abdelkafi y Krichen (2015) dimensionan por búsqueda iterativa, no por una metaheurística, un FV/eólico aislado con hidrógeno que alimenta una desaladora en Kerkennah (Túnez). El óptimo es la menor inversión inicial entre configuraciones que cubren toda la demanda de agua.

## Problema, método y contribución

La intermitencia solar y eólica no cubre hora a hora la desalación, y las baterías se descartan por vida útil, mantenimiento y contenido ambiental. El almacenamiento es electrolizador, tanque de hidrógeno y pila. Hay seis variables: potencia eólica, número de paneles, potencias de pila y electrolizador, capacidad del tanque y tamaño de la desaladora.

El método, secciones 3 y 4, tiene modelos horarios, análisis de fiabilidad y análisis económico. No es un algoritmo genético ni un híbrido de optimizadores. Citan el GA de Koutroulis et al. y el multiobjetivo de Dufo-López et al. como antecedentes, y usan una técnica iterativa que retiene la configuración factible de menor inversión inicial. Los datos horarios de meteorología y carga se presentan como aplicables a cualquier emplazamiento.

## Resultados que el texto sostiene

En Kerkennah la complementariedad estacional permite recomendar el híbrido frente al solo eólico: este último exige más tanque y no mejora la autonomía estival (pp. 12-13). La configuración de menor inversión citada en el cuerpo es una turbina de 1250 kW, 3000 paneles, electrolizador de 450 kW, pila de 500 kW, tanque de 10 078 kg (707,7 m³) y desaladora de 1050 kW (p. 12). La figura 10 muestra que la desaladora opera casi siempre a potencia nominal, que el electrolizador trabaja a plena carga en invierno y que el depósito de agua cubre los déficits (pp. 13-14). La conclusión afirma que todas las configuraciones retenidas garantizan autonomía total y que el método evita estimaciones gruesas de los seis tamaños (p. 14).

## Límites

No hay frente de Pareto ni comparación del coste nivelado con la red o con baterías. El óptimo es solo de inversión inicial.

Lectura crítica: no se demuestra un barrido del catálogo comercial ni robustez ante degradación, precio del agua u otros años meteorológicos. No hay DTW ni control de un optimizador. Sirve como caso FV/eólico/H₂ con desalación, no como validación de una metaheurística híbrida ni como hueco deducido del silencio del resumen.

## Evidencia

Título, autores, *Solar Energy* 120 (2015) 263-276 y DOI en la p. 1 del PDF. Resultados y conclusión en las pp. 12-14.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (/) por equivalentes de ancho completo. El título de la ficha permanece completo.
