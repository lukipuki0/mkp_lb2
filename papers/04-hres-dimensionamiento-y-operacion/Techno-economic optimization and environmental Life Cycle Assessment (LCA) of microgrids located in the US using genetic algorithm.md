# Techno-economic optimization and environmental Life Cycle Assessment (LCA) of microgrids located in the US using genetic algorithm

Ficha de lectura para el estado del arte. PDF: [PDF](./Techno-economic%20optimization%20and%20environmental%20Life%20Cycle%20Assessment%20%28LCA%29%20of%20microgrids%20located%20in%20the%20US%20using%20genetic%20algorithm.pdf). Entrada bibliográfica: [referencias.bib](../referencias.bib). Clave BibTeX: `Nagapurkar_2019`.

Etiquetas secundarias: algoritmo genético como solucionador, LCA, hidrógeno en el espacio de diseño, economías de escala. Sección primaria: sistemas híbridos renovables, dimensionamiento y operación.

## Qué aporta

Nagapurkar y Smith (2019) dimensionan microrredes en Tucson, Lubbock y Dickinson con un algoritmo genético que minimiza el LCOE y añaden un análisis de ciclo de vida de gases de efecto invernadero. El GA es el solucionador del dimensionamiento, no un híbrido de metaheurísticas ni un modelo neuronal.

## Problema, método y contribución

La microrred combina fotovoltaica, aerogeneradores, baterías de plomo-ácido, biodiésel, pilas, electrolizadores y tanques de H₂. Tras validar modelos horarios, un despacho y un GA buscan el menor LCOE en un espacio no convexo. Para una vivienda comprueban el GA con búsqueda exhaustiva y luego escalan a 10 y 50 viviendas. La LCA pasa el inventario anual a kg CO₂ eq. y lo compara con la red. El óptimo económico y el indicador ambiental van separados: no es un ε-constraint ni un NSGA-II.

## Resultados que el texto sostiene

El resumen (p. 1) sitúa los LCOE entre 0,32 y 0,42 USD/kWh y afirma que la huella de carbono es extremadamente baja, aproximadamente una décima parte de la red convencional equivalente. Esos rangos cubren las tres ciudades y los tamaños de comunidad estudiados; no deben leerse como un único diseño. La conclusión (p. 16) repite que la metodología se aplicó a Tucson, Lubbock y Dickinson y anuncia cinco conclusiones de diseño. Las tablas de configuraciones (a partir de la p. 16) están desalineadas en la extracción de texto; esta ficha no cita conteos de pilas o tanques a partir de esas columnas.

## Límites

Costes de capital locales cambian el LCOE; los autores lo advierten al comparar con literatura. La búsqueda exhaustiva solo verifica el caso de una vivienda.

Lectura crítica: incluir electrolizador, pila y tanque en el espacio de diseño no implica que el óptimo de LCOE los seleccione en cada ciudad. Tampoco prueba que un GA supere a PSO o a un MILP en estos sitios: la verificación es contra búsqueda exhaustiva en el caso pequeño. No hay DTW ni aprendizaje. Útil como caso estadounidense de sizing con hidrógeno en el catálogo de componentes y con LCA explícita.

## Evidencia

Título, autores, *Energy Conversion and Management* 181 (2019) 272-291 y DOI en la p. 1. Rango de LCOE y huella en el resumen, p. 1. Conclusiones a partir de la p. 16.
