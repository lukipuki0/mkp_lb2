# Review on recent optimization strategies for hybrid renewable energy system with hydrogen technologies: State of the art, trends and future directions

Ficha de lectura para el estado del arte. PDF: [PDF](./Review%20on%20recent%20optimization%20strategies%20for%20hybrid%20renewable%20energy%20system%20with%20hydrogen%20technologies%EF%BC%9A%20State%20of%20the%20art%2C%20trends%20and%20fu%E2%80%A6.pdf). Entrada bibliográfica: [referencias.bib](../referencias.bib). Clave BibTeX: `Khan_2022`.

Etiquetas secundarias: revisión, metaheurísticas híbridas citadas, HOMER, bibliometría VOSviewer. Sección primaria: sistemas híbridos renovables, dimensionamiento y operación.

## Qué aporta

Khan, Yu y Waseem (2022) revisan la optimización del híbrido renovable con electrolizador, pila y tanque (HRES-H₂). No dimensionan un caso propio. Organizan marco matemático, técnicas clásicas, modernas, metaheurísticas híbridas, software y bibliometría.

## Alcance y marco

El objeto es la intermitencia de viento y sol y el papel del hidrógeno para equilibrar generación y demanda. El marco (pp. 1-2 del PDF, secciones de modelado) recorre modelos de panel, aerogenerador, electrolizador, tanque, pila y convertidor, y separa variables de diseño, restricciones y función objetivo. La taxonomía de técnicas distingue programación entera mixta y métodos iterativos; PSO, GA y otras modernas; y enfoques híbridos de metaheurísticas. El software cubierto incluye HOMER, RETScreen y TRNSYS, con límites propios de cada herramienta. La bibliometría usa VOSviewer sobre Web of Science: 954 artículos iniciales, filtrados a 554 tras excluir ciencias puras ajenas al tema (p. 36). Los autores describen como tendencias el interés por optimización tecnoeconómica, infraestructura para bajar el coste del hidrógeno, multiobjetivo con inteligencia artificial y el efecto de renovables e hidrógeno sobre el calentamiento global (p. 1).

En los hallazgos críticos (p. 40) sostienen que no conocen una revisión previa centrada en técnicas de optimización de HRES-H₂ —es una afirmación de los autores, no una comprobación independiente— y que el coste de estos sistemas aún no es suficientemente competitivo. HOMER aparece como la herramienta más usada por la cantidad de combinaciones que evalúa. Advierten que buscar solo en título, resumen y palabras clave de WoS, y solo en inglés, puede omitir artículos.

## Límites

Una revisión no mide un LCOE nuevo ni compara algoritmos sobre una instancia común. Las tablas de técnicas heredan la heterogeneidad de los estudios fuente.

Lectura crítica: el apartado de metaheurísticas híbridas documenta que el campo ya combina optimizadores para HRES-H₂; no autoriza a decir que falte hibridación porque algún resumen no la nombre. Tampoco trata DTW ni control por refuerzo de operadores como objeto propio. Sirve para situar PSO, GA, métodos iterativos y HOMER en el mismo mapa que el hidrógeno de respaldo, y para no confundir un optimizador aplicado a una microrred con un algoritmo híbrido de búsqueda.

## Evidencia

Título, autores, *International Journal of Hydrogen Energy* 47 (2022) 25155-25201 y DOI en la p. 1. Índice de contenidos en las pp. 2-3. Selección WoS en la p. 36. Hallazgos críticos en la p. 40.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
- El nombre de archivo se acortó de forma visible con `…` para caber en la ruta de Windows. El título completo está en el encabezado de esta ficha.
