# P034 — Taxonomía de Talbi: dónde entra el aprendizaje en la búsqueda

**Título:** Machine Learning into Metaheuristics: A Survey and Taxonomy

**Cita:** `Talbi_2021` — El-Ghazali Talbi (2021). *ACM Computing Surveys*, 54(6), artículo 129, 32 pp. DOI: 10.1145/3459664.

**PDF:** [PDF](./Machine%20Learning%20into%20Metaheuristics%EF%BC%9A%20A%20Survey%20and%20Taxonomy.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

## Qué aporta

Taxonomía unificada de sinergias entre aprendizaje y metaheurísticas, ordenada por el componente de búsqueda: problema objetivo, componentes de bajo nivel y componentes de alto nivel. Es una encuesta, no un experimento.

## Problema y método

Las metaheurísticas generan datos —instancias, trayectorias, poblaciones, movimientos, élites— y casi no los explotan con modelos explícitos. Talbi recorre oportunidades para usar esos datos y fija un vocabulario común. Distingue aprendizaje offline (programación genética y expresión génica para generar heurísticos; también apprenticeship learning, sistemas clasificadores, redes y regresión logística) y online (sobre todo aprendizaje por refuerzo para construir heurísticos voraces o iterativos). La Figura 11 esquematiza la selección online de una metaheurística con refuerzo.

## Hallazgos del texto

No hay una tabla de victorias propia. La tesis es que varias metaheurísticas apoyadas en aprendizaje ya son estado del arte en sus nichos, siempre que el costo de aprender se cuente en la evaluación. El cierre insiste en ese sobrecosto y en que el diseño de metaheurísticas con aprendizaje tendrá impacto, sin cuantificarlo. Un ejemplo citado de refuerzo profundo con redes de grafos formula la búsqueda local de SAT como proceso de decisión de Markov: la red aprende una política de movimientos, no los pesos de un pronóstico energético.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Deja cuestiones abiertas que “requieren investigación más profunda” y recuerda que el overhead puede anular la ganancia. |
| Lectura crítica | Una política de búsqueda local aprendida no es lo mismo que un híbrido GWO–WOA ni que un optimizador que ajusta una red de pronóstico. El artículo no trata HRES, hidrógeno ni DTW; su silencio no es una laguna demostrada. |

## Uso en el estado del arte

Vocabulario de referencia para el seminario: bajo nivel (operadores, parámetros) frente a alto nivel (qué metaheurística lanzar). Encaja con Pei y con la revisión EJOR de 2022, que el propio Talbi cofirma.

Etiquetas secundarias: refuerzo, programación genética, sobrecosto de aprender.

## Evidencia

- PDF p. 1: taxonomía por componente y problemas abiertos.
- PDF p. 21: aprendizaje offline/online, Figura 11 y conclusiones sobre el overhead.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
