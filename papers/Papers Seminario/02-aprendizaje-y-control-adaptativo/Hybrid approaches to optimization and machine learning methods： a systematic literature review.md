# P026 — Revisión de híbridos optimización–aprendizaje para agrupar y clasificar

**Título:** Hybrid approaches to optimization and machine learning methods: a systematic literature review

**Cita:** `Azevedo_2024` — Beatriz Flamia Azevedo, Ana Maria A. C. Rocha y Ana I. Pereira (2024). *Machine Learning*, 113(7), 4055–4097. DOI: 10.1007/s10994-023-06467-x.

**PDF:** [PDF](./Hybrid%20approaches%20to%20optimization%20and%20machine%20learning%20methods%EF%BC%9A%20a%20systematic%20literature%20review.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

## Qué aporta

Revisión sistemática y bibliométrica de métodos híbridos que combinan optimización y aprendizaje automático para agrupamiento y clasificación. No propone un algoritmo nuevo ni un caso energético.

## Problema y método

Ningún método puro cubre datos grandes y no convexos. El recuento arranca en 1970 y la lectura densa cubre los tres años previos. Hay preguntas de investigación y un SWOT de los diez algoritmos más citados. El objeto es clustering y clasificación, no el diseño de metaheurísticas combinatorias.

## Hallazgos del texto

Las dificultades que la revisión encuentra una y otra vez son la convergencia prematura y la estimación de parámetros. PSO aparece como el bioinspirado más usado por simplicidad; k-means, SVM y k-NN concentran el ajuste de parámetros. Muchos “híbridos” usan un enjambre o un genético para fijar centros, pesos o hiperparámetros de un modelo de aprendizaje, o para minimizar el error de clasificación. Eso es un modelo de aprendizaje más un optimizador, no una metaheurística híbrida de búsqueda. La revisión sostiene que, en los trabajos que recoge, el híbrido suele mejorar velocidad, exploración o exactitud sin exigir más cómputo que el método puro, pero esa afirmación resume la literatura, no un experimento de las autoras.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Declaran que no es viable listar la debilidad de cada algoritmo. El SWOT cubre solo los diez más citados. El futuro que señalan es integrar aprendizaje profundo y conocimiento experto. |
| Lectura crítica | No debe citarse como prueba de que hibridar GWO con WOA mejora un HRES. Tampoco trata DTW, control de electrolizadores ni dimensionamiento. La frase “la mayoría de las veces superan al método puro” hereda el sesgo de publicación de los papers revisados. |

## Uso en el estado del arte

Sirve para no confundir tres familias: optimizador que entrena un modelo, metaheurística que usa aprendizaje para elegir operadores, y híbrido de dos buscadores. El lote tiene ejemplos de las tres.

Etiquetas secundarias: revisión sistemática, clustering, SVM, PSO como ajustador.

## Evidencia

- PDF p. 1: alcance clustering/clasificación, recuento desde 1970 y SWOT.
- PDF p. 33–34: respuestas a RQ1–RQ3 sobre convergencia prematura y estimación de parámetros.
- PDF p. 34–35: conclusiones y trabajo futuro.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
