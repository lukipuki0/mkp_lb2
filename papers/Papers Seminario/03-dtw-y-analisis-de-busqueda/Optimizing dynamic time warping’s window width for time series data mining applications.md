# Optimizing dynamic time warping’s window width for time series data mining applications

**Clave de cita:** `Dau_2018`  
**PDF:** [PDF](./Optimizing%20dynamic%20time%20warping%E2%80%99s%20window%20width%20for%20time%20series%20data%20mining%20applications.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Hoang Anh Dau; Diego Furtado Silva; François Petitjean; Germain Forestier; Anthony Bagnall; Abdullah Mueen; Eamonn Keogh |
| Año | 2018 |
| Venue | Data Mining and Knowledge Discovery, 32(4), 1074-1120 |
| DOI | 10.1007/s10618-018-0565-y |
| Sección primaria | `03-dtw-y-analisis-de-busqueda` |
| Etiquetas | ventana de warping; clasificación; clustering; UCR; duplicado exacto de P019 |

## Aporte para el estado del arte

Demuestra que el único parámetro de DTW, el ancho máximo de warping w, decide la calidad de clasificación y de clustering, y que el mejor w no se transfiere entre tareas ni entre tamaños del mismo conjunto. Es la ficha de DTW del lote. P019 es el mismo PDF, mismo SHA-256; no suma evidencia.

## Problema y método

DTW es la distancia de referencia en minería de series, pero w suele fijarse por validación cruzada si hay etiquetas abundantes, o al máximo computacionalmente admisible si no las hay. Los autores muestran que ambas costumbres fallan: con pocos datos de entrenamiento la validación cruzada es subóptima; en clustering no hay etiquetas para aprender w; el w de clasificación no sirve para clustering; y el w depende también del tamaño y de la forma de las series. Proponen métodos para aprender w en supervisado y no supervisado. En clustering, el método es semi-supervisado a posteriori: se pide etiqueta solo de pares que el esquema de puntuación considera informativos, no un etiquetado inicial completo. El banco supera las cien series públicas, con el archivo UCR como referencia. El aprendizaje clásico que critican barre w de 0 a 100 % con 1-NN y leave-one-out.

## Qué sostiene el texto

En el ejemplo Single Plateau, el mejor w cambia con el tamaño del conjunto y, para 32 objetos, el error de leave-one-out empeora tanto si w es demasiado grande como si es demasiado pequeño. La figura 10 muestra que el Rand index de clustering y la accuracy de clasificación no se mueven juntos al variar w. La conclusión afirma que, si w está mal puesto, el resto del pipeline no recupera calidad, y que ajustar w puede cerrar la mayor parte o toda la ventaja de métodos más elaborados publicados como mejoras sobre DTW. Es una tesis experimental de los autores sobre sus conjuntos, no una cota para cualquier serie energética.

## Límites

**Hallazgos de los autores.** Dejan abierto el tratamiento teórico y la extensión a otras distancias con parámetros. Publican código y advierten que buscar w grande en el archivo UCR probablemente no rinda.

**Lectura crítica.** No optimizan una planta ni un metaheurístico. No comparan DTW con regresión de LB2. Usarla para alinear trayectorias de búsqueda es una hipótesis nuestra: el paper no hace ese puente. P019 no es una replicación.

## Relación con análisis de búsqueda

Si una traza de metaheurística se trata como serie, este paper dice que la ventana de alineación es un parámetro de tarea y de tamaño, no una constante heredada de clasificación. Esa lectura es nuestra, no un experimento de los autores.

## Evidencia

- p. 1: tesis sobre w, no transferibilidad y más de cien conjuntos.
- pp. 10-12: efecto del tamaño y desacople entre accuracy y Rand index.
- p. 43: conclusión sobre w crítico y aprendizaje semi-supervisado para clustering.

## Procedencia del archivo

- Estudio canónico de las copias suministradas (`P019`). Misma clave `Dau_2018`. No contar las copias como estudios independientes.
