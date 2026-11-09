# Optimizing dynamic time warping’s window width for time series data mining applications

**Clave de cita:** `Dau_2018`  
**PDF:** [PDF](./Optimizing%20dynamic%20time%20warping%E2%80%99s%20window%20width%20for%20time%20series%20data%20mining%20applications%20%5Bcopia%20P019%5D.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Hoang Anh Dau; Diego Furtado Silva; François Petitjean; Germain Forestier; Anthony Bagnall; Abdullah Mueen; Eamonn Keogh |
| Año | 2018 |
| Venue | Data Mining and Knowledge Discovery, 32(4), 1074-1120 |
| DOI | 10.1007/s10618-018-0565-y |
| Sección primaria | `03-dtw-y-analisis-de-busqueda` |
| Etiquetas | duplicado exacto de P018; misma sustancia; no es un segundo estudio |

## Aporte para el estado del arte

Esta entrada conserva el PDF suministrado, pero no aporta un artículo distinto. El SHA-256 de P019 coincide con el de P018 (`fb8261d977b3c9330bf52b1f89367bb975d8976588220bc252005369a4601273`) y el texto extraído tiene la misma extensión. En el estado del arte debe citarse una sola vez, con clave `Dau_2018`.

## Problema y método

El trabajo, idéntico al de P018, trata el ancho de ventana w de dynamic time warping como parámetro crítico de clasificación y clustering de series. El método clásico que discuten es leave-one-out con 1-NN sobre el entrenamiento, barriendo w. Proponen aprender w en supervisado y, para clustering, un esquema semi-supervisado que pide etiquetas después de agrupar, solo en los pares informativos. El banco declarado supera las cien series públicas. No hay planteamiento de control, de metaheurísticas ni de sistemas renovables en el frente del artículo.

## Qué sostiene el texto

Como en P018, el texto sostiene que un w mal elegido impide resultados de calidad, que el mejor w de clasificación no se transfiere al clustering, y que el w adecuado depende de la estructura, de la forma y del tamaño del conjunto. El ejemplo Single Plateau ilustra que conjuntos pequeños piden ventanas mayores y que tanto el exceso como el defecto de warping suben el error. La figura 10 separa Rand index y accuracy. Nada de esto es una replicación: es la misma corrida impresa dos veces.

## Límites

**Hallazgos de los autores.** Los límites son los del artículo único: falta teoría, la extensión a otras distancias queda abierta y el archivo UCR no agota todas las series posibles, incluidas las energéticas.

**Lectura crítica.** Contar P018 y P019 como dos evidencias inflaría el peso de DTW en el seminario. No muestran un hueco de nuestro proyecto. No comparan ventanas sobre trayectorias de una metaheurística híbrida; esa aplicación sería una hipótesis externa.

## Relación con DTW

Misma relación que P018: referencia de análisis de series para alinear perfiles o trazas, con la advertencia de no heredar w de otra tarea. La ficha separada existe solo para no descartar el archivo duplicado.

## Evidencia

- Asignación del lote: mismo SHA-256 y 47 páginas en P018 y P019.
- p. 1 del PDF: título, autores, DOI 10.1007/s10618-018-0565-y y tesis sobre w.
- p. 43: conclusión sobre el carácter crítico de w; coincide con P018.

## Procedencia del archivo

- Copia separada de `P018` (mismo SHA-256). El nombre lleva el sufijo `[copia P019]`. No es un estudio independiente. Clave compartida: `Dau_2018`.
