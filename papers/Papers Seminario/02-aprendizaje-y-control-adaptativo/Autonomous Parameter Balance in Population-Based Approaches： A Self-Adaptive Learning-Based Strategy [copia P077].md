# Autonomous Parameter Balance in Population-Based Approaches: A Self-Adaptive Learning-Based Strategy

- PDF: [PDF](./Autonomous%20Parameter%20Balance%20in%20Population-Based%20Approaches%EF%BC%9A%20A%20Self-Adaptive%20Learning-Based%20Strategy%20%5Bcopia%20P077%5D.pdf)
- Bibliografía: [referencias.bib](../../referencias.bib)
- Clave de cita: `Vega_2024`

## Lectura útil

LBLP ajusta en línea el tamaño de la población con un componente de aprendizaje y un selector probabilístico. El duplicado binario de este PDF es P010.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Emanuel Vega, José Lemus-Romani, Ricardo Soto, Broderick Crawford, Christoffer Löffler, Javier Peña, El-Gazhali Talbi |
| Soporte | *Biomimetics* 2024, 9, 82; publicado 31 ene. 2024 |
| DOI | 10.3390/biomimetics9020082 |
| Tipo | Control autoadaptativo de población; tres problemas discretos |

El archivo `papers/pdf/Vega_2024.pdf` tiene el mismo SHA-256 que P010 (`9cb64aeed48b698753aedf0110a8ba38adf378af13739c1039d22becd7006ede`). Misma sustancia; no se descarta ningún PDF. En la portada el apellido aparece como El-Gazhali; el correo es el-ghazali.talbi@univ-lille.fr (p. 1). La clave BibTeX repite El-Gazhali.

## Método

Tres componentes (p. 1): uno de optimización, que instancia operadores de algoritmos poblacionales; uno de aprendizaje, que convierte datos dinámicos de la corrida en conocimiento; uno selector probabilístico, que ajusta la población. El parámetro controlado es el tamaño de población, no F/CR ni la heurística de bajo nivel. Problemas: Manufacturing Cell Design, set covering y mochila multidimensional, con un único conjunto de configuración (p. 3). Rivales: metaheurísticas clásicas, autónomas y afinadas con IRace. En SCP usan Wilcoxon–Mann–Whitney y toman p < 0,05 como diferencia significativa (p. 21).

## Qué sostiene el texto

Compite con esas líneas base y muestra que se puede cambiar cuántas soluciones interactúan a lo largo de la búsqueda (p. 1). La regresión lineal usada en el componente de aprendizaje tiene un solo métrico de desempeño como variable independiente; los autores lo marcan como límite (p. 30).

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La regresión con una sola métrica no captura otras señales de la búsqueda; el diseño queda atado a los algoritmos instanciados (p. 30). |
| Lectura crítica | Tres problemas discretos de diseño/cobertura/mochila no son HRES, hidrógeno ni alineación temporal. «Self-adaptive» aquí no es la autoadaptación genotípica de Lehre (2023). No concluir que LBLP cubre el control de operadores del resto del lote. |

## Para el estado del arte

Extensión del programa de Vega 2021: de predecir el movimiento en SHO continuo a redimensionar la población en discreto. Útil en control adaptativo de parámetros estructurales, que Zhang (2023) también distingue de los parámetros numéricos.

Etiquetas secundarias: tamaño de población; MCDP; SCP; MKP; IRace; duplicado exacto de P010.

## Evidencia

- p. 1: tres componentes, problemas y grafía El-Gazhali.
- p. 3: contribuciones y configuración única.
- p. 21: umbral 0,05 en SCP.
- p. 30: límite de la regresión lineal.

## Procedencia del archivo

- Copia separada de `P010` (mismo SHA-256). El nombre lleva el sufijo `[copia P077]`. No es un estudio independiente. Clave compartida: `Vega_2024`.
- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
