# Dynamic Programming Algorithm Optimization for Spoken Word Recognition

Ficha de lectura para el estado del arte. PDF: [PDF](./Dynamic%20Programming%20Algorithm%20Optimization%20for%20Spoken%20Word%20Recognition%20%5Bcopia%20P047%5D.pdf). Entrada bibliográfica: [referencias.bib](../../referencias.bib). Clave BibTeX: `Sakoe_1978`.

Etiquetas secundarias: programación dinámica, restricción de pendiente, reconocimiento de voz. Sección primaria: DTW y análisis de búsqueda.

Duplicado exacto de P046. El nombre de archivo es `Sakoe_1978_DPOptimization.pdf`, sin el sufijo «copy» de P046. El SHA-256 es idéntico (`ae0ddd0be313202b1624d1a3365cae9d186993ca42845d54d823add16a36fb77`) y el texto extraído tiene la misma extensión. Esta ficha no resume un segundo artículo: identifica la copia y remite a la misma sustancia.

## Qué aporta el texto, compartido con P046

Sakoe y Chiba, Nippon Electric Company, publican en febrero de 1978, en *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. ASSP-26, n.º 1, pp. 43-49, el algoritmo de DP-matching con el que normalizan el tiempo del habla. Manuscrito recibido el 17 de febrero de 1977 y revisado el 7 de septiembre de 1977 (p. 1). La distancia normalizada es el mínimo residuo tras alinear dos secuencias mediante una función de alabeo. Comparan la forma simétrica y la asimétrica y añaden una restricción de pendiente para no deformar en exceso patrones de categorías distintas.

## Resultados que el texto sostiene

Son los de P046, porque el PDF es el mismo. El resumen dice que el algoritmo optimizado comete no más de unos dos tercios de los errores del mejor algoritmo convencional comparado. La conclusión (p. 7) establece que la forma simétrica es superior a la asimétrica y que el óptimo experimental de la restricción de pendiente es P = 1. La superioridad se contrasta con variantes previas de DP aplicadas a palabras habladas por otros grupos.

## Límites

No hay contribución bibliográfica adicional. Conservar ambos archivos evita perder un insumo del seminario, pero consolidar citas debe usar una sola entrada `Sakoe_1978`.

Lectura crítica: tratar P047 como paper independiente inflaría el estado del arte de DTW. No habla de hidrógeno, de control adaptativo ni de hibridación de metaheurísticas. Su único uso correcto aquí es anclar la definición de alineamiento temporal por programación dinámica y la evidencia, limitada al habla, de que la forma simétrica y la pendiente importan.

## Evidencia

Misma portada de revista en la p. 1 y mismas conclusiones en la p. 7 que P046. Identidad de archivo por el hash del inventario, no por el nombre. DOI `10.1109/tassp.1978.1163055`, verificado por negociación de contenido; no aparece impreso en el extracto.

## Procedencia del archivo

- Copia separada de `P046` (mismo SHA-256). El nombre lleva el sufijo `[copia P047]`. No es un estudio independiente. Clave compartida: `Sakoe_1978`.
