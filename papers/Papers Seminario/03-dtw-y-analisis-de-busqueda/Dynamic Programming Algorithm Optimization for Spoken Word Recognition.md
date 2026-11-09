# Dynamic Programming Algorithm Optimization for Spoken Word Recognition

Ficha de lectura para el estado del arte. PDF: [PDF](./Dynamic%20Programming%20Algorithm%20Optimization%20for%20Spoken%20Word%20Recognition.pdf). Entrada bibliográfica: [referencias.bib](../../referencias.bib). Clave BibTeX: `Sakoe_1978`.

Etiquetas secundarias: programación dinámica, restricción de pendiente, reconocimiento de voz. Sección primaria: DTW y análisis de búsqueda.

Duplicado exacto de P047 (`Sakoe_1978_DPOptimization.pdf`, mismo SHA-256). No se descarta ninguno de los dos archivos.

## Qué aporta

Sakoe y Chiba (1978) optimizan el DP-matching para normalizar el eje temporal del habla. Es el antecedente clásico del dynamic time warping: una función de alabeo alinea dos secuencias y la distancia normalizada es el residuo mínimo, calculado por programación dinámica.

## Problema, método y contribución

El ritmo de habla deforma el eje temporal de forma no lineal y la normalización lineal no basta. Comparan la distancia normalizada simétrica y la asimétrica. En la asimétrica un eje se transforma sobre el otro; en la simétrica ambos se proyectan a un eje común provisional. La restricción de pendiente evita que un alabeo demasiado flexible empeore la discriminación entre categorías. Parten de muestreo uniforme común y de ausencia de conocimiento a priori sobre qué tramos importan, de modo que cada tramo pesa igual (p. 2). No es una metaheurística ni un controlador de exploración y explotación: es un camino mínimo por programación dinámica, con restricciones de alabeo y ventana de ajuste.

## Resultados que el texto sostiene

El resumen (p. 1) afirma que el algoritmo optimizado no produce más de unos dos tercios de los errores del mejor algoritmo convencional comparado, en experimentos de reconocimiento de palabras. La conclusión (p. 7) fija dos resultados: la forma simétrica reconoce mejor que la asimétrica, y la restricción de pendiente es efectiva, con mejor desempeño cuando la condición de pendiente es P = 1. Los autores subrayan el acuerdo entre la discusión teórica y los experimentos, y la comparación con variantes de DP usadas por otros grupos [3]-[6].

## Límites

El dominio es habla, con patrones muestreados a periodo común. No hay prueba en series de potencia, hidrógeno o trayectorias de un optimizador. La cifra de «dos tercios de los errores» es el resumen del experimento de reconocimiento, no una cota general para cualquier distancia temporal.

Lectura crítica: ancla la definición de alineamiento temporal y la evidencia, limitada al habla, de que simetría y pendiente cambian la discriminación. No demuestra que DTW mejore un dimensionamiento renovable ni que falte en papers de microrredes. P047 no añade resultados.

## Evidencia

Cabecera IEEE TASSP, vol. ASSP-26, n.º 1, febrero de 1978, p. 43, en la p. 1 del PDF. Principio de DP-matching en las pp. 1-2. Conclusiones en la p. 7. DOI verificado por negociación Crossref, no impreso en el extracto.

## Procedencia del archivo

- Estudio canónico de las copias suministradas (`P047`). Misma clave `Sakoe_1978`. No contar las copias como estudios independientes.
