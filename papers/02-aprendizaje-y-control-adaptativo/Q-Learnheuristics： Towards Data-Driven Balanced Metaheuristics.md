# Q-Learnheuristics: Towards Data-Driven Balanced Metaheuristics

Ficha de lectura para el estado del arte. PDF: [PDF](./Q-Learnheuristics%EF%BC%9A%20Towards%20Data-Driven%20Balanced%20Metaheuristics.pdf). Entrada bibliográfica: [referencias.bib](../referencias.bib). Clave BibTeX: `Crawford_2021`.

Etiquetas secundarias: Q-learning, binarización, WOA, SCA, set covering. Sección primaria: aprendizaje y control adaptativo.

## Qué aporta

Crawford y coautores (2021) usan Q-learning para elegir operadores de exploración y explotación, en particular esquemas de binarización cuando una metaheurística continua resuelve un problema binario. Lo aplican a whale optimization y sine-cosine en set covering. El aprendiz es una tabla Q, no una red profunda, y no sustituye a la búsqueda.

## Problema, método y contribución

El equilibrio exploración-explotación sigue abierto. Exploración visita regiones nuevas; explotación visita vecindades de soluciones ya vistas. El marco QL selecciona el operador conducente a ese equilibrio. La revisión interna que la conclusión resume distingue dos usos ya vistos de Q-learning en metaheurísticas: selector de heurísticas de bajo nivel, al estilo hiperheurística, y selector de un operador dentro de un conjunto, por ejemplo una mutación entre varias. Este trabajo está en el segundo uso, aplicado a la binarización. La tabla Q permanece pequeña porque el conjunto de operadores es pequeño, a diferencia de entradas de alta dimensión. Los algoritmos base son WOA y SCA; el problema es de cobertura de conjuntos. No hibridan WOA con SCA en un solo movimiento: cada uno se beneficia de la selección aprendida de esquemas de binarización.

## Resultados que el texto sostiene

El resumen (p. 1) afirma mejoras estadísticas tanto en el equilibrio exploración-explotación como en la calidad de las soluciones al resolver el set covering con WOA y SCA. La conclusión (p. 22) no añade un porcentaje único de mejora; insiste en que las metaheurísticas generan datos que normalmente no reutilizan y en que QL cabe en tablas pequeñas. Mencionan como línea futura combinar QL con otras diferencias temporales, por ejemplo SARSA. Esta ficha no cita medias de fitness de tablas interiores porque la extracción no deja una cifra aislada inequívoca en la conclusión.

## Límites

Un problema combinatorio y dos algoritmos continuos binarizados. No hay transferencia mostrada a espacios continuos de dimensionamiento ni a objetivos de coste y emisiones.

Lectura crítica: no es «red neuronal más optimizador» ni una metaheurística híbrida. Es control tabular del operador de binarización. En el mapa del lote queda junto a la regresión de población y parada y al PPO de portafolio, con una capacidad de estado mucho menor. No informa sobre hidrógeno ni DTW, y esa ausencia no es un hueco del estado del arte energético.

## Evidencia

Título, autores, *Mathematics* 2021, 9, 1839, y DOI en la p. 1. Afirmación estadística en el resumen, p. 1. Conclusión en la p. 22. Entrada reutilizada de `papers/referencias.bib`.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
