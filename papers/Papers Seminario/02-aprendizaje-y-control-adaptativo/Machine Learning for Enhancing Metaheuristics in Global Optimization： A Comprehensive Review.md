# Machine Learning for Enhancing Metaheuristics in Global Optimization: A Comprehensive Review

Ficha de lectura para el estado del arte. PDF: [PDF](./Machine%20Learning%20for%20Enhancing%20Metaheuristics%20in%20Global%20Optimization%EF%BC%9A%20A%20Comprehensive%20Review.pdf). Entrada bibliográfica: [referencias.bib](../../referencias.bib). Clave BibTeX: `Bolufe_2025`.

Etiquetas secundarias: taxonomía funcional, sustitutos, paisaje, optimizadores aprendidos, refuerzo. Sección primaria: aprendizaje y control adaptativo.

## Qué aporta

Bolufé-Röhler y Tamayo-Vera (2025) revisan cómo el aprendizaje automático mejora metaheurísticas de optimización global. No informan un experimento nuevo. Su contribución es una taxonomía funcional que separa papeles del aprendizaje dentro de la búsqueda, precisamente para no mezclar un modelo que elige operadores con un híbrido de algoritmos de optimización ni con el uso inverso de metaheurísticas para entrenar redes.

## Alcance y marco

El objeto son problemas de caja negra, no convexos, a menudo sin gradiente fiable. Las metaheurísticas citadas como base —GA, PSO, DE, estrategias de evolución, recocido simulado, CMA-ES— sufren convergencia prematura, sensibilidad paramétrica y poca eficiencia muestral. El aprendizaje supervisado, no supervisado, por refuerzo y meta-aprendizaje se presenta como vía de adaptación. La taxonomía, anunciada en el resumen (p. 1) y retomada en la conclusión (p. 31), clasifica la integración según su función: control de operadores, modelos sustitutos, aprendizaje del paisaje y optimizadores aprendidos, además de configuración de algoritmos y aprendizaje de representaciones. No es un catálogo de nombres de algoritmos bioinspirados.

La conclusión marca el límite de alcance con una frase útil: la dirección complementaria —usar metaheurísticas para ajustar hiperparámetros, seleccionar rasgos o particionar datos de un modelo de aprendizaje— queda fuera y se remite a otras revisiones [115-117]. El paper se concentra en aprendizaje que mejora la búsqueda, no en búsqueda que entrena el modelo.

## Límites

Como revisión, no mide un porcentaje de mejora propio ni fija un banco de prueba único. Las tendencias que señala dependen del corpus que los autores consideraron representativo; esta ficha no recontó ese corpus.

Lectura crítica: la taxonomía sirve para clasificar otros papers de este lote sin forzarlos a la misma casilla. La regresión que predice población y parada, el Q-learning que elige binarización y el PPO que orquesta metaheurísticas completas son controles distintos, no «metaheurísticas híbridas» en el sentido de recombinar GA y PSO. El silencio sobre hidrógeno o DTW no es un hueco del campo energético. Sirve como mapa de lectura, no como evidencia de que un método concreto gane en microrredes.

## Evidencia

Título, autores, *Mathematics* 2025, 13, 2909, y DOI en la p. 1. Declaración de alcance en el resumen, p. 1. Conclusión y exclusión de la dirección inversa en la p. 31. Entrada reutilizada de `papers/referencias.bib`.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
