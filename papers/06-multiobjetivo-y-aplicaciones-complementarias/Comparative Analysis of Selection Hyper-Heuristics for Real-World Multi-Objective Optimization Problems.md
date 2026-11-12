# Comparative Analysis of Selection Hyper-Heuristics for Real-World Multi-Objective Optimization Problems

Ficha de lectura para el estado del arte. PDF: [PDF](./Comparative%20Analysis%20of%20Selection%20Hyper-Heuristics%20for%20Real-World%20Multi-Objective%20Optimization%20Problems.pdf). Entrada bibliográfica: [referencias.bib](../referencias.bib). Clave BibTeX: `deCarvalho_2021`.

Etiquetas secundarias: hiperheurística de selección, MOEA, hipervolumen, IGD+, problemas reales. Sección primaria: multiobjetivo y aplicaciones complementarias.

## Qué aporta

de Carvalho, Özcan y Sichman (2021) comparan cuatro hiperheurísticas de selección en línea sobre dieciocho problemas multiobjetivo de mundo real. Cada hiperheurística elige entre cinco MOEA completas, no entre operadores elementales. Es selección de algoritmos, no un híbrido que fusione sus operadores ni un modelo neuronal del frente de Pareto.

## Problema, método y contribución

Elegir una metaheurística para un problema real exige ensayo y error. Las cuatro hiperheurísticas proceden de trabajos previos: elección, aprendizaje por refuerzo y una función matemática. La conclusión las nombra MOABHH, HHRL, HHLA y HHCF. Controlan cinco MOEA, frente a tres o cuatro heurísticas de bajo nivel en estudios anteriores, y usan instancias reales en lugar de solo funciones de prueba. Miden hipervolumen e IGD+, comparan con Kruskal-Wallis y resumen el cruce de dominios con tres rankings de Friedman.

## Resultados que el texto sostiene

El resumen (p. 1) concluye que las hiperheurísticas tienen mejor desempeño entre dominios que una sola metaheurística, y por eso las presentan como candidatas al resolver un problema multiobjetivo nuevo. La conclusión (p. 17) precisa el alcance de esa frase: la ventaja reclamada es cruzada, sobre dieciocho problemas reales, no la victoria de una hiperheurística en cada instancia. Sostienen, como conocimiento de los autores y no como hecho verificado fuera del paper, que es el primer estudio conjunto de selección de metaheurísticas —no de operadores simples—, foco multiobjetivo e instancias reales. Esta ficha no extrae un ranking numérico de Friedman porque la conclusión no lo repite en una cifra única.

## Límites

Dieciocho problemas reales no incluyen, por lo que el texto de la conclusión enumera como contribución, un protocolo de microrred o de hidrógeno. El aprendizaje es el de cada hiperheurística ya publicada; el paper las compara, no entrena una política nueva de PPO.

Lectura crítica: «mejor entre dominios que una MOEA fija» no autoriza a reemplazar NSGA-II en cualquier sizing energético. Tampoco muestra que falte DTW. Sirve para anclar la diferencia entre elegir una MOEA completa y autoadaptar un parámetro, como hace el paper de mutación polinomial de este mismo lote.

## Evidencia

Título, autores, *Applied Sciences* 2021, 11, 9153, y DOI en la p. 1. Diseño y conclusión de desempeño cruzado en el resumen, p. 1. Nombres de las cuatro hiperheurísticas y alcance en la p. 17. Entrada reutilizada de `papers/referencias.bib`.
