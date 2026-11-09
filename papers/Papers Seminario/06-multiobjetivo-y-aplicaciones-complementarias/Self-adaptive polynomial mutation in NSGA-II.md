# Self-adaptive polynomial mutation in NSGA-II

Ficha de lectura para el estado del arte. PDF: [PDF](./Self-adaptive%20polynomial%20mutation%20in%20NSGA-II.pdf). Entrada bibliográfica: [referencias.bib](../../referencias.bib). Clave BibTeX: `CarlesBou_2023`.

Etiquetas secundarias: control de parámetro en línea, índice de distribución, hipervolumen e IGD. Sección primaria: multiobjetivo y aplicaciones complementarias.

## Qué aporta

Carles-Bou y Galán (2023) autoadaptan el índice de distribución de la mutación polinomial dentro de NSGA-II. Es control de un parámetro del operador, computacionalmente ligero, no una red que selecciona metaheurísticas ni un híbrido de algoritmos de optimización.

## Problema, método y contribución

NSGA-II sigue usándose como línea base multiobjetivo, casi siempre con SBX y mutación polinomial de probabilidades e índices fijos. Los autores recuerdan que Deb et al. (2007) autoadaptaron el cruce SBX y que Hamdan (2012) mostró que un índice fijo de mutación polinomial no es siempre el mejor. El mecanismo nuevo ajusta en línea el índice ηm de la mutación polinomial. El resto de NSGA-II —orden no dominado y distancia de crowding— se mantiene. La comparación es contra el mismo NSGA-II con índice estático afinado, en veinticinco problemas de DTLZ, WFG, ZDT y otros bancos, con tres indicadores: hipervolumen, dispersión generalizada y distancia generacional invertida modificada.

## Resultados que el texto sostiene

El resumen (p. 1) dice que la versión autoadaptativa supera a la estática en más del 75 % de los problemas en esos tres indicadores. El apartado de conclusiones (p. 13) detalla que hipervolumen e IGD+ mejoran en 19 problemas y empatan en 4 o 5; la única divergencia señalada entre ambos indicadores es WFG5, donde el hipervolumen ve peor a la mutación autoadaptativa. La dispersión generalizada mejora en 13 ocasiones, un resultado más débil que los otros dos indicadores, y los autores lo presentan como alentador pero no equivalente. Un signo «=» en sus tablas significa empate y «-» significa que su algoritmo empeora ese indicador.

## Límites

El banco es de problemas de prueba, no de despacho ni de dimensionamiento. No comparan contra otras MOEA modernas como línea base principal; el control es frente al NSGA-II estático.

Lectura crítica: autoadaptar ηm no es Q-learning ni PPO. Conviene no fundir este paper con las hiperheurísticas que eligen entre cinco MOEA. Sirve para mostrar que un parámetro clásico del mutador multiobjetivo puede moverse en línea con coste bajo y efecto medible en hipervolumen e IGD+. No valida un proyecto de hidrógeno y no abre un hueco de DTW por no mencionarlo.

## Evidencia

Título, autores, *Soft Computing* (2023) 27:17711-17727 y DOI en la p. 1. Umbral del 75 % en el resumen, p. 1. Desglose por indicador en la p. 13. Entrada reutilizada de `papers/referencias.bib`.
