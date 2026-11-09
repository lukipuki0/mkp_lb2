# Predicting population size and termination criteria in metaheuristics: A case study based on spotted hyena optimizer and crow search algorithm

Ficha de lectura para el estado del arte. PDF: [PDF](./Predicting%20population%20size%20and%20termination%20criteria%20in%20metaheuristics%EF%BC%9A%20A%20case%20study%20based%20on%20spotted%20hyena%20optimizer%20and%20crow%20search%20a%E2%80%A6.pdf). Entrada bibliográfica: [referencias.bib](../../referencias.bib). Clave BibTeX: `Vega_2022a`.

Etiquetas secundarias: control en línea de población y parada, SHO, CSA, regresión lineal, MKP. Sección primaria: aprendizaje y control adaptativo.

## Qué aporta

Vega, Soto, Crawford, Peña, Contreras y Castro (2022) proponen LBMB, que predice en la búsqueda el tamaño de población y el criterio de parada. El aprendizaje es regresión lineal múltiple sobre métricas de la corrida, no una red que optimiza el problema ni un híbrido de metaheurísticas.

## Problema, método y contribución

Configurar población y parada consume gran parte del tiempo de implementación, según el antecedente que citan. LBMB separa cuatro módulos (pp. 1-2). Movements aplica la intensificación y diversificación de un algoritmo base intercambiable: spotted hyena optimizer o crow search. Administrator añade o elimina agentes con una ruleta de Monte Carlo sobre esquemas de tamaño. Learning-population ajusta una regresión por esquema con porcentajes de soluciones factibles, mejores e infactibles, y prioriza el mejor pronóstico. Learning-termination estima si seguir iterando aporta mejora. Los umbrales α, β, γ y ε fijan cada cuántas iteraciones cambia el tamaño, se revisa la parada, se reentrenan las regresiones y cuántas iteraciones iniciales quedan libres. No es un memético ni un selector de operadores: es control supervisado, en línea, de dos decisiones de configuración.

## Resultados que el texto sostiene

Evalúan 24 funciones continuas y la mochila multidimensional, contra SHO y CSA clásicos, versiones con IRace y otros algoritmos (pp. 1-2). La conclusión (p. 17) sostiene que estimar las iteraciones en marcha evita cómputo que no mejora la solución. En MKP, Mann-Whitney da superioridad (p < 0,05) solo en instancias listadas, no en todas; por ejemplo LBMB-CSA frente a SHO en 5.100 (00, 03, 04). La tabla 20 (p. 18) muestra tiempos medios de LBMB-CSA de unos 10-18 s frente a más de 100 s de CSA+IRace en instancias pequeñas, sin alcanzar el óptimo conocido.

## Límites

Los autores presentan el trabajo como primer intento y dejan abierta la incorporación de otros métodos de aprendizaje. La regresión asume relaciones lineales entre porcentajes agregados y el desempeño futuro.

Lectura crítica: no demuestra que el mismo controlador sirva para dimensionar microrredes o alinear series con DTW. Tampoco muestra un hueco en hidrógeno. Sirve para separar, en el estado del arte, un controlador de población y parada basado en regresión de un algoritmo híbrido de búsqueda.

## Evidencia

Título, autores, *Applied Soft Computing* 128 (2022) 109513 y DOI en la p. 1. Diseño en las pp. 1-3. Conclusión y tablas de MKP en las pp. 17-18.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
- El nombre de archivo se acortó de forma visible con `…` para caber en la ruta de Windows. El título completo está en el encabezado de esta ficha.
- La clave propuesta `Vega_2022` ya identificaba otro estudio. Esta ficha y su entrada BibTeX usan `Vega_2022a`.
