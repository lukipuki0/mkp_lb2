# Towards Autonomous Bio-Inspired Optimization: Deep Reinforcement Learning for Adaptive Metaheuristic Orchestration

Ficha de lectura para el estado del arte. PDF: [PDF](./Towards%20Autonomous%20Bio-Inspired%20Optimization%EF%BC%9A%20Deep%20Reinforcement%20Learning%20for%20Adaptive%20Metaheuristic%20Orchestration.pdf). Entrada bibliográfica: [referencias.bib](../referencias.bib). Clave BibTeX: `Carrasco_2026`.

Etiquetas secundarias: hiperheurística, PPO, mochila multidimensional, set covering, transferencia de población. Sección primaria: aprendizaje y control adaptativo.

## Qué aporta

Carrasco, Crawford, Valle, Zulantay, Giachetti y Soto (2026) formulan la selección en línea entre metaheurísticas poblacionales que comparten población y, en variantes más ricas, configuración. El agente es PPO. No sustituye a la metaheurística ni recombina operadores de dos algoritmos en cada individuo.

## Problema, método y contribución

Frente a hiperheurísticas que eligen operadores o ajustan una sola metaheurística, aquí la decisión recae en siete metaheurísticas poblacionales, sobre mochila multidimensional y set covering. Cada decisión transfiere la población completa. El estado tiene catorce dimensiones, independiente del tamaño, y la acción ocurre cada cuanto de iteraciones. Tres espacios de acción comparten estado, recompensa y portafolio: selección discreta; selección y configuración mediante mapas a parámetros nativos; y coevolución por una asignación en el símplex. En cb9 (n = 500, 25 000 evaluaciones, Q = 25) usan cinco pliegues y veinte semillas. La selección también se evalúa en 65 instancias de set covering.

## Resultados que el texto sostiene

En cb9 el agente de selección queda por delante de las líneas ciegas round-robin y aleatoria. El de coevolución queda segundo, a 0,043 puntos porcentuales del mejor valor conocido, por detrás de la mejor metaheurística individual. A mitad de presupuesto esos dos agentes ocupan el primer y el segundo puesto. En set covering, donde ningún método individual domina, la selección queda primera entre once estrategias. Un control sin entrenar y una ablación del portafolio indican que la política aprende sobre todo a qué miembros evitar. El agente de selección y configuración conjunta todavía no mejora las líneas ciegas. La conclusión (pp. 24-25) reporta un GAP medio de 0,362 % para el agente de coevolución en cb9, en el régimen de presupuesto en que la señal de mejora sigue siendo informativa.

## Límites

Dos dominios binarios y un portafolio concreto de siete métodos. El mapa de traducción no basta, en este estudio, para que configurar en línea supere a no aprender.

Lectura crítica: orquestar metaheurísticas con PPO es control adaptativo de alto nivel, distinto de una metaheurística híbrida y distinto de usar una red como modelo sustituto del LCOE. No hay resultados de microrredes. No debe leerse como prueba de que el mismo agente mejoraría un HRES, ni como hueco energético.

## Evidencia

Título, autores, *Biomimetics* 2026, 11, 682, y DOI en la p. 1. Cifras del resumen en la p. 1. Conclusiones en las pp. 24-25. Entrada reutilizada de `papers/referencias.bib`.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
