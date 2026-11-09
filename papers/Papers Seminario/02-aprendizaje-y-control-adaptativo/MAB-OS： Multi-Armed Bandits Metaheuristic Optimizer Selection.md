# MAB-OS: Multi-Armed Bandits Metaheuristic Optimizer Selection

- PDF: [PDF](./MAB-OS%EF%BC%9A%20Multi-Armed%20Bandits%20Metaheuristic%20Optimizer%20Selection.pdf)
- Bibliografía: [referencias.bib](../../referencias.bib)
- Clave de cita: `Meidani_2022`

## Lectura útil

MAB-OS elige en línea, por bandit y UCB, qué regla de actualización aplicar —HHO, DE o WOA— durante tramos de la misma corrida. No entrena una red ni calcula rasgos de paisaje.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Kazem Meidani, Seyedali Mirjalili, Amir Barati Farimani |
| Soporte | *Applied Soft Computing* 128 (2022) 109452; en línea 5 ago. 2022 |
| DOI | 10.1016/j.asoc.2022.109452 |
| Tipo | Selección online de optimizadores; experimento en paisajes continuos |

## Método

Cada algoritmo del portafolio es un brazo. Cada cierto intervalo de K iteraciones se observa la mejora de fitness y se actualiza Q; la acción se muestrea con UCB (Fig. 1 y Algoritmo 1, p. 4). Es explícitamente agnóstico de rasgos: no usa landscape analysis (p. 2). Los autores critican los selectores que fijan un solo algoritmo para toda la corrida, porque entonces no pueden superar al mejor brazo. Con presupuesto finito de evaluaciones, añadir brazos consume el presupuesto; por eso parten de tres algoritmos ya competitivos (p. 4). Código citado: https://github.com/BaratiLab/MAB-OS.

## Qué sostiene el texto

Con HHO, DE y WOA, MAB-OS tiene el mejor desempeño global en tipos distintos de paisaje, en tasa de convergencia y solución final (p. 1). No se reproduce aquí un porcentaje que el resumen no da.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La recompensa por mejora reciente puede engañar; el número de brazos debe equilibrarse con el presupuesto (pp. 2 y 4). |
| Lectura crítica | Tres brazos continuos y sin restricciones no representan un portafolio de hiperheurísticas combinatorias ni un problema energético. UCB sobre optimizadores completos no es lo mismo que UCB sobre vecindarios de VNS. No hay DTW ni HRES. El artículo no muestra que el selector gane en cada función, solo un mejor resultado global. |

## Para el estado del arte

Pieza clara de selección de algoritmos online y feature-agnostic. Contrastar con Malan (rasgos de paisaje para seleccionar) y con TRACE (modelo fijo entrenado con rasgos).

Etiquetas secundarias: bandits; UCB; HHO; DE; WOA; hibridación por portafolio, no por fusión de ecuaciones.

## Evidencia

- p. 1: tres brazos y afirmación de mejor desempeño global.
- p. 2: crítica a la selección única y rechazo de rasgos de paisaje.
- p. 4: Fig. 1, recompensa por mejora y Algoritmo 1.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
