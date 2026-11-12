# P033 — Aprendizaje al servicio de la metaheurística, no al revés

**Título:** Machine learning at the service of meta-heuristics for solving combinatorial optimization problems: A state-of-the-art

**Cita:** `Karimi_Mamaghan_2022` — Maryam Karimi-Mamaghan, Mehrdad Mohammadi, Patrick Meyer, Amir Mohammad Karimi-Mamaghan y El-Ghazali Talbi (2022). *European Journal of Operational Research*, 296(2), 393–422. DOI: 10.1016/j.ejor.2021.04.032.

**PDF:** [PDF](./Machine%20learning%20at%20the%20service%20of%20meta-heuristics%20for%20solving%20combinatorial%20optimization%20problems%EF%BC%9A%20A%20state-of-the-art.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

## Qué aporta

Revisión técnica, invitada, de cómo el aprendizaje automático entra en el diseño de metaheurísticas para problemas combinatorios. El sentido de la flecha es aprendizaje → metaheurística. No es un algoritmo nuevo ni un caso de energía.

## Problema y método

Los autores constatan que ya hay integraciones distintas y que faltaba una revisión técnica, no solo un listado. Separan este tema de otros tres: metaheurísticas usadas dentro de tareas de aprendizaje, aprendizaje para métodos exactos, y sinergias más antiguas. La taxonomía (Figura 1) organiza la integración por el componente de búsqueda: problema objetivo, inicialización, evolución (selección de operadores, modelo de evolución aprendible, generación de vecinos) y otros elementos de alto y bajo nivel. Cada bloque discute ventajas, requisitos y retos.

## Hallazgos del texto

La contribución no es un porcentaje de mejora. Es el mapa: el aprendizaje puede sembrar soluciones iniciales con conocimiento de instancias parecidas, elegir operadores, orientar la vecindad o extraer patrones de buenas y malas soluciones. Advierten que varios mecanismos deben ajustarse y que corridas repetidas no dan el mismo resultado. Señalan como dirección abierta el uso de programación con conjuntos de respuesta y otras representaciones. El cierre recoge líneas prometedoras, no un ranking de algoritmos.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Es un estado del arte hasta 2021; no cubre los marcos de 2024 de este lote. La revisión no cuantifica un efecto medio único. |
| Lectura crítica | No autoriza a decir que “falta” aplicar aprendizaje a un HRES solo porque el artículo no lo trata: el alcance declarado es combinatorio. No hay DTW ni control de planta. Debe distinguirse de Azevedo, donde el optimizador suele estar al servicio del modelo de aprendizaje. |

## Uso en el estado del arte

Marco para clasificar a Calvet (entradas dinámicas), a Pei (selección de operadores) y a Talbi (taxonomía por componente) sin fundirlos en “híbrido IA”.

Etiquetas secundarias: COP, selección de operadores, inicialización, revisión EJOR.

## Evidencia

- PDF p. 1: objetivo de revisión técnica y métricas de calidad, convergencia y robustez.
- PDF p. 2–3: contraste con revisiones de metaheurísticas dentro del aprendizaje.
- PDF p. 4: Figura 1 y bloques de inicialización y evolución.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
