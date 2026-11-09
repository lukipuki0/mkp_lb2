# P039 — No hay almuerzo gratis al comparar optimizadores

**Título:** No Free Lunch Theorems for Optimization

**Cita:** `Wolpert_1997` — David H. Wolpert y William G. Macready (1997). *IEEE Transactions on Evolutionary Computation*, 1(1), 67–82. DOI: 10.1109/4235.585893.

**PDF:** [PDF](./No%20Free%20Lunch%20Theorems%20for%20Optimization.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

## Qué aporta

Teoremas de “no free lunch”: si un algoritmo rinde mejor en una clase de problemas, esa ventaja se compensa en otra clase, cuando se promedia sobre todas las funciones del marco. Es fundamento de la comparación de optimizadores, no un método nuevo ni un caso aplicado.

## Problema y método

Pregunta qué se puede afirmar antes de conocer la función de costo. Cuenta evaluaciones ante un oráculo, barre problemas o algoritmos, y trata funciones que cambian en el tiempo y distinciones minimax que sobreviven a la uniformidad.

## Hallazgos del texto

El resultado central es geométrico: estar bien adaptado a un problema es una relación entre algoritmo y clase, no una propiedad absoluta del algoritmo. La uniformidad no impide que, mirando una función cada vez, dos algoritmos se distingan en sentido minimax. Otra consecuencia práctica: un algoritmo que no recuerda puntos ya visitados desperdicia evaluaciones; recordarlo (búsqueda tabú) lo hace más eficiente sin violar el teorema, porque cambia la clase de procedimientos. El texto explica así que algoritmos ciegos funcionen bien en la práctica sin contradecir el NFL: la práctica no muestrea todas las funciones posibles.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El marco es de optimización de caja negra sobre un conjunto finito de funciones; no modela el costo de cada evaluación ni restricciones de un HRES. |
| Lectura crítica | El DOI no está impreso en este PDF de 1997; se acepta porque Crossref devuelve el mismo título, autores, volumen 1, número 1 y páginas 67–82. El teorema no dice que hibridar sea inútil: dice que la ganancia hay que justificarla en la clase de problemas (paisaje del dimensionamiento, no en “el algoritmo es mejor”). No hay DTW ni hidrógeno. |

## Uso en el estado del arte

Freno para leer los bancos CEC de Lale o los frentes de Tezer como superioridad universal. Obliga a declarar la clase de problemas del seminario antes de elegir híbrido.

Etiquetas secundarias: NFL, evaluación de oráculo, búsqueda tabú, minimax.

## Evidencia

- PDF p. 1: enunciado de los teoremas y lectura geométrica.
- PDF p. 2: funciones dependientes del tiempo, distinciones minimax y algoritmos que revisitan puntos.
- PDF p. 16: cierre formal; la paginación de revista llega a 82.
