# Hyper-heuristics: A survey and taxonomy

- PDF: [PDF](./Hyper-heuristics%EF%BC%9A%20A%20survey%20and%20taxonomy.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Dokeroglu_2024`

## Lectura útil

Es una revisión taxonómica, no un experimento: reordena dos décadas de hiperheurísticas en selección, heurísticas de bajo nivel, problemas objetivo e hiperheurísticas paralelas.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Tansel Dokeroglu, Tayfun Kucukyilmaz, El-Ghazali Talbi |
| Soporte | *Computers & Industrial Engineering* 187 (2024) 109815; recibido 18 nov. 2022, aceptado 2 dic. 2023, en línea 4 dic. 2023 |
| DOI | 10.1016/j.cie.2023.109815 |
| Tipo | Survey; no hay algoritmo nuevo ni banco de pruebas propio |

## Alcance del marco

Los autores separan hiperheurística de metaheurística: la primera selecciona, genera o secuencia heurísticas; no recorre ella misma el espacio de soluciones (p. 1, Fig. 1). La taxonomía propuesta (Fig. 2, p. 2) tiene cuatro ejes: selection hyper-heuristics, incluidas técnicas de aprendizaje automático; low-level heuristics; problemas objetivo; e hiperheurísticas paralelas. Declaran haber inspeccionado más de 150 trabajos recientes y sitúan el corte de búsqueda bibliográfica en mayo de 2023: más de 9660 resultados en Google Scholar y 1003 en Web of Science para «hyper-heuristic» (p. 1). Esos conteos son de motores de búsqueda, no de una revisión sistemática con criterios de inclusión reproducibles en el extracto.

La sección 2 recorre Burke, Özcan, HyFlex y jMetal, y usa No Free Lunch para justificar elevar el nivel de la búsqueda (p. 2). La sección 4 cierra con cuestiones abiertas, no con una tabla de resultados.

## Qué sostiene el texto y qué no

Sostienen un crecimiento de la literatura, un giro hacia multiobjetivo, aprendizaje automático y cómputo paralelo, y la ausencia de una teoría ampliamente aceptada (pp. 1 y 12). No sostienen que una familia supere a otra con un efecto cuantitativo propio: no hay experimento de los autores.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Falta base teórica sólida; el costo de evaluar la función objetivo sigue siendo crítico; proponen programación dinámica y mejor configuración de parámetros (p. 12). |
| Lectura crítica | Los conteos de Scholar/WoS no son una revisión sistemática. Cubrir «machine learning» dentro de selección no equivale a una taxonomía fina de Q-learning, PPO o bandits. No hay DTW, HRES ni hidrógeno. No usar el silencio del survey como hueco demostrado de un proyecto concreto. |

## Para el estado del arte

Marco para distinguir selección de generación y no llamar «híbrido» a todo selector. Útil antes de citar control por RL.

Etiquetas secundarias: taxonomía; hiperheurísticas paralelas; multiobjetivo como tendencia citada, no como resultado.

## Evidencia

- p. 1: definición, conteos de mayo de 2023 y cuatro categorías.
- p. 2: Fig. 2, NFL y encuestas previas.
- p. 12: conclusiones y problemas abiertos.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
