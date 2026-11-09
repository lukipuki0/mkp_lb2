# A comprehensive review on optimization of hybrid renewable energy systems using various optimization techniques

**Clave de cita:** `Thirunavukkarasu_2023`  
**PDF:** [PDF](./A%20comprehensive%20review%20on%20optimization%20of%20hybrid%20renewable%20energy%20systems%20using%20various%20optimization%20techniques.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | M. Thirunavukkarasu; Yashwant Sawle; Himadri Lala |
| Año | 2023 |
| Venue | Renewable and Sustainable Energy Reviews, 176, 113192 |
| DOI | 10.1016/j.rser.2023.113192 |
| Sección primaria | `04-hres-dimensionamiento-y-operacion` |
| Etiquetas | revisión; metaheurísticas híbridas; HOMER; hidrógeno como dirección abierta |

## Aporte para el estado del arte

Sirve como mapa de cómo se dimensionan y operan HRES, no como experimento nuevo. Ordena métodos clásicos, técnicas de inteligencia artificial de un solo algoritmo, híbridos de dos o más algoritmos y herramientas comerciales. Sirve para situar un optimizador híbrido frente al sizing, sin tomar sus juicios como benchmark propio.

## Problema y método

El problema que recorre la revisión es el diseño de sistemas híbridos renovables, aislados o conectados, con coste neto presente alto, arquitectura multidimensional e incertidumbre de recurso y demanda. El texto no ejecuta un caso: clasifica estudios previos por técnica y por criterio económico o de tamaño. Entre las familias aparecen programación matemática, métodos gráficos y probabilísticos, enjambres y evolutivos, y software como HOMER. Los híbridos se entienden aquí como combinación de algoritmos de optimización, no como un modelo neuronal ajustado por un optimizador.

## Qué sostiene el texto

Los autores sintetizan un desplazamiento desde el sizing clásico hacia técnicas modernas, porque estas entregan un rango de soluciones y no un único punto. Afirman que la IA converge más rápido que los métodos clásicos, pero exige equilibrar exploración y explotación y consume más diseño e implementación. HOMER se destaca por facilidad de uso, con límite en objetivos múltiples. Señalan que la mayoría de los trabajos optimiza solo electricidad y descuida calor, hidrógeno y almacenamiento de hidrógeno, y que no conocen un único software que arme un HRES con tecnología de hidrógeno. También reclaman análisis de sensibilidad ante carga, clima, degradación y aspectos sociales. Que CSA, Jaya o DHO superen a GA y PSO es lectura de literatura citada, no una corrida de este artículo.

## Límites

**Hallazgos de los autores.** La revisión no garantiza optimalidad global ni un ranking único: insiste en que el balance diversificación-intensificación condiciona el resultado y en que costes y COE no son comparables entre estudios.

**Lectura crítica.** No mide el hueco de un proyecto concreto. Que el resumen no nombre control adaptativo en línea, DTW o un marco como LB2 no demuestra que ese marco falte ni que estos papers lo validen. Las tablas resumen trabajos ajenos; cualquier cifra de un caso citado debe volver a su fuente.

## Relación con hibridación y renovables

Ubica la hibridación algorítmica como respuesta a las deficiencias de un solo método de IA en el sizing de HRES. El hidrógeno aparece como aplicación poco cubierta, no como resultado experimental de los autores.

## Evidencia

- p. 1: objetivo de la revisión y preferencia declarada por IA y por híbridos frente a métodos clásicos; HOMER como herramienta popular.
- pp. 27-28: límites de objetivos únicos, ausencia de software integrado para hidrógeno y necesidad de equilibrar diversificación e intensificación.
