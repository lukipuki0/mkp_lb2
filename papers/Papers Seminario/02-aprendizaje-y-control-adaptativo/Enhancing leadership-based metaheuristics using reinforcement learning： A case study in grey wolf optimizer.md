# Enhancing leadership-based metaheuristics using reinforcement learning: A case study in grey wolf optimizer

- PDF: [PDF](./Enhancing%20leadership-based%20metaheuristics%20using%20reinforcement%20learning%EF%BC%9A%20A%20case%20study%20in%20grey%20wolf%20optimizer.pdf)
- Bibliografía: [referencias.bib](../../referencias.bib)
- Clave de cita: `Maleki_2025`

## Lectura útil

RL-LGWO no sustituye GWO por un agente: el refuerzo multiagente desacopla y adapta cómo se actualizan los líderes, para que no se estanquen.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Afifeh Maleki, Mehdy Roayaei, Seyedali Mirjalili |
| Soporte | *Knowledge-Based Systems* 330 (2025) 114471 |
| DOI | 10.1016/j.knosys.2025.114471 |
| Tipo | Variante de GWO con RL sobre el mecanismo de liderazgo |

## Método

El diagnóstico es específico de metaheurísticas con líderes: poca exploración, líder estancado, pérdida de diversidad y poco uso de los datos de la búsqueda (p. 1). Los agentes comparten experiencia; el RL separa la actualización del líder del esquema fijo alfa–beta–delta. El lazo de búsqueda sigue siendo GWO. Lo comparan con dos mejoras de GWO, tres variantes GWO ya basadas en RL, PSO, WOA y GWO original, en 23 funciones clásicas y en CEC2022.

## Qué sostiene el texto

Mejor solución en 17 de 23 funciones clásicas, con convergencia y estabilidad superiores y un aumento menor de tiempo frente a GWO (p. 1). En CEC2022, mejor que los competidores en 10 de 12 funciones. Esas fracciones son del resumen de los autores; no se han recontado desde las tablas completas en esta ficha. No reportan un caso industrial.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El aumento de tiempo existe, aunque lo califican de menor. El resumen proyecta el mecanismo a otras metaheurísticas con liderazgo; eso no está medido fuera de GWO. |
| Lectura crítica | Ganar en 17/23 y 10/12 depende del conjunto de rivales elegido, varios de ellos también GWO. No es evidencia sobre DE, VNS, control de F/CR, DTW, HRES o hidrógeno. Compartir experiencia entre agentes no es transferencia entre problemas, a diferencia de Durgut. |

## Para el estado del arte

Caso de RL incrustado en la regla social del enjambre, distinto de usar RL para elegir un optimizador entero (MAB-OS) o una topología (MSTPSO). Útil en hibridación aprendizaje–metaheurística, no en energía.

Etiquetas secundarias: GWO; multiagente; CEC2022; exploración–explotación.

## Evidencia

- p. 1: mecanismo, rivales y resultados 17/23 y 10/12.
- p. 4: tabla de estudios híbridos metaheurística–RL y sus límites, como contexto de los autores.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
