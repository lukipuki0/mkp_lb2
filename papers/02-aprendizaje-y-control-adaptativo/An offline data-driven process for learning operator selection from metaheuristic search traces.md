# An offline data-driven process for learning operator selection from metaheuristic search traces

- PDF: [PDF](./An%20offline%20data-driven%20process%20for%20learning%20operator%20selection%20from%20metaheuristic%20search%20traces.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Kalatzantonakis_2025`

## Lectura útil

TRACE entrena clasificadores supervisados, offline, para predecir si un vecindario de VNS mejorará la solución; no es un agente de refuerzo que explore durante la búsqueda.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Panagiotis Kalatzantonakis, Angelo Sifaleras, Nikolaos Samaras |
| Soporte | *Swarm and Evolutionary Computation* 98 (2025) 102058; recibido 10 may. 2025, en línea 19 jul. 2025 |
| DOI | 10.1016/j.swevo.2025.102058 |
| Tipo | Proceso de aprendizaje offline aplicado a GVNS en CVRP |

## Método

GVNS registra historia de acciones, métricas de grafo, estado temporal e indicadores UCB, etiquetados por si el operador mejoró la solución. En despliegue, TRACE-VNS consulta el modelo y no mantiene recompensa (Tabla 4, p. 4). Doce clasificadores —árboles, redes y métodos de núcleo— se comparan por PR-AUC en CVRP, 84 instancias. UCB es un rasgo, no la política de producción. El ganador no es un optimizador neuronal.

## Qué sostiene el texto

LightGBM alcanza el mayor PR-AUC, 0,94; Random Forest y Extra Trees también rinden fuerte (p. 16). TRACE-VNS mejora velocidad de convergencia y calidad final frente a GVNS convencional en las 84 instancias (p. 1). Los autores escriben ganancias de calidad por encima del 30 % (p. 16); esa cifra es suya y depende del protocolo del artículo. La historia de recompensa y el estado de búsqueda dominan la importancia; las métricas de grafo aportan contexto. No hicieron ablación de grupos de rasgos: con millones de filas la juzgaron inviable (p. 16).

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El modelo queda fijo tras el entrenamiento; no se adapta a la instancia en curso. La ablación completa no se hizo. Generalizar a otras metaheurísticas exige adaptación estructural, aunque la describen como mínima (pp. 1 y 4). |
| Lectura crítica | CVRP no es un problema continuo ni energético. Un PR-AUC alto no implica que el mismo modelo sirva para mutaciones de DE. UCB en las trazas puede filtrarse al clasificador y confundirse con «el método es RL». No hay DTW ni HRES. |

## Para el estado del arte

Contraste entre predicción offline del éxito de un operador y control online por Q-learning, PPO o bandits.

Etiquetas secundarias: VNS; CVRP; clasificadores supervisados; rasgos de grafo.

## Evidencia

- p. 1: definición de TRACE, 12 clasificadores, 84 instancias.
- p. 4: Tabla 4, offline frente a online, y rasgos.
- p. 16: PR-AUC 0,94, ganancia declarada y límite de la ablación.
