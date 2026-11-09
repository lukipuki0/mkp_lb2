# Recent advances in selection hyper-heuristics

- PDF: [PDF](./Recent%20advances%20in%20selection%20hyper-heuristics.pdf)
- Bibliografía: [referencias.bib](../../referencias.bib)
- Clave de cita: `Drake_2020`

## Lectura útil

Revisión crítica, posterior a Burke et al. (2013), limitada a hiperheurísticas de selección: historia, clasificación ampliada, HyFlex/CHeSC y limitaciones de evaluación.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | John H. Drake, Ahmed Kheiri, Ender Özcan, Edmund K. Burke |
| Soporte | *European Journal of Operational Research* 285 (2020) 405–428; invitado; en línea 7 ago. 2019 |
| DOI | 10.1016/j.ejor.2019.07.073 |
| Tipo | Survey de selección; no propone un selector nuevo |

## Alcance

Definen la hiperheurística de selección como búsqueda sobre heurísticas de bajo nivel —movimientos, búsquedas locales o incluso metaheurísticas— con barrera de dominio: el nivel alto no debe leer información específica del problema (pp. 1–2). El corpus se restringe a artículos completos posteriores a Burke et al. (2013). La clasificación amplía la de Burke et al. (2010) según la retroalimentación, la naturaleza de las heurísticas y otros ejes (Fig. 1, p. 2).

El eje empírico es HyFlex y CHeSC 2011. Con el sistema de puntos tipo Fórmula 1, AdapHH obtiene 181,00 y lidera la tabla reproducida (p. 7). Los autores advierten que otra métrica puede cambiar lecturas, aunque Adriaensen et al. (2015), con seis métricas, seguían colocando a AdapHH primero (p. 8). También separan selección de operadores, selección entre metaheurísticas, construcción de soluciones y métodos generativos que diseñan el propio selector (secciones 6–8).

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Falta evaluación incremental (delta); el presupuesto de 10 minutos nominales de CHeSC no iguala el número de llamadas entre dominios; hay poca teoría (p. 21). |
| Lectura crítica | Es una fotografía hasta el cierre posterior a 2012, no del estado 2024–2026. HyFlex no incluye control continuo, DTW ni sistemas energéticos. El ranking CHeSC no valida un controlador de parámetros de un proyecto posterior. |

## Para el estado del arte

Citarlo para la distinción selección/generación, la barrera de dominio y la precaución metodológica: el esquema de puntuación altera conclusiones. Complementa, no sustituye, a Dokeroglu (2024), que añade paralelismo y aprendizaje reciente, ni a Li (2024), centrado en RL.

Etiquetas secundarias: HyFlex; CHeSC 2011; multiobjetivo como sección del survey.

## Evidencia

- pp. 1–2: definición, barrera de dominio y alcance posterior a 2013.
- p. 7: puntuaciones CHeSC; AdapHH 181,00.
- p. 8: sensibilidad de la métrica.
- p. 21: evaluación incremental y desbalance de esfuerzo.
