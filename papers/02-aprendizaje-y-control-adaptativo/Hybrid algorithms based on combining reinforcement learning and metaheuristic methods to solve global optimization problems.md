# Hybrid algorithms based on combining reinforcement learning and metaheuristic methods to solve global optimization problems

- PDF: [PDF](./Hybrid%20algorithms%20based%20on%20combining%20reinforcement%20learning%20and%20metaheuristic%20methods%20to%20solve%20global%20optimization%20problems.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Seyyedabbasi_2021`

## Lectura útil

Tres variantes —RLI−GWO, RLEx−GWO y RLWOA— usan una Q-table para decidir si el agente de búsqueda explora o explota. El optimizador sigue siendo GWO o WOA, no la red de Q-learning.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Amir Seyyedabbasi, Royal Aliyev, Farzad Kiani, Murat Ugur Gulle, Hasan Basyildiz, Mohammed Ahmed Shah |
| Soporte | *Knowledge-Based Systems* 223 (2021) 107044; en línea 22 abr. 2021 |
| DOI | 10.1016/j.knosys.2021.107044 |
| Tipo | Hibridación Q-learning + metaheurística, con bancos y un caso de cinemática |

## Método

Cada acción de la Q-table tiene recompensa o castigo según la comparación entre información actual y previa; eso guía al agente desde la explotación hacia regiones no visitadas (p. 19). Las bases son I-GWO, Ex-GWO y WOA. No es un portafolio que elija entre GWO y WOA en cada iteración: hay tres algoritmos híbridos separados. La evaluación usa 30 funciones de CEC 2014 y 2015, comparadas con GWO, RLGWO, I-GWO, Ex-GWO y WOA, y además la cinemática inversa de brazos de robot (p. 1).

## Qué sostiene el texto

Los autores dicen que las tres propuestas equilibran mejor exploración y explotación y convergen bien, y que RLWOA da mejores soluciones en los problemas relevantes, incluido el brazo (pp. 1 y 19). No se extrae de esta lectura un ranking numérico función por función.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El cierre describe el mecanismo y la convergencia, sin un análisis de sensibilidad de la tabla Q ni un costo de entrenamiento desglosado en el pasaje final (p. 19). |
| Lectura crítica | CEC 2014/2015 y un problema de cinemática no informan sobre energía, hidrógeno ni DTW. Llamarlos «hybrid algorithms» aquí significa controlador tabular sobre la fase de búsqueda, no fusión de GWO con WOA. RLWOA «mejor» es el juicio del resumen frente a los rivales listados, no una dominancia general. |

## Para el estado del arte

Antecedente de 2021 del uso de Q-learning para conmutar exploración y explotación dentro de GWO/WOA. Maleki (2025) ataca otro punto del mismo GWO: la actualización de líderes, no solo la fase. Conviene no fundir ambos en «RL mejora GWO» sin decir qué decisión controla cada uno.

Etiquetas secundarias: Q-learning; GWO; WOA; cinemática inversa; CEC 2014/2015.

## Evidencia

- p. 1: tres nombres, 30 funciones y caso del brazo.
- p. 19: Q-table, recompensa/castigo y conclusión.
