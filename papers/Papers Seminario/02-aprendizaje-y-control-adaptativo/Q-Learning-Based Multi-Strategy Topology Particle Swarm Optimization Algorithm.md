# Q-Learning-Based Multi-Strategy Topology Particle Swarm Optimization Algorithm

- PDF: [PDF](./Q-Learning-Based%20Multi-Strategy%20Topology%20Particle%20Swarm%20Optimization%20Algorithm.pdf)
- Bibliografía: [referencias.bib](../../referencias.bib)
- Clave de cita: `Hao_2025`

## Lectura útil

MSTPSO usa Q-learning para cambiar la topología del enjambre, no para reemplazar la actualización de partículas por una red neuronal.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Xiaoxi Hao, Shenwei Wang, Xiaotong Liu, Tianlei Wang, Guangfan Qiu, Zhiqiang Zeng |
| Soporte | *Algorithms* 2025, 18, 672; recibido 22 ago. 2025, publicado 22 oct. 2025 |
| DOI | 10.3390/a18110672 |
| Tipo | PSO con control tabular de topología y perturbaciones DE |

## Método

El fallo que atacan es la convergencia prematura y la topología fija en PSO de alta dimensión (p. 1). Q-learning elige entre topología totalmente informada (FIPS), mundo pequeño y conjunto de ejemplares. Además hay perturbación por evolución diferencial, reinicio del óptimo global si se detecta estancamiento, y una memoria de experiencia de dos niveles. Sigue siendo un PSO: la política solo decide la estructura de información entre partículas. Lo híbrido aquí es PSO más perturbación DE más un controlador Q, no un algoritmo de optimización híbrida en el sentido de portafolio de metaheurísticas.

## Qué sostiene el texto

En 29 funciones CEC2017, frente a variantes PSO y otros evolutivos, los autores reportan mejor fitness y más estabilidad en funciones de alta dimensión y complejas, en suites de 10, 30, 50 y 100 dimensiones (pp. 1 y 22–23). La ablación atribuye contribución crítica al control multi-topología y a la detección de estancamiento. El Wilcoxon de la Tabla 7 se usa para marcar superioridad significativa frente a la mayoría de los representativos; el extracto no fija el recuento exacto de signos, así que no se reproduce un «gana en k de n» no copiado.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Los parámetros de Q-learning son sensibles; entropía de diversidad y matrices de distancia encarecen poblaciones grandes; solo hay bancos, no problemas con restricciones ni ingeniería real (p. 23). |
| Lectura crítica | CEC2017 no sustituye un caso de control, series temporales o energía. La perturbación DE puede explicar parte de la ganancia; la ablación la menciona, pero un estado del arte no debe atribuir todo el efecto a Q-learning. No hay DTW ni HRES. |

## Para el estado del arte

Ejemplo de control discreto del grafo de comunicación del enjambre, complementario a quienes controlan F/CR o el operador de mutación. Sirve para separar «modelo que elige topología» de «PSO híbrido con otro optimizador».

Etiquetas secundarias: PSO; experiencia replay; CEC2017; balance exploración–explotación.

## Evidencia

- p. 1: tres topologías, DE, reinicio y 29 funciones CEC2017.
- pp. 22–23: conclusiones, Wilcoxon y tres límites explícitos.
