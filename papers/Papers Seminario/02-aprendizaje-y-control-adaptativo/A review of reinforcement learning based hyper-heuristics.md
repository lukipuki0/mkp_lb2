# A review of reinforcement learning based hyper-heuristics

- PDF: [PDF](./A%20review%20of%20reinforcement%20learning%20based%20hyper-heuristics.pdf)
- Bibliografía: [referencias.bib](../../referencias.bib)
- Clave de cita: `Li_2024`

## Lectura útil

Revisión de hiperheurísticas cuyo selector es aprendizaje por refuerzo: separa métodos basados en valor y métodos basados en política, y propone un marco general. No entrena ningún agente.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Cuixia Li, Xiang Wei, Jing Wang, Shuozhe Wang, Shuyan Zhang |
| Soporte | *PeerJ Computer Science* 10:e2141, 2024; enviado 12 mar. 2024, publicado 28 jun. 2024 |
| DOI | 10.7717/peerj-cs.2141 |
| Tipo | Survey; 31 páginas de revista, 32 segmentos en el extracto por el salto de forma final |

## Alcance

Definen RL-HH como la combinación de la búsqueda global de la hiperheurística y el ajuste de estrategia del agente (p. 1). El marco distingue valor (SARSA, Q-learning tabular, DQN, DDQN, D3QN) y política. En valor, el agente actualiza la puntuación de la heurística de bajo nivel según la solución actual (p. 8). En política, describen el costo de optimizar la política en estados y acciones de alta dimensión y el riesgo de degradación (p. 17). El cierre compara ambos: el valor informa sobre la estructura pero sufre la maldición de la dimensionalidad; la política escala mejor a acciones continuas y consume más cómputo (p. 19).

## Límites que el propio survey enumera

Falta análisis teórico de convergencia, velocidad y estabilidad; los autores listan además otras carencias del campo en la p. 20, empezando por esa. No reportan un metaanálisis con tamaños de efecto.

## Lectura crítica

La dicotomía valor/política es útil, pero varios trabajos del lote no caben limpios: TRACE es supervisión offline; MAB-OS es bandit sobre optimizadores completos; DRLDE es PPO sobre acciones de DE, no sobre un pool de heurísticas combinatorias. El survey no cubre DTW, HRES ni hidrógeno, y esa omisión no es un hueco empírico demostrado. Tampoco valida que RL-HH supere a la selección sin aprendizaje en un dominio concreto.

## Para el estado del arte

Mapa para clasificar el resto del lote antes de comparar cifras. Citar junto a Drake (selección sin exigir RL) y Dokeroglu (taxonomía más amplia), no como sustituto de los experimentos de cada algoritmo.

Etiquetas secundarias: Q-learning; DQN; política; marco conceptual de RL-HH.

## Evidencia

- p. 1: definición, vacío que declaran cubrir y dos categorías.
- pp. 6–8: DQN/DDQN/D3QN y actualización de heurísticas de bajo nivel.
- pp. 17–20: costo de la política y límites teóricos.
