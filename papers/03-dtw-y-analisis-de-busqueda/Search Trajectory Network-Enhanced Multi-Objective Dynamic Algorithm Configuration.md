# Search Trajectory Network-Enhanced Multi-Objective Dynamic Algorithm Configuration

- PDF: [PDF](./Search%20Trajectory%20Network-Enhanced%20Multi-Objective%20Dynamic%20Algorithm%20Configuration.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Reijnen_2025`

## Lectura útil

Mete la red de trayectoria en el estado de un agente que reconfigura un algoritmo evolutivo multiobjetivo durante la corrida. No es el Reijnen de 2022 sobre DE.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Robbert Reijnen; Zaharah Bukhsh; Hoong Chuin Lau; Yaoxin Wu; Yingqian Zhang |
| Soporte | ECAI 2025, pp. 4921-4928 |
| DOI | 10.3233/FAIA251403 |
| Tipo | DAC multiobjetivo con STN |
| Páginas del PDF | 8 |

## Método

El resumen (p. 1) aprende representaciones de STN con una red de grafos para seguir convergencia, diversidad y su compromiso, y con eso reconfigura el algoritmo. El DOI está en el registro del PDF.

## Qué sostiene el texto

La p. 1 dice que supera métodos de configuración por DRL del estado del arte y que escala a instancias grandes. Esta ficha no releyó las tablas.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El resultado declarado es frente a configuradores DRL, no frente a un controlador por DTW. |
| Lectura crítica | Hay una política aprendida, pero la sección primaria es análisis de trayectoria porque el estado nuevo es la STN. Ochoa_2021 no entrena ese agente. No es el paper de SSCI 2022 citado en papers_que_sirven.md. |

## Para el estado del arte

Cierra el arco STN: de herramienta de visualización a estado de un configurador. Sigue siendo un modelo aprendido sobre el grafo, no una distancia de warping sobre el fitness.

Etiquetas secundarias: STN; grafo neuronal; refuerzo; multiobjetivo.

## Evidencia

- p. 1: DOI, STN como estado y la comparación declarada con configuradores DRL.

## Procedencia del archivo

El PDF llegó suelto en `papers/FAIA-413-FAIA251403.pdf` y se movió a esta sección. La ficha usa la apertura del PDF, no una relectura de las tablas de resultados.
