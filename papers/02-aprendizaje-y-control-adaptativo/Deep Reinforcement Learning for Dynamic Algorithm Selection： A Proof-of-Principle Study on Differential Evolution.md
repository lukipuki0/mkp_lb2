# Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution

- PDF: [PDF](./Deep%20Reinforcement%20Learning%20for%20Dynamic%20Algorithm%20Selection%EF%BC%9A%20A%20Proof-of-Principle%20Study%20on%20Differential%20Evolution.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Guo_2024`

## Lectura útil

Entrena un agente para elegir, durante la corrida, qué algoritmo del grupo aplicar. El estudio de principio es sobre variantes de DE.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Hongshu Guo; Yining Ma; Zeyuan Ma; Jiacheng Chen; Xinglin Zhang; Zhiguang Cao; Jun Zhang; Yue-Jiao Gong |
| Soporte | IEEE Transactions on Systems, Man, and Cybernetics: Systems, 54(7), 4247-4259 |
| DOI | 10.1109/TSMC.2024.3374889 |
| Tipo | Selección dinámica de variantes por DRL |
| Páginas del PDF | 13 |

## Método

El resumen (p. 1) modela la selección como un proceso de decisión de Markov y entrena por policy gradient. El estado usa rasgos de paisaje y del algoritmo. Al cambiar, restaura el contexto del algoritmo elegido.

## Qué sostiene el texto

La p. 1 presenta el marco como genérico y capaz de cambiar de algoritmo en línea. No se copia aquí el margen contra los nueve algoritmos de la comparación, porque el extracto de apertura no lo cierra.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Es un proof of principle. El objeto declarado es DE. |
| Lectura crítica | Selecciona algoritmos, no operadores sueltos. Los rasgos no son DTW de la serie de fitness. No es MAB-OS, que no usa paisaje. |

## Para el estado del arte

Ejemplo de política aprendida para rotar variantes completas. El cambio con restauración de contexto es distinto de inyectar solo el elite.

Etiquetas secundarias: DE; policy gradient; rasgos de paisaje; cambio de algoritmo.

## Evidencia

- p. 1: DOI, policy gradient, rasgos de paisaje y algoritmo, y la restauración de contexto.

## Procedencia del archivo

El PDF llegó suelto en `papers/Deep-Reinforcement-Learning-for-Dynamic-Algorithm-Selection-A-Proof-of-Principle-Study-on.pdf` y se movió a esta sección. La clave ya existía en `referencias.bib`. La ficha usa la apertura del PDF, no una relectura de las tablas.
