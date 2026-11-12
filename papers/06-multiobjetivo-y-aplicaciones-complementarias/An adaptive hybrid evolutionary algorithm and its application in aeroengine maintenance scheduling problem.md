# An adaptive hybrid evolutionary algorithm and its application in aeroengine maintenance scheduling problem

- PDF: [PDF](./An%20adaptive%20hybrid%20evolutionary%20algorithm%20and%20its%20application%20in%20aeroengine%20maintenance%20scheduling%20problem.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Fu_2021`

## Lectura útil

MOEA/D-IHO no es un híbrido DE+enjambre fijo: dentro de MOEA/D alterna `DE/rand/1` y un operador de búsqueda de cuervos según cómo cambian GD y MS en ventanas de 7 iteraciones.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Guo-Zhong Fu, Hong-Zhong Huang, Yan-Feng Li, Jie Zhou |
| Soporte | *Soft Computing* (2021) 25:6527–6538; aceptado 1 feb. 2021, publicado 28 feb. 2021 |
| DOI | 10.1007/s00500-021-05647-y |
| Tipo | Algoritmo multiobjetivo con indicador colaborativo y un caso de mantenimiento |

## Método

El selector mira la variación de generational distance y maximum spread durante 7 iteraciones y fija el operador de las 7 siguientes (p. 1). El pool es `DE/rand/1` con cruce binomial y un crow search operator con un tercer cuervo (p. 4, ecs. 4–6), dentro de MOEA/D. En motores hay 5 unidades, población 200 y 200 generaciones; las variables continuas se redondean. Los objetivos incluyen makespan e intercambios de componentes (pp. 8–9).

## Qué sostiene el texto

En los bancos usados, los diagramas de caja de HV e IGD (Figs. 3 y 4) y el texto de la sección 5 presentan a MOEA/D-IHO como casi el mejor frente a pares que también usan el operador DE (p. 10). En el caso de mantenimiento obtienen soluciones duplicadas y muestran 7 representativas (Fig. 5, p. 10). No publican, en el pasaje revisado, un porcentaje de mejora frente a Kleeman y Lamont (2007); la aplicación demuestra que el frente puede construirse, no que domine la política industrial.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | No probaron 3 o más objetivos; F y CR de DE no se autoajustan; sugieren FSTDE u otros generadores de descendencia (p. 10). |
| Lectura crítica | La ventana de 7 iteraciones es una regla fija, no un aprendizaje entre problemas. El caso tiene 5 motores y redondeo heurístico. No hay vínculo con hidrógeno, HRES ni DTW. GD exige una referencia del frente; eso condiciona el selector fuera de bancos con frente conocido. |

## Para el estado del arte

Selección de operador por indicadores del frente, sin Q-learning ni PPO. Útil como aplicación multiobjetivo, no como control aprendido.

Etiquetas secundarias: MOEA/D; mantenimiento de motores aeronáuticos; indicadores GD/MS/HV/IGD.

## Evidencia

- p. 1: ventana de 7 iteraciones, pool DE/CSO y caso final.
- p. 4: formulación del CSO y del mecanismo colaborativo.
- pp. 8–10: HV, IGD, 5 motores, población 200 y límites declarados.
