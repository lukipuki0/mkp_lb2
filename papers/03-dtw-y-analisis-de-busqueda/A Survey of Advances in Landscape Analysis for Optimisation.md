# A Survey of Advances in Landscape Analysis for Optimisation

- PDF: [PDF](./A%20Survey%20of%20Advances%20in%20Landscape%20Analysis%20for%20Optimisation.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Malan_2021`

## Lectura útil

Seguimiento del survey de 2013: la análisis de paisajes pasó de herramienta teórica a soporte práctico para explicar algoritmos, predecir desempeño y configurar o seleccionar algoritmos. No propone una medida nueva ni un experimento propio.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autora | Katherine Mary Malan, University of South Africa |
| Soporte | *Algorithms* 2021, 14, 40; recibido 30 nov. 2020, publicado 28 ene. 2021 |
| DOI | 10.3390/a14020040 |
| Tipo | Survey de técnicas y aplicaciones |

## Alcance

Además del paisaje de fitness clásico, recoge paisajes multiobjetivo, de violación de restricciones, dinámicos y acoplados, y paisajes de error o pérdida en entrenamiento de redes (p. 1, sección 2). Incorpora 11 técnicas de análisis posteriores al survey de 2013 y revisa muestreo y robustez de las medidas. Las aplicaciones que organiza son cuatro: entender problemas complejos, explicar comportamiento de algoritmos, predecir desempeño y automatizar configuración o selección (p. 10). El financiamiento declarado es National Research Foundation de Sudáfrica, grant 120837 (p. 10).

## Límites

| Origen | Límite |
| --- | --- |
| Autora | El cierre es programático: la análisis reduce imprevisibilidad, pero no cuantifica cuántas técnicas sobreviven a un cambio de muestra. |
| Lectura crítica | Ella misma cita que el análisis exploratorio es muy sensible a la estrategia de muestreo (Renau et al., referencia en p. 13). Por tanto, meter rasgos de paisaje en un estado de RL —como hace DRLDE— hereda ese riesgo; este survey no lo cuantifica. No trata DTW, series de potencia ni hidrógeno. «Error landscapes» de redes no son un optimizador híbrido. |

## Para el estado del arte

Base conceptual cuando un controlador usa rasgos del paisaje o cuando se discute exploración y explotación sin definir el paisaje. Complementa a Ravber (2026), que mide exploración por cuencas, y a Ding (2025), que los usa como estado.

Etiquetas secundarias: selección automática de algoritmos; paisajes multiobjetivo; paisajes de pérdida.

## Evidencia

- p. 1: alcance respecto de 2013 y tipos nuevos de paisaje.
- p. 10: cuatro usos aplicados y conclusión.
- p. 13: sensibilidad al muestreo, vía referencias del propio artículo.
