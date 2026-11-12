# Capacity optimization of a wind-photovoltaic-electrolysis-battery (WPEB) hybrid energy system for power and hydrogen generation

**Clave de cita:** `Li_2024a`  
**PDF:** [PDF](./Capacity%20optimization%20of%20a%20wind-photovoltaic-electrolysis-battery%20%28WPEB%29%20hybrid%20energy%20system%20for%20power%20and%20hydrogen%20generation.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Runzhao Li; Xiaoming Jin; Ping Yang; Ming Zheng; Yun Zheng; Chunrong Cai; Xiang Sun; Zhibin Luo; Luyao Zhao; Zhaohe Huang; Wenzhao Yang |
| Año | 2024 |
| Venue | International Journal of Hydrogen Energy, 52, 311-333 |
| DOI | 10.1016/j.ijhydene.2023.08.029 |
| Sección primaria | `05-hidrogeno-y-contexto-energetico` |
| Etiquetas | WPEB utility-scale; LCOE; gradiente; 8760 h; no es metaheurística poblacional |

## Aporte para el estado del arte

Trata el hueco que los autores ven entre el WPEB residencial o comercial, de lazo cerrado power-to-H2-to-power, y el utility-scale de lazo abierto power-to-H2 para recortar vertidos. El solver es descenso por gradiente sobre un dominio factible, no un enjambre.

## Problema y método

El sistema es eólico, fotovoltaico, electrolisis y batería, por encima de 10 MW, orientado a producir hidrógeno y reducir curtailment. El objetivo único es minimizar LCOE, con restricciones de balance de potencia, tasa anual de venta a red y capacidades de viento, fotovoltaica y estación de electrolisis. El dominio factible se identifica con simulación de 8760 h. Un año meteorológico típico se arma con una métrica de complementariedad viento-solar. La batería desplaza energía en el tiempo para recortar paradas del electrolizador. No hay población ni operador estocástico: la búsqueda del óptimo dentro del dominio es por gradiente.

## Qué sostiene el texto

El caso óptimo que reportan es W190-P10-E95-B30, con LCOE 0,2692 ¥/kWh y LCOH 14,1574 ¥/kg. IRR 12,60 % y ROI 78,97 %. Tres conclusiones de método: subir la proporción de viento reduce LCOE, la capacidad de venta a red y la volatilidad de la generación neta; los límites de capacidad de electrolisis salen de la restricción de venta a red y del objetivo de LCOE; la potencia de batería no debería quedar por debajo del mínimo técnico del electrolizador y prefieren descarga de más de 1 h. Esas frases son del resumen, ligadas a este protocolo.

## Límites

**Hallazgos de los autores.** El trabajo se presenta como hoja de ruta para utility-scale en zonas con mucho viento y sol, con datos de curtailment provincial chino en las figuras. No optimiza emisiones como objetivo separado.

**Lectura crítica.** El gradiente puede quedarse en un óptimo local si el LCOE simulado no es convexo; el texto no demuestra convexidad. No compara contra NSGA-III ni contra un metaheurístico híbrido, así que no rankea familias de solvers. No hay DTW ni control adaptativo de la búsqueda. Las unidades en yuanes no se convierten en el artículo a dólares.

## Relación con hidrógeno

Complementa los casos multiobjetivo: aquí el hidrógeno entra en el LCOH, pero la búsqueda es monoobjetivo y determinista. Conviene no etiquetarlo como híbrido algorítmico; el híbrido es la planta WPEB.

## Evidencia

- p. 1: arquitectura de lazo abierto, método de gradiente y tres conclusiones.
- p. 1, highlights: caso W190-P10-E95-B30 y cifras de LCOE, LCOH, IRR y ROI.

## Procedencia del archivo

- La clave propuesta `Li_2024` ya identificaba otro estudio. Esta ficha y su entrada BibTeX usan `Li_2024a`.
