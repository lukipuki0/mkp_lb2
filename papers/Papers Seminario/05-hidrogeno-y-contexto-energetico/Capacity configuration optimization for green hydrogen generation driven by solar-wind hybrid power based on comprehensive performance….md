# Capacity configuration optimization for green hydrogen generation driven by solar-wind hybrid power based on comprehensive performance criteria

**Clave de cita:** `Su_2023`  
**PDF:** [PDF](./Capacity%20configuration%20optimization%20for%20green%20hydrogen%20generation%20driven%20by%20solar-wind%20hybrid%20power%20based%20on%20comprehensive%20performance%E2%80%A6.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Wei Su; Wenjin Zheng; Qi Li; Zhenyue Yu; Yunbin Han; Zhang Bai |
| Año | 2023 |
| Venue | Frontiers in Energy Research, 11, 1256463 |
| DOI | 10.3389/fenrg.2023.1256463 |
| Sección primaria | `05-hidrogeno-y-contexto-energetico` |
| Etiquetas | NSGA-III; multiobjetivo; hidrógeno verde; red eléctrica |

## Aporte para el estado del arte

Caso de configuración de capacidades para hidrógeno verde solar-eólico conectado a red, con tres criterios y un algoritmo evolutivo multiobjetivo. Documenta un uso de NSGA-III sobre la planta, no un híbrido que aprenda a controlar sus propios operadores.

## Problema y método

El sistema acopla generación solar-eólica, electrolizadores, baterías, tanques de hidrógeno y la red. La fluctuación del recurso obliga a dimensionar generación, producción y almacenamiento juntos. Los objetivos son LCOH, reducción de emisiones de carbono del conjunto eólico-fotovoltaico más la del hidrógeno, y tasa de energía abandonada. La maximización de la reducción de carbono se reescribe como minimización de su recíproco para encajar en NSGA-III. Las variables son las capacidades de aerogeneradores, fotovoltaica, electrolizador alcalino, baterías y tanques. Hay balance de potencia y una estrategia de operación coordinada; si falta recurso, se compra electricidad de red. El caso tiene una capacidad de producción de 20 000 t/año. IRR y payback se calculan con tasa de interés del 8 %, pero el optimizador no usa el IRR como objetivo.

## Qué sostiene el texto

En el caso estudiado, la tasa de energía abandonada baja a 3,32 % y el factor de carga medio del electrolizador queda en 64,77 %. Con precio de venta de 27,80 CNY/kgH2, el IRR llega al 8 %. El texto atribuye a la conexión a red la venta de excedentes y la reducción de ciclos de arranque-parada del electrolizador. No publica aquí el frente de Pareto completo en el pasaje revisado; las cifras citadas son las del caso implementado en el resumen.

## Límites

**Hallazgos de los autores.** Los números dependen del precio del hidrógeno, del interés del 8 % y del emplazamiento del caso. No comparan NSGA-III contra un MILP ni contra HOMER en las páginas de método revisadas.

**Lectura crítica.** Es un estudio de capacidad, no una validación de control adaptativo de metaheurísticas ni de DTW. NSGA-III se usa como solver externo. No debe inferirse que tres objetivos bastan en otro país: la transformación del máximo de carbono en un recíproco cambia la escala del objetivo y puede sesear el frente.

## Relación con hidrógeno y multiobjetivo

Etiqueta secundaria de multiobjetivo. El hidrógeno es producto, no solo almacenamiento estacional. Útil junto a los papers de cadena de suministro para bajar de la red nacional al diseño de planta.

## Evidencia

- p. 1: 20 000 t/año, 3,32 %, 64,77 % y 27,80 CNY/kg para IRR del 8 %.
- pp. 6-7: LCOH, CER, AEPR, cambio a minimización y variables de capacidad.

## Procedencia del archivo

- El nombre de archivo se acortó de forma visible con `…` para caber en la ruta de Windows. El título completo está en el encabezado de esta ficha.
