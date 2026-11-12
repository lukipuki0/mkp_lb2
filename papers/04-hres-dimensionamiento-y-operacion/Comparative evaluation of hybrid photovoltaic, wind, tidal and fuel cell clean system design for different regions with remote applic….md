# Comparative evaluation of hybrid photovoltaic, wind, tidal and fuel cell clean system design for different regions with remote application considering cost

**Clave de cita:** `Naderipour_2021`  
**PDF:** [PDF](./Comparative%20evaluation%20of%20hybrid%20photovoltaic%2C%20wind%2C%20tidal%20and%20fuel%20cell%20clean%20system%20design%20for%20different%20regions%20with%20remote%20applic%E2%80%A6.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Amirreza Naderipour; Zulkurnain Abdul-Malek; Saber Arabi Nowdeh; Hesam Kamyab; Amir Reza Ramtin; Saman Shahrokhi; Jiří Jaromír Klemeš |
| Año | 2021 |
| Venue | Journal of Cleaner Production, 283, 124207 |
| DOI | 10.1016/j.jclepro.2020.124207 |
| Sección primaria | `04-hres-dimensionamiento-y-operacion` |
| Etiquetas | WOA; mareal; pila de combustible; tres regiones; fiabilidad |

## Aporte para el estado del arte

Compara configuraciones de un HRES remoto con fotovoltaica, viento, mareas y pila de combustible con almacenamiento de hidrógeno, en tres climas de Irán. El optimizador es whale optimisation, un poblacional único, no un híbrido de algoritmos ni un modelo de aprendizaje.

## Problema y método

El diseño minimiza el net present cost del sistema híbrido sujeto a una probabilidad de déficit de carga, y reporta también el coste de energía. Usan datos reales de radiación, viento y velocidad del agua en Gorgan, Urmia y Yazd. Evalúan siete combinaciones de componentes. WOA busca la configuración de potencias y tanque; PSO es el comparador. La pila y el tanque de hidrógeno son equipos del sistema, no una cadena de suministro. No hay control en línea de parámetros de WOA.

## Qué sostiene el texto

Para las tres regiones, la combinación de menor coste y mejor fiabilidad entre las siete es PV/WT/mareal/FC con almacenamiento de hidrógeno. El resumen da COE de 0,7789, 1,0864 y 0,4477 USD/kWh y LDP de 0,0090, 0,0093 y 0,0090 para Gorgan, Urmia y Yazd. La tabla 4 detalla, para WOA, potencias de mareas, PV, viento, electrolizador, tanque, pila e inversor; por ejemplo Gorgan queda en 430 kW mareales, 91 kW PV y 29 kW eólicos. Mareal pesa más en Gorgan y Yazd; PV pesa más en Urmia; el viento aporta menos. Los autores dicen que WOA supera a PSO en convergencia y precisión en este diseño. La p. 19 reescribe los LDP de forma ilegible en el texto extraído; uso las cifras del resumen, que coinciden con el COE repetido en la conclusión.

## Límites

**Hallazgos de los autores.** Los resultados son específicos de la carga y el recurso de esas tres ciudades. No optimizan emisiones como objetivo separado ni acoplan transporte.

**Lectura crítica.** WOA no es un híbrido algorítmico aunque el sistema energético sea híbrido. No hay aprendizaje sobre la traza ni DTW. La superioridad frente a PSO no se extiende a MILP o a NSGA. El hidrógeno aquí es almacenamiento de una pila remota, no hidrógeno verde de escala utility.

## Relación con renovables

Encaja en dimensionamiento regional con restricción de fiabilidad. La etiqueta de hidrógeno es secundaria y se limita al par electrolizador-pila-tanque.

## Evidencia

- p. 1: siete combinaciones, COE, LDP y papel de cada recurso.
- pp. 18-19: tabla 4 de capacidades y conclusión de que PV/WT/mareal/FC es la combinación elegida.

## Procedencia del archivo

- El nombre de archivo se acortó de forma visible con `…` para caber en la ruta de Windows. El título completo está en el encabezado de esta ficha.
