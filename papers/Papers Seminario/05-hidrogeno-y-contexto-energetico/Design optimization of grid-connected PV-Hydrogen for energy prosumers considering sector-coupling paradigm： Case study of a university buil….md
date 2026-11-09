# Design optimization of grid-connected PV-Hydrogen for energy prosumers considering sector-coupling paradigm: Case study of a university building in Algeria

**Clave de cita:** `Mokhtara_2021`  
**PDF:** [PDF](./Design%20optimization%20of%20grid-connected%20PV-Hydrogen%20for%20energy%20prosumers%20considering%20sector-coupling%20paradigm%EF%BC%9A%20Case%20study%20of%20a%20university%20buil%E2%80%A6.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Charafeddine Mokhtara; Belkhir Negrou; Noureddine Settou; Abdessalem Bouferrouk; Yufeng Yao |
| Año | 2021 |
| Venue | International Journal of Hydrogen Energy, 46(75), 37564-37582 |
| DOI | 10.1016/j.ijhydene.2020.10.069 |
| Sección primaria | `05-hidrogeno-y-contexto-energetico` |
| Etiquetas | prosumidor; acoplamiento edificio-tranvía; PSO; HOMER; ε-constraint |

## Aporte para el estado del arte

Caso de campus universitario en Ouargla que puede producir electricidad e hidrógeno para un tranvía de pila de combustible. Muestra cuándo el hidrógeno no compite en coste con PV más red, y cuándo el acoplamiento de sectores cambia esa conclusión. El solver es PSO y HOMER; NSGA-II solo aparece como literatura previa.

## Problema y método

El HRES es fotovoltaico, batería e hidrógeno, conectado a red, para autosuficiencia del campus y para hidrógeno del tranvía eléctrico. El problema multiobjetivo se convierte en uno solo por ε-constraint: se minimiza COE y se tratan LPSP y uso no renovable (NRU) como restricciones. Añaden pérdida de suministro de hidrógeno (LHSP). Hay estudio de sensibilidad a la demanda de hidrógeno del tranvía y al NRU, y dos escenarios. En el primero el edificio importa y exporta electricidad. En el segundo solo importa electricidad y exporta hidrógeno a una hidrogenera in situ. No hay metaheurística híbrida ni modelo de aprendizaje: PSO busca tamaños y HOMER simula.

## Qué sostiene el texto

Con NRUmax del 100 %, el sistema sin hidrógeno es la mejor solución económica: COE 0,016 USD/kWh, paridad de red y 13 % de NRU; la conclusión precisa RU del 87 % para ese grid-PV. Con NRUmax del 1 %, la configuración red/PV/electrolizador/pila/tanque alcanza 0 % de NRU y COE 0,1 USD/kWh, unas seis veces más. En el segundo escenario, aumentar el número de tranvías, y por tanto la demanda de hidrógeno, reduce LHSP, COE, NRU y emisiones de CO2 en las corridas comparadas, y sube la fracción renovable. La conclusión de los autores es doble: hoy gana el PV con red si solo importa la economía; con requisitos futuros de sistema energético, gana el PV con hidrógeno.

## Límites

**Hallazgos de los autores.** Los precios bajos de la red argelina hacen difícil justificar el tanque solo por fiabilidad del edificio. No extrapolan a electricidad cara. El acoplamiento exige un consumidor de hidrógeno cercano.

**Lectura crítica.** No valida un optimizador adaptativo ni DTW. PSO aquí no se compara con un MILP del mismo modelo, así que la optimalidad global no está demostrada. El ε-constraint fija el compromiso mediante umbrales elegidos, no mediante un frente explorado por NSGA. No debe citarse como prueba de que el hidrógeno siempre abarata el COE: el primer escenario dice lo contrario.

## Relación con hidrógeno

Es contexto de prosumidor y de acoplamiento edificio-transporte, complementario al utility-scale chino y a la cadena de suministro nacional. Etiqueta secundaria de dimensionamiento, con la salvedad de que el algoritmo no es híbrido.

## Evidencia

- pp. 1-2: ε-constraint, PSO, HOMER, COE 0,016 y 0,1 USD/kWh.
- pp. 14 y 16: lectura económica del primer escenario y efecto del número de tranvías en el segundo.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
- El nombre de archivo se acortó de forma visible con `…` para caber en la ruta de Windows. El título completo está en el encabezado de esta ficha.
