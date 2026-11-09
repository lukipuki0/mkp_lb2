# Steady-state modelling of hybrid energy system for off grid electrification of cluster of villages

Ficha de lectura para el estado del arte. PDF: [PDF](./Steady-state%20modelling%20of%20hybrid%20energy%20system%20for%20off%20grid%20electrification%20of%20cluster%20of%20villages.pdf). Entrada bibliográfica: [referencias.bib](../../referencias.bib). Clave BibTeX: `Gupta_2010`.

Etiquetas secundarias: MILP, despacho combinado, penetración fotovoltaica, batería y diésel. Sección primaria: sistemas híbridos renovables, dimensionamiento y operación.

## Qué aporta

Gupta, Saini y Sharma (2010) formulan un modelo de programación lineal entera mixta, en serie temporal, para la operación y la configuración de un híbrido aislado que electrifica un grupo de aldeas. No es una metaheurística. El caso es el bloque Narendra Nagar, distrito de Tehri Garhwal, Uttarakhand (p. 9).

## Problema, método y contribución

El híbrido combina microhidráulica, biogás, biomasa, fotovoltaica, baterías y diésel, con buses de CA y CC. El modelo responde cómo repartir la potencia, qué despacho usar y cuál es la penetración económica del área fotovoltaica. Un factor P mide esa penetración. El factor EPDF (e), a lo sumo 1, representa la fluctuación del recurso hidráulico de menor coste unitario. La función de coste introduce una constante coste/unidad para que los recursos más baratos cubran más demanda. El control, en C++, usa despacho combinado y balance horario. Extienden modelos que no calculaban la penetración económica de la FV. No hay electrolizador ni pila; la conclusión dice que el algoritmo podría extenderse a eólica y pila (p. 15).

## Resultados que el texto sostiene

La tabla 13 (p. 15) reporta costes unitarios mensuales óptimos del orden de 5,65 a 6,45 Rs/kWh en los meses tabulados, con mínimos horarios de 1,45 o 3,98 Rs/kWh y máximos horarios de hasta 17,72 Rs/kWh. El texto inmediato indica que el coste unitario de generación pasa de 6,23 a 7,17 Rs en el análisis asociado. La conclusión afirma que el algoritmo diseña un sistema de mínimo coste y mantiene el diésel en salida constante y de alta eficiencia pese a la fluctuación fotovoltaica. El modelo es modular y de estado estacionario horario, no dinámico de pequeña señal.

## Límites

Precios y recurso son de un diseño mensual en rupias de la época del estudio. No hay incertidumbre estocástica ni frente de emisiones.

Lectura crítica: es un antecedente de despacho económico y de penetración FV, útil para contrastar métodos exactos con GA, PSO o DE de papers posteriores del mismo tipo de microrred rural india. No valida un controlador de metaheurísticas ni un uso de DTW. La ausencia de hidrógeno es una elección de sistema, no un hueco demostrado.

## Evidencia

Título, autores, *Renewable Energy* 35 (2010) 520-535 y DOI en la p. 1. Preguntas de diseño en las pp. 1-2. Caso en la p. 9. Tabla 13 y conclusiones en la p. 15.
