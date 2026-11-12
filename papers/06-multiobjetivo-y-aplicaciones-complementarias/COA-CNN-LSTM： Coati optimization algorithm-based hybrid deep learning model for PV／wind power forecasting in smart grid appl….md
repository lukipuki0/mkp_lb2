# COA-CNN-LSTM: Coati optimization algorithm-based hybrid deep learning model for PV/wind power forecasting in smart grid applications

**Clave de cita:** `AbouHouran_2023`  
**PDF:** [PDF](./COA-CNN-LSTM%EF%BC%9A%20Coati%20optimization%20algorithm-based%20hybrid%20deep%20learning%20model%20for%20PV%EF%BC%8Fwind%20power%20forecasting%20in%20smart%20grid%20appl%E2%80%A6.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Mohamad Abou Houran; Syed M. Salman Bukhari; Muhammad Hamza Zafar; Majad Mansoor; Wenjie Chen |
| Año | 2023 |
| Venue | Applied Energy, 349, 121638 |
| DOI | 10.1016/j.apenergy.2023.121638 |
| Sección primaria | `06-multiobjetivo-y-aplicaciones-complementarias` |
| Etiquetas | pronóstico; CNN-LSTM; Coati como afinador; no es híbrido de búsqueda energética |

## Aporte para el estado del arte

Documenta un uso frecuente y distinto: un metaheurístico ajusta hiperparámetros de una red que pronostica potencia solar y eólica. No dimensiona la planta ni controla el balance interno de una metaheurística de optimización.

## Problema y método

El problema es el pronóstico de corto plazo, día adelante y hora adelante, cuando viento y sol dependen de velocidad de viento, temperatura y humedad. La arquitectura de predicción es CNN-LSTM. El Coati optimization algorithm busca hiperparámetros de esa red para mejorar tasa de aprendizaje y error. Hay una prueba de causalidad de Granger entre generación eólica y variables meteorológicas, y métricas de error y de ajuste de serie, entre ellas Nash-Sutcliffe. Los comparadores son GWO-CNN-LSTM, LSTM, CNN y PSO-CNN-LSTM. El híbrido del título es red convolucional más recurrente, más un optimizador de hiperparámetros. No es la taxonomía de Talbi aplicada a dos metaheurísticas que busquen el mismo óptimo de diseño.

## Qué sostiene el texto

El resumen reporta que el RMSE de corto plazo baja 0,5 % en día adelante y 5,8 % en hora adelante, y da nMAE 4,6 %, RE 27 % y nRMSE 6,2 % frente a esos comparadores. Granger 0,0992 y Nash-Sutcliffe 0,98. La conclusión dice que el modelo fue eficiente en ocho emplazamientos. Esas cifras son resultados experimentales de los autores en sus series, no una cota teórica. RE no queda definido de forma inequívoca en el resumen; no lo reinterpretaré como un 27 % de mejora.

## Límites

**Hallazgos de los autores.** El pronóstico no sustituye un EMS. El texto no muestra que el mismo COA sirva para sizing. La complejidad del optimizador se discute como criterio de usabilidad, no como coste medido en todos los sitios.

**Lectura crítica.** Afinar una LSTM con un bioinspirado no es un algoritmo híbrido de optimización global ni un controlador adaptativo de exploración. No hay DTW: la comparación de series usa error de pronóstico y Nash-Sutcliffe. No valida un proyecto de balanceo por regresión. Ocho sitios no autorizan extrapolar a cualquier red inteligente.

## Relación con renovables

Aporta la capa de predicción que otros papers de HRES piden para la carga y el recurso. Debe citarse en aplicaciones complementarias, con la etiqueta explícita de modelo neuronal más optimizador.

## Evidencia

- p. 1: arquitectura, comparadores y métricas del resumen.
- p. 16: conclusión sobre ocho emplazamientos y uso de Granger dentro del pronóstico.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (/, :) por equivalentes de ancho completo. El título de la ficha permanece completo.
- El nombre de archivo se acortó de forma visible con `…` para caber en la ruta de Windows. El título completo está en el encabezado de esta ficha.
