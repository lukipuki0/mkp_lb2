# Selección de gráficos y diagramas

Guía de consulta previa a generar o sugerir una figura. No existe un gráfico universalmente mejor: la elección depende de la pregunta, la estructura de los datos, la evidencia y el espacio de lectura. Estas recomendaciones son criterios operativos de la skill, no resultados ni decisiones metodológicas de MASRE.

## Consulta complementaria obligatoria

En cada generación, adaptación o recomendación, lee también el [catálogo operativo de las infografías aportadas](catalogo-infografias.md): traduce las ocho páginas del PDF y la imagen a variantes, condiciones de uso y correcciones. No adoptes una variante solo por aparecer en el catálogo. Si una representación pedida no corresponde a los datos, explica el problema y sugiere una alternativa mejor respaldada.

## 1. Decidir antes de dibujar

1. Formula qué debe poder comparar o comprender el lector en una frase.
2. Identifica la unidad de observación: persona, respuesta, estudio, requisito, ejecución u otra documentada. Distingue datos individuales de resúmenes publicados.
3. Comprueba variables y estructura: categorías nominales u ordinales, cantidades, tiempo, pares vinculados, jerarquías o relaciones. Revisa unidades, denominadores, faltantes y solapamientos.
4. Consulta la matriz siguiente y elige la representación más simple que conserve lo relevante. Una tabla o ninguna figura también pueden ser la mejor opción.
5. Elige después la herramienta según el [flujo](flujo.md#elección-de-herramienta), no al revés.
6. Registra brevemente la elección y sus límites en el README externo. No incrustes la justificación en la figura.

Si faltan datos, pregunta solo lo que cambie la representación o interpretación: por ejemplo, «¿los porcentajes son sobre personas o sobre respuestas y admiten selección múltiple?». No elijas silenciosamente un denominador, agregación o codificación de entrevistas. Si la elección ya está respaldada, procede sin pedir aprobación por decisiones estéticas reversibles.

## 2. Matriz por intención

| Qué debe comprender el lector | Estructura necesaria | Primera opción | Cuándo cambiarla o evitarla |
| --- | --- | --- | --- |
| Comparar cantidades entre categorías | Una cantidad comparable por categoría | Barras o puntos; barras horizontales para etiquetas largas | Conserva el orden si es ordinal; ordena por valor si interesa el ranking. Usa tabla cuando importe leer cada cifra exacta. |
| Comparar grupos dentro de categorías | Mismas medidas y unidades en cada grupo | Barras agrupadas o puntos por grupo | Si se superponen demasiadas series, usa paneles pequeños con escalas comparables; no apiles si interesa comparar cada componente con precisión. |
| Mostrar evolución | Tiempo u otra variable ordenada y valores asociados | Líneas con puntos cuando ayuden a identificar observaciones | Respeta distancias temporales reales y faltantes; no conectes categorías nominales. Barras pueden convenir para cantidades de períodos discretos. |
| Mostrar cambio antes/después | Mediciones vinculadas de las mismas unidades | Puntos conectados o gráfico de pendientes | Sin vínculo individual, compara grupos por separado; no inventes pares ni atribuyas causalidad al cambio. |
| Mostrar distribución de una variable | Observaciones numéricas individuales | Histograma; puntos individuales si son pocos | El ancho de los intervalos cambia el histograma. No reconstruyas una distribución a partir de una media o porcentajes resumidos. |
| Comparar distribuciones entre grupos | Observaciones individuales por grupo | Puntos por grupo; cajas si el resumen es útil | Las cajas ocultan forma y tamaño de muestra. Violines o densidades solo si la estimación aporta y los datos la sostienen; no suavices muestras escasas para aparentar evidencia. |
| Mostrar relación entre dos cantidades | Dos variables numéricas medidas sobre las mismas unidades | Dispersión | La asociación no demuestra causalidad. No añadas regresión ni inferencia sin método respaldado; para solapamientos considera transparencia o agregación declarada. |
| Mostrar partes de un total | Componentes del mismo universo, sin solapamiento y con total conocido | Barras apiladas; al 100 % para comparar composición | Circular o anillo pueden servir para pocas partes claramente distinguibles de un solo total; evita usarlos para diferencias pequeñas o muchos grupos. Si importan magnitudes absolutas, no ocultes los tamaños detrás de porcentajes. |
| Mostrar frecuencias de respuesta múltiple | Conteos o porcentajes con denominador explícito | Barras independientes | No uses circular ni apilado al 100 %: las categorías pueden solaparse y sumar más de 100 %. Distingue personas de menciones. |
| Mostrar respuestas ordinales, como Likert | Frecuencias por nivel, grupo y denominador | Barras apiladas al 100 %; divergentes si existe un centro semántico claro | Conserva el orden y el tratamiento explícito de neutralidad y faltantes. No conviertas categorías en medias sin justificación metodológica. |
| Comparar muchas combinaciones | Matriz de valores comparables por fila y columna | Mapa de calor con escala y leyenda | No mezcles unidades en una escala común. Una celda ausente no equivale a cero; si se requieren cifras exactas, considera tabla. |
| Mostrar una estimación y su incertidumbre | Estimación e intervalo documentados | Puntos con intervalos; bandas para una curva | Identifica si se trata de IC, rango, DE u otra medida y su nivel cuando corresponda. No inventes intervalos ni equipares dispersión con incertidumbre de una estimación. |
| Mostrar categorías y subcategorías | Jerarquía respaldada | Árbol o esquema anidado | Treemap solo si las áreas representan cantidades pertinentes; no des tamaños cuantitativos a conceptos sin medidas. |
| Mostrar pasos y decisiones | Actividades, condiciones y orden documentados | Diagrama de flujo | No uses líneas de datos ni Sankey para un proceso sin cantidades de flujo. |
| Mostrar responsabilidades y relaciones | Componentes y vínculos respaldados | Diagrama conceptual o de arquitectura | No implica stack ni despliegue decidido. Etiqueta relaciones y no conectes todo con todo. |
| Mostrar interacciones en el tiempo | Participantes, mensajes y orden documentados | Diagrama de secuencia | No inventes sincronía, concurrencia ni tiempos de respuesta. |
| Mostrar transferencias cuantificadas | Orígenes, destinos, unidades y cantidades compatibles | Sankey cuando el reparto del flujo sea la pregunta | Sin pesos usa un diagrama de relaciones. Comprueba balances; no sugieras conservación si faltan entradas, salidas o pérdidas. |
| Comparar evidencia cualitativa | Temas, dimensiones, extractos o presencia documentados | Matriz comparativa o tabla trazable | Un mapa conceptual puede mostrar relaciones respaldadas. No conviertas frecuencia de palabras en importancia ni menciones en prevalencia de participantes. |

## 3. Reglas para no inducir a error

- **Barras:** normalmente parten de cero porque la longitud representa cantidad. Si interesa una diferencia pequeña, prefiere puntos con escala explícita antes que truncar barras.
- **Escalas:** una escala logarítmica requiere justificación y rotulación, y no admite cero ni negativos en su forma habitual. No la uses solo para mejorar el aspecto. Evita doble eje si paneles alineados permiten una lectura menos ambigua.
- **Comparabilidad:** conserva unidades, períodos y denominadores compatibles. En paneles comparativos usa la misma escala salvo que una diferencia explícita sea indispensable.
- **Datos ausentes:** no los sustituyas por cero ni unas una serie a través de ellos sin una transformación justificada y declarada.
- **Porcentajes:** identifica en el README externo base, unidad de conteo, selección múltiple y redondeo; no ajustes datos para forzar un total de 100 %.
- **Color:** secuencial para magnitud, divergente cuando existe un punto de referencia significativo y categórico para grupos sin orden. No dependas solo de color ni uses un arcoíris por decoración.
- **Formas llamativas:** radar, burbujas, nubes de palabras y 3D no son opciones por defecto. Exige una ventaja de lectura concreta; barras, puntos o tablas suelen facilitar comparaciones. En burbujas cuantitativas codifica por área, no por radio.
- **Evidencia agregada:** no simules observaciones, cajas, violines o dispersiones a partir de cifras resumidas de artículos. Conserva la distinción entre reproducción, adaptación y análisis propio.
- **Legibilidad:** verifica etiquetas, símbolos, contraste y recortes al tamaño final. Si la figura necesita interacción para entenderse, prepara una alternativa estática legible para el informe.

## 4. Cómo recomendar sin imponer

Cuando el usuario pida consejo o la representación solicitada resulte inadecuada, ofrece una recomendación principal y, solo si aporta una diferencia real, una alternativa:

- **Recomiendo:** tipo de gráfico y pregunta que resuelve.
- **Porque:** estructura concreta de los datos que lo hace adecuado.
- **Alternativa:** qué permite ver mejor y qué sacrifica.
- **Falta confirmar:** dato o decisión indispensable, solo si existe.

No cambies una figura explícitamente solicitada sin explicar el motivo. Si producirla requiere suponer contenido o alterar la metodología, consulta antes; si solo cambia la presentación respaldada, elige la opción simple y documenta la elección.

### Ejemplos de aplicación, no datos del proyecto

- Porcentajes publicados de una pregunta de selección múltiple: barras independientes, no circular; confirmar el denominador si la fuente no lo aclara.
- Duración de cada ejecución de dos métodos: comparar distribuciones; si las ejecuciones están emparejadas, representar también ese vínculo cuando sea la pregunta. Con solo medias publicadas, no generar violines.
- Pasos propuestos de revisión de requisitos: diagrama de flujo y estado de propuesta en el README; no presentarlo como evaluación realizada.
- Temas de entrevistas sin codificación cuantitativa confirmada: matriz cualitativa trazable; no asignar porcentajes ni pesos a los temas.
- Tres cifras que el lector necesita citar exactamente: tabla breve antes que un gráfico ornamental.

## 5. Fuentes de consulta

Síntesis operativa original para esta skill. Las siguientes fuentes sirven para profundizar y contrastar la selección; no se incorporan automáticamente a la bibliografía de la tesis ni respaldan datos de MASRE. Páginas consultadas durante esta actualización:

- [From Data to Viz](https://www.data-to-viz.com/): selección por estructura de datos, familias de gráficos y advertencias. Su catálogo no obliga a usar todas las variantes.
- Claus O. Wilke, [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/): guía de representación fiel y legibilidad.
- Wilke, [Visualizing proportions](https://clauswilke.com/dataviz/visualizing-proportions.html): ventajas y límites de circulares, barras agrupadas y apiladas según la pregunta.
- Wilke, [Visualizing uncertainty](https://clauswilke.com/dataviz/visualizing-uncertainty.html): representación de incertidumbre y significado de intervalos.

No copies figuras, datos o código externo sin comprobar procedencia y condiciones de uso. Para un gráfico especializado no cubierto aquí, busca documentación pertinente si es necesario; no improvises una codificación ni instales herramientas automáticamente.
