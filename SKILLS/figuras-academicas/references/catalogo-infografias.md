# Catálogo operativo de las infografías aportadas

Consulta obligatoria junto con la [guía de selección](seleccion-de-graficos.md) antes de generar, adaptar o recomendar una figura. Esta traducción a reglas textuales permite utilizar las referencias sin tener que interpretar el PDF en cada ejecución. No obliga a elegir una variante vistosa: primero pregunta y datos, después representación y herramienta.

## Procedencia y alcance

- [PDF original: Infografías de gráficos estadísticos](Infografias_graficos_estadisticos.pdf): ocho páginas en español. Se extrajo su texto y se revisaron visualmente las ocho páginas. Las secciones siguientes mantienen su orden y numeración de variantes.
- [Imagen original: gráficos de visualización de datos](graficos-visualizacion-datos_1.jpg): resumen de ocho intenciones, con dos ejemplos por intención. Se conserva como referencia visual complementaria; sus imprecisiones se aclaran al final.
- Los originales fueron aportados por el usuario. No se verificaron autoría ni licencia; no se atribuyen a una fuente académica ni se incorporan a la bibliografía de la tesis.
- El PDF declara «Ejemplos con datos simulados · Formas esquemáticas · Infografías educativas». Sus cifras, relaciones y diseños no son datos ni resultados de MASRE. No los reutilices como evidencia.
- Esto es una interpretación operativa en español, no una transcripción literal ni una traducción entre idiomas. Las columnas de condiciones añaden cautelas de la skill; no deben atribuirse literalmente al material aportado.

## PDF, página 1: barras — comparar categorías

| Variante del PDF | Cuándo sirve | Condición o límite operativo |
| --- | --- | --- |
| 01 Simples | Una medida por categoría | Misma unidad y base comparable; eje cuantitativo desde cero. |
| 02 Horizontales | Etiquetas largas o ranking | Conserva el orden natural si las categorías son ordinales. |
| 03 Agrupadas | Contrastar subgrupos dentro de categorías | Si hay demasiadas series, usa paneles; no comprimas etiquetas hasta volverlas ilegibles. |
| 04 Apiladas | Leer total y composición | Los segmentos internos sin base común dificultan comparaciones precisas. |
| 05 Apiladas al 100 % | Comparar proporciones entre grupos | Componentes excluyentes de cada total; no ocultes diferencias de tamaño de grupo. |
| 06 Divergentes | Valores a ambos lados de una referencia | Cero o referencia con significado real; no inventes valores negativos para centrar categorías. |

## PDF, página 2: tortas — partes de un único total

El PDF propone pocas categorías, idealmente de 3 a 5. Es una orientación de legibilidad, no un umbral estadístico: importa que las partes sean excluyentes, completen el total y se distingan. Para comparar diferencias pequeñas, prefiere barras.

| Variante del PDF | Cuándo considerarla | Condición o alternativa preferible |
| --- | --- | --- |
| 01 Clásica | Composición simple de un total | No usar para respuestas múltiples ni comparar muchos grupos. |
| 02 Dona | Misma composición, con espacio central para el total | El hueco no aporta precisión; no colocar dentro notas innecesarias. |
| 03 Explotada | La lámina separa un sector para resaltarlo | No es opción por defecto: puede dificultar comparar áreas; prefiere color o etiqueta. |
| 04 Dona multinivel | Partes jerárquicas de un mismo universo | Validar relación padre-hijo y totales; un árbol puede explicar mejor una jerarquía sin cantidades. |
| 05 Semicircular | Composición representada en medio círculo | El semicírculo completo representa 100 %; no confundir con indicador de progreso sin base documentada. |
| 06 Con desglose | Detalle de una parte | Identifica el subtotal que se desglosa y evita confundir porcentajes del subtotal con los del total. |

## PDF, página 3: histogramas — forma de una distribución

Requieren observaciones numéricas o frecuencias por intervalos documentadas; no se reconstruyen a partir de una media. Las clases son contiguas. Para comparar distribuciones, conserva límites de clase y escalas compatibles.

| Variante del PDF | Uso | Comprobación |
| --- | --- | --- |
| 01 Simple | Cantidad de observaciones por intervalo | Declarar límites y ancho de clase; barras categóricas no son histogramas. |
| 02 Frecuencia relativa | Fracción o porcentaje por intervalo | Base explícita; no confundir porcentaje por clase con densidad. |
| 03 Densidad | Comparar forma con área total igual a 1 | El área, no la suma de alturas, representa probabilidad. Con anchos desiguales, ajustar altura por ancho. |
| 04 Superpuestos | Contrastar distribuciones | Mismas clases; transparencia o paneles si se ocultan mutuamente. |
| 05 Acumulado | Frecuencia hasta cada límite | Debe ser no decreciente; distinguir conteo de proporción acumulada. |
| 06 Bidimensional | Frecuencia de pares numéricos por celdas | Definir intervalos de ambos ejes y escala de color; no confundir celda vacía con dato faltante. |

## PDF, página 4: frecuencias — contar, ordenar y acumular

«Frecuencia» describe la medida, no un único tipo de gráfico.

| Variante del PDF | Uso | Comprobación |
| --- | --- | --- |
| 01 Barras de frecuencia | Recuentos por valor o categoría | Unidad contada y categorías documentadas. |
| 02 Polígono | Comparar perfiles de frecuencia por clases numéricas | Puntos en posiciones representativas de las clases; no conectar categorías nominales. |
| 03 Ojiva | Leer acumulados hasta límites de clase | Usar límites ordenados; no presentarla como observaciones individuales. |
| 04 Pareto | Ranking descendente con porcentaje acumulado | Misma medida aditiva no negativa; acumulado sobre total definido. No demuestra por sí solo una regla 80/20 ni prioridad causal. |
| 05 Relativa | Expresar recuentos respecto de una base | Identificar denominador y si las respuestas se solapan. |
| 06 Acumulada escalonada | Proporción de observaciones hasta cada valor | Para una distribución empírica, cada salto proviene de observaciones reales; no simularlas desde resúmenes. |

El Pareto combina barras y una línea acumulada. Es una excepción deliberada a la preferencia por evitar doble eje: identificar claramente cantidad y porcentaje; si confunde, separar en paneles alineados.

## PDF, página 5: cajas — mediana y dispersión

El boxplot resume mediana, cuartiles y posibles atípicos; no muestra toda la distribución. Requiere observaciones o un resumen estadístico suficiente y documentado. Los bigotes no son necesariamente mínimo y máximo: indicar su convención en el README externo.

| Variante del PDF | Uso | Condición |
| --- | --- | --- |
| 01 Vertical | Resumir una variable | No sustituye revisar la forma ni la muestra. |
| 02 Horizontal | Mismo resumen con otra orientación | Útil cuando las etiquetas requieren espacio. |
| 03 Por grupos | Comparar distribuciones | Misma unidad, escala y convención de bigotes. |
| 04 Con muescas | Mostrar incertidumbre aproximada de la mediana | Documentar método y supuestos; no tratar solapamientos como prueba formal automática. |
| 05 Caja + puntos | Mostrar resumen y observaciones | Preferible si permite ver tamaño de muestra y agrupamientos; desplazamiento visual no altera los valores del eje medido. |
| 06 Ancho variable | Codificar tamaño de muestra | Declarar relación entre ancho y n; no asumir proporcionalidad lineal ni fabricar n. |

## PDF, página 6: líneas — series ordenadas

| Variante del PDF | Uso | Condición |
| --- | --- | --- |
| 01 Simple | Evolución de una serie | Orden y distancia reales en el eje; no unir categorías nominales. |
| 02 Múltiples | Contrastar trayectorias | Unidades y escala comparables; paneles si las curvas se ocultan. |
| 03 Con marcadores | Identificar mediciones efectivas | No hacer pasar interpolación por observaciones. |
| 04 Escalonada | Estado constante hasta un cambio | Solo si el fenómeno cambia por saltos; declarar cuándo se aplica el cambio. |
| 05 Área | Destacar magnitud respecto de una base | Normalmente base cero; el relleno enfatiza cantidad, no solo trayectoria. |
| 06 Banda | Mostrar intervalo alrededor de una curva | Definir si es IC, dispersión, rango u otra medida y cómo se obtuvo; nunca inventarlo. |

## PDF, página 7: dispersión — pares de variables numéricas

| Variante del PDF | Uso | Condición |
| --- | --- | --- |
| 01 Simple | Relación entre X e Y de las mismas unidades | Un punto por observación; asociación no implica causalidad. |
| 02 Por grupos | Distinguir categorías en la relación | Color y, si hace falta, forma; grupos respaldados, no inferidos por estética. |
| 03 Con tendencia | Mostrar un ajuste justificado | No añadir regresión automáticamente; documentar método, alcance y supuestos. |
| 04 Burbujas | Incorporar una tercera cantidad | Codificar magnitud por área, no radio; leyenda de tamaño y valores compatibles. |
| 05 Transparencia | Reducir ocultamiento por superposición | No vuelve visibles todos los casos; considerar densidad o paneles si no alcanza. |
| 06 Matriz | Explorar pares de varias variables | Controlar densidad y legibilidad; no implica una batería de conclusiones inferenciales. |

## PDF, página 8: decisión rápida

- Comparar categorías → barras.
- Partes de un total → torta/dona si la composición es simple; barras apiladas para grupos, o independientes si interesa precisión.
- Distribución numérica → histograma; puntos para muestras pequeñas.
- Recuentos y acumulados → barras, polígono, ojiva, Pareto o acumulada según la pregunta y la estructura.
- Comparar medianas y dispersión → cajas, preferentemente con observaciones cuando estén disponibles y sean legibles.
- Evolución ordenada → líneas.
- Relación entre cantidades emparejadas → dispersión.

La regla del PDF «barras = categorías · histograma = intervalos numéricos contiguos» evita confundir ambas representaciones. Para evidencia cualitativa, incertidumbre, procesos o arquitectura, completar con la [matriz general](seleccion-de-graficos.md#2-matriz-por-intención); el PDF no cubre todos esos casos.

## Imagen: aportes y correcciones

| Intención de la imagen | Ejemplos que muestra | Interpretación operativa y cautelas |
| --- | --- | --- |
| Desviación | Barra divergente; «tabla de columna» | Divergentes para diferencias respecto de una referencia. La segunda miniatura contrapone dos componentes: no adoptar su nombre ambiguo ni exigir categorías binarias. |
| Correlación | Dispersión; «línea de tiempo» con barras y línea | Dispersión para pares numéricos. El segundo dibujo es un gráfico combinado, no una evidencia de correlación; usar solo con eje ordenado y medidas claramente identificadas, evitando doble eje innecesario. |
| Clasificación | Barras; tira de puntos | Se refiere a ranking, no a entrenar un clasificador. Barras ordenadas o puntos con eje común para comparar magnitudes. |
| Distribución | Histograma; «velas (o cajas)» | Usar intervalos contiguos en histogramas: no copiar la recomendación de dejar huecos. La miniatura muestra cajas; las velas financieras son otro gráfico y no son sinónimos. |
| Cambios en el tiempo | Línea; mapa de calor calendario | Calendario si existen fechas y mediciones por día con cobertura verificable; huecos y faltantes diferenciados. Línea para lectura de tendencia. |
| Magnitud | Columnas; Marimekko | Columnas para comparar categorías, no como sustituto de dispersión entre dos variables continuas. Marimekko combina anchos y alturas: usar solo para composición con cantidades consistentes; si se requiere precisión, preferir barras o tabla. |
| Parte de un todo | Tarta; Venn | Circular para partes excluyentes. Venn muestra pertenencia y solapamiento de conjuntos, no una partición del total; no hacer áreas proporcionales si no se calculan como tales. |
| Espacial | Coroplético; mapa de flujo | Solo si la geografía es parte de la pregunta y hay ubicaciones fiables. Coropletas: preferir tasas comparables cuando los totales reflejen tamaño poblacional; distinguir ausencia de datos. Flujos: origen/destino documentados y pesos reales si el ancho codifica cantidad. |

No copies automáticamente la paleta, cabeceras, tarjetas, órbitas decorativas, pies ni textos del material. Las infografías enseñan variantes; una figura académica debe seguir las reglas de entrega de la skill y ser legible al tamaño final.

## Protocolo de uso en cada figura

1. Leer la guía general y este catálogo; identificar la intención y una familia candidata. Para diagramas fuera del PDF, usar la matriz general y las cautelas aplicables.
2. Comprobar que existen los datos exigidos por la variante. No inventar observaciones, totales, relaciones ni intervalos para encajar en el ejemplo visual.
3. Si hay una alternativa más adecuada, explicar brevemente qué mejora y qué sacrifica. Si el usuario pidió un tipo inadecuado, señalarlo antes de sustituirlo; consultar solo si falta una decisión sustantiva.
4. Registrar en el README externo: pregunta, familia/variante, motivo, datos y límites. Añadir página del PDF o sección de esta guía si se usó como referencia; no convertirlo en cita académica.
5. Elegir la herramienta disponible y verificar exportación y legibilidad según el flujo. Abrir los originales cuando se necesite comparar detalles visuales; no es necesario renderizar de nuevo las ocho páginas en cada uso.
