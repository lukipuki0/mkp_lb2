# Revisión visual y presentación opcional

## Comprobación de contenido

- Contrasta componentes, conexiones y orden con el texto seleccionado; verifica límites y supuestos.
- Distingue decisiones existentes de propuestas. No presentes una vista como validación de MASRE.
- Comprueba el título opcional y la ausencia por defecto de subtítulos, citas/fuentes, acotaciones de alcance y notas explicativas incrustadas. Conserva ejes, unidades y leyendas de lectura; otras anotaciones internas requieren petición explícita o semántica indispensable. Revisa en el README externo el pie, tipo de atribución y correspondencia de citas con la bibliografía; usa corchetes exclusivamente numéricos.
- Confirma que abstracciones y agrupaciones no oculten diferencias pertinentes.
- En gráficos de datos, contrasta valores y transformaciones con la fuente; comprueba tipo de gráfico, ejes, unidades, escalas y leyenda. No presentes incertidumbre ni agregaciones sin respaldo.

## Inspección visual de exportaciones

Mira cada exportación completa y al tamaño real de inserción, no solo ampliada en pantalla:

- Texto, puntas y dirección de flechas legibles; etiquetas concisas sin colisiones ni recortes.
- Flechas sin atravesar texto o cajas ajenas; puntos de entrada/salida comprensibles y rutas ajustadas cuando sea necesario.
- Título opcional separado de los primeros elementos; alineación y espacios consistentes, sin bandas vacías heredadas de notas retiradas.
- Relaciones y límites distinguibles sin depender solo del color; contraste suficiente y comprobación en escala de grises cuando el documento o su impresión lo requieran.
- Formas e iconos reconocibles, sin símbolos ausentes ni sustituciones de fuente que alteren tamaño o significado.
- Contenedores y leyendas necesarios para comprender, no ornamentales; notas explicativas fuera de la figura.
- Páginas y recortes completos; ninguna caja, etiqueta o punta de flecha perdida. Correspondencia entre partes, mapa y figura completa cuando existan.

Si queda densa, abstrae o divide preservando lo relevante. Usa orientación y distribución compatibles con el documento; no hay regla universal de verticalidad ni tamaño fijo en puntos. No supongas que todos los nodos deben conectarse si la fuente no lo afirma.

XML bien formado y pruebas unitarias solo verifican estructura/contratos. Sin renderizador disponible o autorizado, informa exactamente que faltan exportación e inspección visual al tamaño final. Entrega editable como borrador, no como figura visualmente validada.

## Presentación, solo si se solicita

Adapta al espacio real de la presentación, sin imponer editor ni resolución. Usa una figura completa si es legible; añade mapa y una vista por parte solo si facilitan explicación. El título debe comunicar la idea, y las notas externas pueden conservar detalle y procedencia sin saturar la figura. No insertes pies ni acotaciones automáticamente en el escrito. No publiques assets ni enlaces automáticamente.

## Entrega y estado

Entrega `.drawio` para diagramas; para gráficos de datos, script reproducible y referencia a los datos de entrada. Usa nombres semánticos estables e incluye solo exportaciones efectivamente generadas. En Plotly, comprueba por separado la versión interactiva y la estática si ambas se entregan. No prometas PNG/PDF inexistentes ni revisión humana no realizada. Resume:

- Archivos y ubicación junto al informe elegido.
- Pie/título propuesto, atribución y fuentes verificadas o pendientes.
- Comprobaciones estructurales y visuales realizadas por separado.
- Limitaciones y revisiones pendientes concretas.

No generes figuras reales al probar la skill; la muestra técnica no constituye evidencia del proyecto.
